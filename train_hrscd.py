"""Matched supervised HRSCD experiments. Never accesses test during training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.hrscd import HRSCD
from utils.hrscd_training import supervised_loss, Metrics


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) and k != 'geometry' else v for k,v in batch.items()}


def atomic_save(state, path):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp); os.replace(temp, path)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval(); metrics = Metrics(); losses = []
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast('cuda', enabled=amp):
            outputs = model(batch['A'], batch['B'])
            loss, _ = supervised_loss(outputs, batch)
        if not torch.isfinite(loss) or not all(torch.isfinite(x).all() for x in outputs):
            raise FloatingPointError('Nonfinite validation')
        metrics.update(outputs, batch); losses.append((float(loss), len(batch['name'])))
    result = metrics.result()
    result['loss'] = sum(l*n for l,n in losses) / sum(n for _,n in losses)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='datasets/HRSCD_117_29_145')
    p.add_argument('--output', required=True)
    p.add_argument('--graph', action='store_true')
    p.add_argument('--cache', default='sam_mask_cache/HRSCD_supervised_p32c0')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--effective-batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=.00035)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--sample-list', help='JSON containing train/val sample names, smoke only')
    p.add_argument('--resume')
    p.add_argument('--no-amp', action='store_true')
    opt = p.parse_args()
    if opt.sample_list and not opt.smoke_steps:
        p.error('Sample restriction is for smoke tests only')
    if opt.effective_batch_size % opt.batch_size or opt.epochs < 1:
        p.error('Invalid epochs or accumulation')
    out = Path(opt.output)
    if out.exists() and any(out.iterdir()) and not opt.resume:
        raise FileExistsError('Use a new run directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    torch.set_num_threads(4); seed_all(opt.seed)
    torch.backends.cudnn.benchmark = False
    selected = json.loads(Path(opt.sample_list).read_text()) if opt.sample_list else {}
    train = HRSCD(opt.root, 'train', augment=not bool(opt.smoke_steps), names=selected.get('train'))
    val = HRSCD(opt.root, 'val', names=selected.get('val'))
    gen = torch.Generator().manual_seed(opt.seed)
    loader = DataLoader(train,batch_size=opt.batch_size,shuffle=True,num_workers=opt.workers,
                        pin_memory=True,worker_init_fn=worker_seed,generator=gen)
    validation = DataLoader(val,batch_size=opt.batch_size,num_workers=opt.workers,pin_memory=True)
    from main_model import ChangeDetection
    model_opt = argparse.Namespace(backbone='msam_96',neck='fpn+drop',head='fcn',pretrain='',
                                   num_classes=6,input_size=512,gradient_checkpointing=True)
    model = ChangeDetection(model_opt).to(device)
    initial_hash = hashlib.sha256()
    for v in model.state_dict().values():
        initial_hash.update(v.detach().cpu().numpy().tobytes())
    optimizer = torch.optim.AdamW([
        {'params':[v for k,v in model.named_parameters() if v.requires_grad and 'backbone' in k],'lr':opt.lr/10},
        {'params':[v for k,v in model.named_parameters() if v.requires_grad and 'backbone' not in k],'lr':opt.lr}],weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt.epochs)
    graph = masks = None
    if opt.graph:
        from utils.hrscd_graph import SupervisedGraph, TrainingMasks
        graph = SupervisedGraph(model.backbone)
        masks = TrainingMasks(opt.root,opt.cache,device)
    seed_all(opt.seed)
    scaler = torch.cuda.amp.GradScaler(enabled=not opt.no_amp, init_scale=128.0)
    config = vars(opt).copy()
    config.update(initial_model_sha256=initial_hash.hexdigest(),train_pairs=len(train),val_pairs=len(val),
                  split_sha256=hashlib.sha256((Path(opt.root)/'original_pairs.csv').read_bytes()).hexdigest(),
                  normalization='RGB MobileSAM declared mean/std',semantic_ignore=0,change_target='independent change PNG',
                  graph_policy='supervised prototypes/edges/GAT active from epoch 0; no EMA/pseudo',
                  checkpoint_policy='atomic last and best Fscd only')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    start, step, best = 0, 0, -1.
    if opt.resume:
        ck = torch.load(opt.resume,map_location='cpu',weights_only=False)
        for k in ('graph','seed','lr','epochs','batch_size','effective_batch_size','split_sha256'):
            if ck['config'][k] != config[k]: raise ValueError('Resume mismatch: '+k)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        if graph: graph.load_state_dict(ck['graph'])
        start,step,best=ck['epoch'],ck['step'],ck['best']
        random.setstate(ck['rng']['python']); np.random.set_state(ck['rng']['numpy'])
        torch.set_rng_state(ck['rng']['torch']); torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        gen.set_state(ck['rng']['loader'])
        del ck
    print('CONFIG',json.dumps(config),flush=True)
    smoke_history=[]; begin=time.time()
    for epoch in range(start,opt.epochs):
        model.train()
        if graph: graph.train()
        optimizer.zero_grad(set_to_none=True)
        if graph: graph.zero_grad()
        accum=1 if opt.smoke_steps else opt.effective_batch_size//opt.batch_size
        for i,batch in enumerate(loader):
            batch=to_device(batch,device); graph_stats={}
            region_masks = None
            if graph:
                # Cache generation must not change the experiment's training RNG.
                with torch.random.fork_rng(devices=[device.index or 0]):
                    region_masks=[masks.get(n,g) for n,g in zip(batch['name'],batch['geometry'])]
            with torch.autocast('cuda',enabled=not opt.no_amp):
                if graph:
                    outputs,features=model(batch['A'],batch['B'],return_graph_features=True,detach_graph_features=False)
                else: outputs=model(batch['A'],batch['B'])
                loss,parts=supervised_loss(outputs,batch)
                if graph:
                    with torch.random.fork_rng(devices=[device.index or 0]):
                        torch.manual_seed(opt.seed+step)
                        gl,graph_stats=graph.loss(features,region_masks,batch)
                    if opt.smoke_steps and step==0:
                        gg=torch.autograd.grad(gl,features,retain_graph=True,allow_unused=True)
                        graph_stats['backbone_graph_grad_l1']=sum(float(g.abs().sum()) for g in gg if g is not None)
                    loss=loss+gl
            if not torch.isfinite(loss) or not all(torch.isfinite(x).all() for x in outputs):
                raise FloatingPointError('Nonfinite loss/output')
            group_start=(i//accum)*accum
            group_size=min(accum,len(loader)-group_start)
            scaler.scale(loss/group_size).backward()
            grad_norm=None
            if (i+1)%accum==0 or i+1==len(loader):
                scaler.unscale_(optimizer)
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if graph:
                    scaler.unscale_(graph.optimizer)
                    graph_norm=torch.nn.utils.clip_grad_norm_(list(graph.parameters()),1.)
                if not torch.isfinite(grad_norm) or (graph and not torch.isfinite(graph_norm)):
                    raise FloatingPointError('Nonfinite gradients; stopped without corrupting checkpoint')
                scaler.step(optimizer)
                if graph: scaler.step(graph.optimizer)
                scaler.update(); optimizer.zero_grad(set_to_none=True)
                if graph: graph.zero_grad()
            step+=1
            record=dict(epoch=epoch,step=step,batch=i+1,batches=len(loader),loss=float(loss),
                        **{k:float(v.detach()) for k,v in parts.items()},**graph_stats)
            if grad_norm is not None: record['grad_norm']=float(grad_norm)
            if opt.smoke_steps: smoke_history.append(record)
            if opt.smoke_steps or i==0 or (i+1)%50==0:
                record['elapsed_s']=round(time.time()-begin,2)
                print('TRAIN',json.dumps(record),flush=True)
            if opt.smoke_steps and step>=opt.smoke_steps: break
        if opt.smoke_steps and step<opt.smoke_steps: continue
        metrics=evaluate(model,validation,device,not opt.no_amp)
        print('VAL',epoch,json.dumps(metrics),flush=True)
        if opt.smoke_steps:
            smoke_eval=evaluate(model,DataLoader(train,batch_size=opt.batch_size),device,not opt.no_amp)
            (out/'smoke.json').write_text(json.dumps({'history':smoke_history,'val':metrics,'train':smoke_eval,
                'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3},indent=2)+'\n')
            print('SMOKE_COMPLETE',flush=True); return
        score=metrics['Fscd'] if metrics['Fscd'] is not None else -1.
        improved=score>best; best=max(best,score); scheduler.step()
        payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(),epoch=epoch+1,step=step,best=best,config=config,
                     graph=graph.state_dict() if graph else None,
                     rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                          'cuda':torch.cuda.get_rng_state_all(),'loader':gen.get_state()})
        atomic_save(payload,out/'last.pth')
        if improved: atomic_save(payload,out/'best.pth')
        with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps({'epoch':epoch,**metrics})+'\n')
    print('TRAINING_COMPLETE',flush=True)


if __name__=='__main__':
    main()
