import os
import time
import copy
import random
import numpy as np
import argparse
import torch.nn as nn
import torch.autograd
from skimage import io
from torch import optim
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

working_path = os.path.dirname(os.path.abspath(__file__))

from utils.loss import CrossEntropyLoss2d
from utils.utils import accuracy, SCDD_eval_all, AverageMeter
from utils.common import ScaleInOutput
from utils.eval_checkpoint import resolve_checkpoint_path
from datasets import RS_ST as RS
from main_model import ChangeDetection

NET_NAME = 'SAM'
#DATA_NAME = 'SECOND'
DATA_NAME = 'Landsat' 


args = {
    'train_batch_size': 2,
    'val_batch_size': 2,
    'lr': 0.1,
    'epochs': 100,
    'gpu': True,
    'psd_TTA': True,
    'lr_decay_power': 1.5,
    'train_crop_size': False,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 100,
    'predict_step': 5,
    'pseudo_thred': 0.8,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME)
}
###############################################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
writer = SummaryWriter(args['log_dir'])
scale = ScaleInOutput(512)

def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('student', 'state_dict', 'model', 'net'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    return checkpoint


def _strip_module_prefix(state_dict):
    return {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }


def _infer_num_classes(state_dict):
    for key in ('head.head1.2.weight', 'head.head2.2.weight'):
        value = state_dict.get(key)
        if value is not None and value.dim() == 4:
            return value.shape[0]
    return None


def load_checkpoint_safely(net, pth_path):
    checkpoint = torch.load(pth_path, map_location='cpu')
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))

    model_state = net.state_dict()
    matched_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched_state[key] = value
        else:
            skipped.append(key)

    missing, unexpected = net.load_state_dict(matched_state, strict=False)
    print('Loaded checkpoint: {}'.format(pth_path))
    print('Matched params: {}/{}'.format(len(matched_state), len(model_state)))
    if skipped:
        print('Skipped {} params with missing keys or shape mismatch, e.g. {}'.format(
            len(skipped), skipped[:4]))
    if missing:
        print('Missing model params after loading: {}'.format(len(missing)))
    if unexpected:
        print('Unexpected checkpoint params after loading: {}'.format(len(unexpected)))
    return state_dict


def main(opt):
    opt.pth_path = resolve_checkpoint_path(opt.pth_path, working_path)
    checkpoint = torch.load(opt.pth_path, map_location='cpu')
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
    checkpoint_num_classes = _infer_num_classes(state_dict)
    if checkpoint_num_classes is not None and checkpoint_num_classes != opt.num_classes:
        print('Checkpoint semantic head has {} classes; overriding opt.num_classes={}.'.format(
            checkpoint_num_classes, opt.num_classes))
        opt.num_classes = checkpoint_num_classes

    net = ChangeDetection(opt).cuda()
    load_checkpoint_safely(net, opt.pth_path)
    net.eval()
    val_set = RS.Data('val')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], shuffle=False)
    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    validate(val_loader, net, criterion, 1)


def validate(val_loader, net, criterion, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels_A, labels_B = data
        if args['gpu']:
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()

        with torch.no_grad():
            imgs_A, imgs_B = scale.scale_input((imgs_A, imgs_B)) 
            outs = net(imgs_A, imgs_B)
            outs = scale.scale_output(outs)
            outputs_A, outputs_B, out_change  = outs

            # loss_A = criterion(outputs_A, labels_A)
            # loss_B = criterion(outputs_B, labels_B)
            # loss = loss_A * 0.5 + loss_B * 0.5
        # val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A * change_mask.squeeze().long()).numpy()
        preds_B = (preds_B * change_mask.squeeze().long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)

        if curr_epoch % args['predict_step'] == 0 and vi == 0:
            pred_A_color = RS.Index2Color(preds_A[0])
            pred_B_color = RS.Index2Color(preds_B[0])
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + '_A.png'), pred_A_color)
            io.imsave(os.path.join(args['pred_dir'], NET_NAME + '_B.png'), pred_B_color)
            print('Prediction saved!')

    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes)

    curr_time = time.time() - start
    print('%.1fs Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f'\
    %(curr_time,  Fscd*100, IoU_mean*100, Sek*100, acc_meter.average()*100))

    return Fscd, IoU_mean, Sek, acc_meter.avg


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Change Detection train')
    parser.add_argument("--backbone", type=str, default="msam_96")
    parser.add_argument("--neck", type=str, default="fpn+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")
    parser.add_argument("--pretrain", type=str, default="")  
    parser.add_argument("--cuda", type=str, default="1")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=RS.num_classes)

    parser.add_argument("--learning-rate", type=float, default=0.00035)
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    parser.add_argument(
        '--pth_path',
        required=False,
        default='checkpoints/Landsat/'
                'Landsat_pseudo_graph_stage1_seed42_epoch089_Fscd0.8323.pth',
        help='Absolute path, or a path relative to the project root.',
    )

    opt = parser.parse_args()

    main(opt)
