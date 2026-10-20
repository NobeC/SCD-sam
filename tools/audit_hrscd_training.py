"""Small train/val input audit; no test access and no learned statistics."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets.hrscd import HRSCD, PALETTE, CLASSES, MEAN, STD

root=Path('datasets/HRSCD_117_29_145')
out=Path('diagnostics/hrscd_supervised_20261010');out.mkdir(parents=True,exist_ok=True)
summary={};selected={};pictures=[]
for split,count in [('train',64),('val',8)]:
    ds=HRSCD(root,split)
    records=[]
    for i in np.linspace(0,len(ds)-1,count,dtype=int):
        arrays=ds.read_raw(int(i))
        sample=ds[int(i)]
        restored=np.rint(sample['A'].numpy().transpose(1,2,0)*STD+MEAN).clip(0,255).astype(np.uint8)
        assert np.array_equal(restored,arrays[0])
        records.append({'name':ds.names[i],'change_pixels':int(arrays[4].sum()),
                        'classes_A':np.unique(arrays[2]).tolist(),'classes_B':np.unique(arrays[3]).tolist()})
    # Smoke samples chosen only from train; val keeps a fixed evenly-spaced subset.
    chosen=sorted(records,key=lambda r:r['change_pixels'],reverse=True)[:2] if split=='train' else records[:2]
    selected[split]=[r['name'] for r in chosen]
    summary[split]={'audited_count':count,'records':records}
    for row in chosen:
        pictures.append((split,row,ds.read_raw(ds.names.index(row['name']))))
fig,axs=plt.subplots(len(pictures),5,figsize=(15,3*len(pictures)))
for row,(split,rec,arrays) in enumerate(pictures):
    for col,array in enumerate(arrays):
        ax=axs[row,col]
        if col in (2,3): ax.imshow(PALETTE[array])
        elif col==4: ax.imshow(array,cmap='gray',vmin=0,vmax=1)
        else: ax.imshow(array)
        ax.set_title(['A (RGB)','B (RGB)','Semantic A','Semantic B','Change'][col]);ax.axis('off')
    axs[row,0].set_ylabel(split+' / '+rec['name'])
fig.suptitle('HRSCD audit: rows train x2, val x2; labels preserved (0 unknown)',fontsize=14)
fig.tight_layout();fig.savefig(out/'label_audit.png',dpi=120);plt.close(fig)
(out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'smoke_samples.json').write_text(json.dumps(selected,indent=2)+'\n')
print('AUDIT_COMPLETE',selected,flush=True)
