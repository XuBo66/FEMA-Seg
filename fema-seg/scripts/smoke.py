"""Generate synthetic data and execute real training/evaluation/inference."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import yaml


def main():
    root=Path('runs/smoke_data').resolve();root.mkdir(parents=True,exist_ok=True);rows=[]
    for i in range(12):
        y=np.zeros((32,32),np.uint8);y[4:16,4+i%4:20]=1;y[20:]=2
        rgb=np.array([[40,70,120],[50,180,30],[140,140,140]],np.uint8)[y]
        Image.fromarray(rgb).save(root/f'{i}.png');Image.fromarray(y).save(root/f'{i}_label.png');np.save(root/f'{i}.npy',np.ones_like(y,dtype='float32'))
        rows.append({'id':str(i),'stratum':str(i%2),'image':str(root/f'{i}.png'),'label':str(root/f'{i}_label.png'),'pseudo':str(root/f'{i}_label.png'),'weight':str(root/f'{i}.npy')})
    for name,subset in [('train',rows[:8]),('val',rows[8:])]:
        (root/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in subset))
    (root/'subset.json').write_text(json.dumps({'selected_ids':['0','1']}))
    cfg=yaml.safe_load(Path('configs/cityscapes.yaml').read_text());cfg.update(device='cpu',output='runs/smoke',vegetation=[1],train_manifest=str(root/'train.jsonl'),val_manifest=str(root/'val.jsonl'),subset=str(root/'subset.json'))
    cfg['model'].update(classes=3,dims=[8,12,16,24],depths=[2,2,2,2],state=2,context_dim=8,context_depth=3,heads=2,bridges=[1,2,3],backend='reference')
    cfg['train'].update(epochs=2,fine_epochs=1,warmup=1,crop=[32,32],amp=False,steps_per_epoch=1)
    dest=root/'config.yaml';dest.write_text(yaml.safe_dump(cfg))
    commands=[['train.py','--config',str(dest),'--stop-after','1'],['train.py','--config',str(dest),'--resume','runs/smoke/last.pt'],
              ['evaluate.py','--checkpoint','runs/smoke/best.pt','--manifest',str(root/'val.jsonl'),'--device','cpu','--output','runs/smoke/evaluation.json'],
              ['infer.py','--checkpoint','runs/smoke/best.pt','--image',rows[8]['image'],'--device','cpu','--output','runs/smoke/pred.png']]
    for cmd in commands:subprocess.run([sys.executable,*cmd],check=True)
    print('Synthetic pipeline passed; these are NOT paper benchmark results.')


if __name__=='__main__':main()
