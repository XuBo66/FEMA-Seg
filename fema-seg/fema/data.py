import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


def records(path):
    path=Path(path).resolve()
    rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    for r in rows:
        for key in ('image','label','pseudo','weight'):
            if r.get(key): r[key]=str((path.parent/r[key]).resolve())
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate sample id')
    return rows


class SegData(Dataset):
    def __init__(self,rows,classes,crop=None,augment=False,kind='label'):
        self.rows,self.classes,self.crop,self.augment,self.kind=rows,classes,crop,augment,kind

    def __len__(self): return len(self.rows)

    def __getitem__(self,i):
        r=self.rows[i]; image=Image.open(r['image']).convert('RGB')
        target=np.array(Image.open(r[self.kind]),dtype=np.int64)
        if target.ndim!=2: raise ValueError('Labels must be single channel train IDs; run prepare first')
        if not np.all((target==255)|((target>=0)&(target<self.classes))): raise ValueError('Invalid train ID')
        weight=np.load(r['weight']).astype('float32') if self.kind=='pseudo' else np.ones(target.shape,dtype='float32')
        if image.size!=target.shape[::-1] or weight.shape!=target.shape: raise ValueError('Image/mask/weight alignment error')
        mask=Image.fromarray(target.astype('uint8')); weights=Image.fromarray(weight)
        if self.augment:
            scale=random.uniform(.5,2)
            size=tuple(max(1,round(v*scale)) for v in image.size)
            image=image.resize(size,Image.Resampling.BILINEAR)
            mask=mask.resize(size,Image.Resampling.NEAREST);weights=weights.resize(size,Image.Resampling.NEAREST)
            if random.random()<.5:
                image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask=mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT);weights=weights.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image=ImageEnhance.Brightness(image).enhance(random.uniform(.8,1.2))
            image=ImageEnhance.Contrast(image).enhance(random.uniform(.8,1.2))
        if self.crop:
            ch,cw=self.crop; w,h=image.size
            canvas=Image.new('RGB',(max(w,cw),max(h,ch)));canvas.paste(image);image=canvas
            canvas=Image.new('L',image.size,255);canvas.paste(mask);mask=canvas
            canvas=Image.new('F',image.size,0);canvas.paste(weights);weights=canvas
            x=random.randint(0,image.width-cw) if self.augment else 0
            y=random.randint(0,image.height-ch) if self.augment else 0
            box=(x,y,x+cw,y+ch);image=image.crop(box);mask=mask.crop(box);weights=weights.crop(box)
        x=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
        x=(x-x.new_tensor([.485,.456,.406])[:,None,None])/x.new_tensor([.229,.224,.225])[:,None,None]
        return x,torch.from_numpy(np.asarray(mask).astype('int64')),torch.from_numpy(np.asarray(weights).copy()),r['id']
