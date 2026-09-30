"""Convert labels from explicit manifests; never infer train/val splits."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

CITY_IDS=[7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
UAV_COLORS=[[0,0,0],[128,0,0],[128,64,128],[0,128,0],[128,128,0],[64,0,128],[192,0,192],[64,64,0]]
CAM_COLORS=[[128,128,128],[128,0,0],[192,192,128],[128,64,128],[0,0,192],[128,128,0],[192,128,128],[64,64,128],[64,0,128],[64,64,0],[0,128,192]]


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--format',choices=['cityscapes','camvid','uavid','trainids'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();src=Path(a.manifest).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);rows=[]
    for line in src.read_text().splitlines():
        r=json.loads(line);image=(src.parent/r['image']).resolve();label=(src.parent/r['label']).resolve()
        source=Image.open(label)
        if a.format in ['camvid','uavid']: source=source.convert('RGB')
        y=np.array(source);converted=np.full(y.shape[:2],255,np.uint8)
        if a.format=='trainids':
            if y.ndim!=2:raise ValueError('Expected single-channel train IDs')
            converted=y.astype(np.uint8)
        elif a.format=='cityscapes':
            if y.ndim!=2:raise ValueError('Expected Cityscapes labelIds, not RGB')
            for i,v in enumerate(CITY_IDS):converted[y==v]=i
        else:
            if y.ndim!=3:raise ValueError('Expected RGB color masks')
            colors=CAM_COLORS if a.format=='camvid' else UAV_COLORS
            for i,v in enumerate(colors):converted[np.all(y[:,:,:3]==v,axis=-1)]=i
        dest=out/f'{len(rows):06d}.png';Image.fromarray(converted).save(dest)
        rows.append(dict(r,image=str(image),label=str(dest)))
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


if __name__=='__main__':main()
