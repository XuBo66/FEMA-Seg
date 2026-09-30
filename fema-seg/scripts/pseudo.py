"""SAM geometry distillation from DINO boxes in target train-ID space."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def merge(shape,detections,masks,background):
    labels=np.full(shape,background,np.uint8);confidence=np.ones(shape,np.float32)
    best=np.full(shape,-1.,np.float32)
    for d,m in zip(detections,masks):
        choose=m&(d['score']>best) # Equal scores: stable first box wins.
        labels[choose]=d['class_id'];confidence[choose]=d['score'];best[choose]=d['score']
    confidence[labels==255]=0
    return labels,confidence


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--detections',required=True);p.add_argument('--output',required=True)
    p.add_argument('--sam-checkpoint');p.add_argument('--sam-type',default='vit_h');p.add_argument('--device',default='cuda');p.add_argument('--threshold',type=float,default=.45)
    p.add_argument('--classes',type=int,required=True);p.add_argument('--background',type=int,required=True,help='Explicit train ID, or 255 to ignore uncovered pixels (protocol deviation)')
    p.add_argument('--boxes-only',action='store_true');a=p.parse_args()
    if a.background!=255 and not 0<=a.background<a.classes:raise ValueError('Invalid background')
    if not a.boxes_only:
        if not a.sam_checkpoint:raise ValueError('SAM checkpoint required')
        from segment_anything import sam_model_registry,SamPredictor
        predictor=SamPredictor(sam_model_registry[a.sam_type](checkpoint=a.sam_checkpoint).to(a.device).eval())
    src=Path(a.manifest).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    detections=json.loads(Path(a.detections).read_text());rows=[]
    for line in src.read_text().splitlines():
        row=json.loads(line)
        for key in ['image','label']:
            if row.get(key):row[key]=str((src.parent/row[key]).resolve())
        image=np.array(Image.open(row['image']).convert('RGB'));h,w=image.shape[:2]
        if row['id'] not in detections:raise ValueError('Missing detections for '+row['id'])
        ds=[d for d in detections[row['id']] if d['score']>=a.threshold];masks=[]
        if not a.boxes_only:predictor.set_image(image)
        for d in ds:
            if not 0<=d['class_id']<a.classes or not 0<=d['score']<=1:raise ValueError('Invalid detection')
            box=np.asarray(d['box_xyxy'],dtype=np.float32)
            if box.shape!=(4,) or box[2]<=box[0] or box[3]<=box[1]:raise ValueError('Invalid box')
            if a.boxes_only:
                mask=np.zeros((h,w),bool);x1,y1,x2,y2=box.round().astype(int)
                mask[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]=True
            else:
                logits,_,_=predictor.predict(box=box,multimask_output=True,return_logits=True)
                # Stability = intersection/union of masks at +/-1 around threshold.
                threshold=predictor.model.mask_threshold
                stable=(logits>threshold+1).sum((1,2))/np.maximum((logits>threshold-1).sum((1,2)),1)
                mask=logits[stable.argmax()]>threshold
            masks.append(mask)
        label,weight=merge((h,w),ds,masks,a.background)
        stem=f'{len(rows):06d}';Image.fromarray(label).save(out/f'{stem}.png');np.save(out/f'{stem}.npy',weight)
        row.update(pseudo=str(out/f'{stem}.png'),weight=str(out/f'{stem}.npy'));rows.append(row)
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'provenance.json').write_text(json.dumps(vars(a),indent=2))


if __name__=='__main__':main()
