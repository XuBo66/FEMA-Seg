import cv2
import numpy as np


def boundary(mask,d=3):
    padded=np.pad(mask.astype(np.uint8),1)
    eroded=cv2.erode(padded,np.ones((3,3),np.uint8),iterations=d)[1:-1,1:-1]
    return mask.astype(bool)&~eroded.astype(bool)


class Metrics:
    def __init__(self,classes,vegetation,d=3):
        self.c,self.veg,self.d=classes,vegetation,d
        self.cm=np.zeros((classes,classes),np.int64)
        self.bi=np.zeros(classes);self.bu=np.zeros(classes);self.gvi=[]

    def update(self,pred,target):
        if pred.shape!=target.shape or target.ndim!=2: raise ValueError('Expected aligned 2D masks')
        valid=target!=255
        if not valid.any(): return
        if np.any((target[valid]<0)|(target[valid]>=self.c)): raise ValueError('Invalid target ID')
        if np.any((pred<0)|(pred>=self.c)): raise ValueError('Invalid prediction ID')
        self.cm+=np.bincount((target[valid]*self.c+pred[valid]).ravel(),minlength=self.c**2).reshape(self.c,self.c)
        # Exclude a d-pixel halo around void to avoid invented void boundaries.
        interior=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8),iterations=self.d).astype(bool)
        for c in range(self.c):
            p=boundary((pred==c)&valid,self.d)&interior
            t=boundary((target==c)&valid,self.d)&interior
            self.bi[c]+=(p&t).sum();self.bu[c]+=(p|t).sum()
        pg=np.isin(pred[valid],self.veg).mean();tg=np.isin(target[valid],self.veg).mean()
        self.gvi.append((pg-tg)**2)

    def result(self):
        tp=np.diag(self.cm);union=self.cm.sum(0)+self.cm.sum(1)-tp;row=self.cm.sum(1)
        def ratios(a,b): return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b>0)
        iou,acc,biou=ratios(tp,union),ratios(tp,row),ratios(self.bi,self.bu)
        def mean(x): return float(np.nanmean(x)) if np.isfinite(x).any() else None
        rmse=float(np.sqrt(np.mean(self.gvi))) if self.gvi else None
        return {'mIoU':mean(iou),'mPA':mean(acc),'mBIoU':mean(biou),
                'GVI_RMSE_fraction':rmse,'GVI_RMSE_percentage_points':None if rmse is None else 100*rmse,
                'per_class_IoU':[float(v) if np.isfinite(v) else None for v in iou],
                'confusion':self.cm.tolist(),'images':len(self.gvi),'boundary_pixels':self.d}
