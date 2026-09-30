import cv2
import numpy as np
import torch
from torch.nn import functional as F


def canny_labels(target):
    """Canny per binary class avoids arbitrary numeric label gradients."""
    result=[]
    for y in target.detach().cpu().numpy():
        edge=np.zeros(y.shape,np.uint8)
        for c in np.unique(y):
            if c!=255: edge|=cv2.Canny((y==c).astype(np.uint8)*255,100,200)
        valid=cv2.erode((y!=255).astype(np.uint8),np.ones((3,3),np.uint8))
        result.append((edge>0)*valid)
    return torch.as_tensor(np.array(result),device=target.device,dtype=torch.float32)


def semantic(logits,y,weight):
    valid=y!=255; safe=y.masked_fill(~valid,0)
    p=logits.float().softmax(1); chosen=p.gather(1,safe[:,None]).squeeze(1).clamp_min(1e-7)
    w=weight*valid
    focal=(-(1-chosen).square()*chosen.log()*w).sum()/w.sum().clamp_min(1)
    one=F.one_hot(safe,logits.shape[1]).permute(0,3,1,2).float()
    inter=(p*one*w[:,None]).sum((0,2,3))
    denom=((p+one)*w[:,None]).sum((0,2,3))
    dice=(1-(2*inter+1e-6)/(denom+1e-6)).mean() if valid.any() else logits.sum()*0
    return focal+.5*dice


def objective(output,y,weight,gamma):
    loss=semantic(output['logits'],y,weight)
    edges=canny_labels(y)
    ph,pw=output['padded_size'];h,w=y.shape[-2:]
    yp=F.pad(y,(0,pw-w,0,ph-h),value=255)
    wp=F.pad(weight,(0,pw-w,0,ph-h))
    ep=F.pad(edges,(0,pw-w,0,ph-h))
    aux_loss=loss*0; edge_loss=loss*0
    for a in output['aux']:
        size=a['logits'].shape[-2:]
        ya=F.interpolate(yp[:,None].float(),size=size,mode='nearest')[:,0].long()
        wa=F.interpolate(wp[:,None],size=size,mode='nearest')[:,0]
        aux_loss=aux_loss+semantic(a['logits'],ya,wa)
        if 'indices' in a:
            e=F.adaptive_max_pool2d(ep[:,None],size)[:,0].flatten(1).gather(1,a['indices'])
            v=(ya!=255).flatten(1).gather(1,a['indices'])
            edge_loss=edge_loss+(F.binary_cross_entropy_with_logits(a['edge'].float(),e,reduction='none')*v).sum()/v.sum().clamp_min(1)
    # Auxiliary semantic supervision is necessary to train top-k classifier heads;
    # its coefficient is an explicit reconstruction assumption.
    return loss+.2*aux_loss/len(output['aux'])+gamma*edge_loss/len(output['aux'])
