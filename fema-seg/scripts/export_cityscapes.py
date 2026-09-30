import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    mapping=np.array([7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33],np.uint8)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for path in Path(a.input).rglob('*.png'):
        if '.overlay.' in path.name:continue
        mask=np.array(Image.open(path))
        if mask.ndim!=2 or mask.max()>18:raise ValueError('Expected train IDs 0..18')
        Image.fromarray(mapping[mask]).save(out/path.name.replace('_leftImg8bit','_pred'))


if __name__=='__main__':main()
