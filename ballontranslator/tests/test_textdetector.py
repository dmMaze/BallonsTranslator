import sys, os
import os.path as osp
sys.path.append(os.getcwd())

from dl import ComicTextDetector
from utils.io_utils import imread, imwrite
from ui.imgtrans_proj import ProjImgTrans

if __name__ == '__main__':

    test_dir = r'data/testpacks/manga'
    mask_dir = osp.join(test_dir, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    proj = ProjImgTrans(test_dir)

    params = ComicTextDetector.params
    params['device']['select'] = 'cuda'
    ctd = ComicTextDetector(**params)

    for imgname in proj.pages:
        img_path = osp.join(proj.directory, imgname)
        img = imread(img_path)
        mask, blk_lst = ctd.detect(img)
        imwrite(osp.join(mask_dir, imgname), mask)
        proj.pages[imgname] = blk_lst
    proj.save()