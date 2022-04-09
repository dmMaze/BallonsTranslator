import sys, os
import os.path as osp
sys.path.append(os.getcwd())

from dl import OCRBase, OCRMIT32px
from utils.io_utils import imread, imwrite

from ui.misc import ProjImgTrans

if __name__ == '__main__':
    setup_params = OCRMIT32px.setup_params
    setup_params['device']['delect'] = 'cpu'
    ocr = OCRMIT32px(**setup_params)
    img_path = r'data/testpacks/textline/ballontranslator.png'
    img = imread(img_path)
    rst = ocr.run_ocr(img)

    proj_dir = r'data/testpacks/manga'
    proj = ProjImgTrans(proj_dir)
    for imgname in proj.pages:
        img_path = osp.join(proj.directory, imgname)
        img = imread(img_path)
        blk_list = proj.pages[imgname]
        ocr.ocr_blk_list(img, blk_list)
    proj.save()

