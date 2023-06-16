import sys, os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))

from dl import InpainterBase, AOTInpainter, PatchmatchInpainter
from utils.io_utils import imread, imwrite, find_all_imgs
from ui.imgtrans_proj import ProjImgTrans
from ui.constants import PROGRAM_PATH
os.chdir(PROGRAM_PATH)

import numpy as np
import cv2
from tqdm import tqdm

SAVE_DIR = 'tmp/inpaint_test'

def test_inpainter(inpainter: InpainterBase, proj: ProjImgTrans, inpaint_by_block=False, show=False):

    os.makedirs(SAVE_DIR, exist_ok=True)
    print('Testing inpainter:', inpainter.__class__.__name__)
    print('Inpainter params:', inpainter.params)
    inpainter.inpaint_by_block = inpaint_by_block
    print('inpaint by block: ', inpainter.inpaint_by_block)
    time_cost = 0
    for page_name in tqdm(proj.pages):
        blk_list = proj.pages[page_name]
        proj.set_current_img(page_name)
        img, mask = proj.img_array, proj.mask_array

        t0 = cv2.getTickCount()
        inpainted = inpainter.inpaint(img, mask, blk_list)
        time_cost += (cv2.getTickCount() - t0) / cv2.getTickFrequency()

        if show:
            cv2.imshow('img', img)
            cv2.imshow('mask', mask)
            cv2.imshow('inpainted', inpainted)
            cv2.waitKey(0)
        imwrite('tmp/inpaint_test/{}_inpainted.png'.format(page_name), inpainted)

    print(f'Time cost: {time_cost}, avg: {time_cost / len(proj.pages)}')


def test_aot(proj: ProjImgTrans, device: str = 'cpu', inpaint_size: int = 1024, inpaint_by_block=True, show=False):
    
    params = AOTInpainter.params
    params['device']['select'] = device
    params['inpaint_size']['select'] = inpaint_size
    aot = AOTInpainter(**params)

    img = np.ones((inpaint_size, inpaint_size, 3), dtype=np.uint8)
    mask = np.ones((inpaint_size, inpaint_size), dtype=np.uint8)
    aot.inpaint(img, mask)
    test_inpainter(aot, proj, show=show, inpaint_by_block=inpaint_by_block)

def test_patchmatch(proj: ProjImgTrans, inpaint_by_block=True, show=False):
    inpainter = PatchmatchInpainter()
    test_inpainter(inpainter, proj, show=show, inpaint_by_block=inpaint_by_block)

if __name__ == '__main__':

    manga_dir = 'data/testpacks/manga'
    manga_proj = ProjImgTrans(manga_dir)
    # comic_proj = ProjImgTrans(comic_dir2)
    test_aot(manga_proj, device='cuda', inpaint_by_block=True, inpaint_size=2048)
    # test_patchmatch(comic_proj, inpaint_by_block=False)