import click

import sys, os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))

from tqdm import tqdm

from utils.config import load_config

from utils.shared import PROGRAM_PATH
from utils.textblock import visualize_textblocks
from utils.proj_imgtrans import ProjImgTrans
from utils.config import pcfg
from utils.io_utils import imread, imwrite
from modules import MODULETYPE_TO_REGISTRIES, init_translator_registries, init_inpainter_registries, init_ocr_registries, init_textdetector_registries


os.chdir(PROGRAM_PATH)


@click.group()
def cli():
    """text detector testing scripts.
    """



def init_module(module_type: str, module_name: str):
    assert module_type in MODULETYPE_TO_REGISTRIES
    module_cls = MODULETYPE_TO_REGISTRIES[module_type].get(module_name)
    module_cls_params = getattr(pcfg.module, module_type + '_params')
    module_params = module_cls_params.get(module_name, {})
    return module_cls(**module_params)


@cli.command('run_detector')
@click.option('--proj_dir')
@click.option('--detector', default=None)
@click.option('--config', default='config/config.json')
@click.option('--save_dir', default='tmp/test_ctd')
def run_detector(proj_dir, detector, config, save_dir):

    init_textdetector_registries()
    load_config(config)
    if detector is None:
        detector = pcfg.module.textdetector

    detector = init_module('textdetector', detector)
    print('detector params:', detector.params)

    proj = ProjImgTrans(proj_dir)
    for page_name in tqdm(proj.pages):
        blk_list = proj.pages[page_name]
        proj.set_current_img(page_name)
        mask, blk_list = detector.detect(proj.img_array, blk_list)
        blk_list = blk_list[:1]
        print(blk_list[0].get_text())
        vis = visualize_textblocks(proj.img_array, blk_list)
        imwrite(osp.join(save_dir, proj.current_img), vis, ext='.jpg')
        pass
    


if __name__ == '__main__':
    cli()
