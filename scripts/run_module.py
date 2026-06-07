import argparse
import sys, os
import os.path as osp
APP_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from ballontranslator.utils.shared import PROGRAM_PATH


os.chdir(PROGRAM_PATH)


def init_module(module_type: str, module_name: str):
    from ballontranslator.modules import MODULETYPE_TO_REGISTRIES
    from ballontranslator.utils.config import pcfg

    assert module_type in MODULETYPE_TO_REGISTRIES
    module_cls = MODULETYPE_TO_REGISTRIES[module_type].get(module_name)
    module_cls_params = getattr(pcfg.module, module_type + '_params')
    module_params = module_cls_params.get(module_name, {})
    return module_cls(**module_params)

def run_detector(proj_dir, detector, config, save_dir):
    from tqdm import tqdm

    from ballontranslator.modules import init_module_registries
    from ballontranslator.utils.config import load_config, pcfg
    from ballontranslator.utils.io_utils import imwrite
    from ballontranslator.utils.proj_imgtrans import ProjImgTrans
    from ballontranslator.utils.textblock import visualize_textblocks

    init_module_registries('textdetector')
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
    


def main(argv=None):
    parser = argparse.ArgumentParser(description='text detector testing scripts.')
    subparsers = parser.add_subparsers(dest='command')

    detector_parser = subparsers.add_parser('run_detector')
    detector_parser.add_argument('--proj_dir')
    detector_parser.add_argument('--detector', default=None)
    detector_parser.add_argument('--config', default='config/config.json')
    detector_parser.add_argument('--save_dir', default='tmp/test_ctd')

    args = parser.parse_args(argv)
    if args.command == 'run_detector':
        run_detector(args.proj_dir, args.detector, args.config, args.save_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
