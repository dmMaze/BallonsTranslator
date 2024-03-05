import sys
import os
import os.path as osp

import numpy as np

APP_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))

sys.path.append(APP_ROOT)


if __name__ == '__main__':

    os.chdir(APP_ROOT)
    os.environ['QT_API'] = 'pyqt6'

    from launch import main, args
    from ui.config_proj import ProjImgTrans
    from utils.io_utils import imread, imwrite, json_dump_nested_obj

    test_dir = 'tests/test_dir/text_rendering'
    if not osp.exists(test_dir):
        os.makedirs(test_dir)

    canvas_size = [1024, 1024]
    gen_dict = {
        'current_img': None,
        'pages': {
            "chs_horizontal.png": [{"xyxy": [254, 141, 470, 284], "lines": [[[254, 141], [470, 141], [470, 284], [254, 284]]], "language": "unknown", "vertical": False, "font_size": 90, "distance": None, "angle": 0, "vec": None, "norm": -1, "merged": False, "sort_weight": -1, "text": [""], "translation": "测试测试", "fg_r": 0, "fg_g": 0, "fg_b": 0, "bg_r": 236, "bg_g": 228, "bg_b": 255, "line_spacing": 1.2, "letter_spacing": 1.0, "font_family": "microsoft Himalaya", "bold": False, "underline": False, "italic": False, "_alignment": 0, "rich_text": "", "_bounding_rect": [303, 227, 318, 172], "default_stroke_width": 0, "stroke_decide_by_colordiff": True, "font_weight": 50, "opacity": 1.0, "shadow_radius": 0.0, "shadow_strength": 1.0, "shadow_color": [0, 0, 0], "shadow_offset": [0.0, 0.0], "src_is_vertical": True, "_detected_font_size": -1, "region_mask": None, "region_inpaint_dict": None}, {"xyxy": [223, 408, 439, 551], "lines": [[[254, 141], [470, 141], [470, 284], [254, 284]]], "language": "unknown", "vertical": False, "font_size": 90, "distance": None, "angle": 0, "vec": None, "norm": -1, "merged": False, "sort_weight": -1, "text": [""], "translation": "测试测试", "fg_r": 0, "fg_g": 0, "fg_b": 0, "bg_r": 236, "bg_g": 228, "bg_b": 255, "line_spacing": 1.2, "letter_spacing": 1.0, "font_family": "microsoft Himalaya", "bold": False, "underline": False, "italic": False, "_alignment": 1, "rich_text": "", "_bounding_rect": [272, 494, 318, 172], "default_stroke_width": 0, "stroke_decide_by_colordiff": True, "font_weight": 50, "opacity": 1.0, "shadow_radius": 0.0, "shadow_strength": 1.0, "shadow_color": [0, 0, 0], "shadow_offset": [0.0, 0.0], "src_is_vertical": True, "_detected_font_size": -1, "region_mask": None, "region_inpaint_dict": None}]
        }
    }

    proj = ProjImgTrans(test_dir)
    proj_updated = False
    if proj.is_empty:
        proj.load_from_dict(gen_dict)
        proj_updated = True

    if len(proj.not_found_pages) > 0:
        for k, blk in proj.not_found_pages.items():
            proj.pages[k] = proj.not_found_pages[k]
            img = np.full(canvas_size, 255, dtype=np.uint8)
            imwrite(osp.join(test_dir, k), img)
        proj_updated = True
        proj.load_from_dict(gen_dict)
        
    
    if proj.current_img is None and not proj.is_empty:
        proj.current_img = proj.idx2pagename(0)
        # proj.current_img = proj.

    if proj_updated:
        proj.save()

    args.debug = True
    args.proj_dir = test_dir
    main()