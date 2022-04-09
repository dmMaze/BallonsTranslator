import json, re, cv2, glob
import os.path as osp
import numpy as np
from pathlib import Path

IMG_EXT = ['.bmp', '.jpg', '.png', '.jpeg']
NP_BOOL_TYPES = (np.bool_, np.bool8)
NP_FLOAT_TYPES = (np.float_, np.float16, np.float32, np.float64)
NP_INT_TYPES = (np.int_, np.int8, np.int16, np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, np.uint64)

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.ScalarType):
            if isinstance(obj, NP_BOOL_TYPES):
                return bool(obj)
            elif isinstance(obj, NP_FLOAT_TYPES):
                return float(obj)
            elif isinstance(obj, NP_INT_TYPES):
                return int(obj)
        return json.JSONEncoder.default(self, obj)



def find_all_imgs(img_dir, abs_path=False):
    imglist = []
    for filep in glob.glob(osp.join(img_dir, "*")):
        filename = osp.basename(filep)
        file_suffix = Path(filename).suffix
        if file_suffix.lower() not in IMG_EXT:
            continue
        if abs_path:
            imglist.append(filep)
        else:
            imglist.append(filename)
    return imglist

def imread(imgpath, read_type=cv2.IMREAD_COLOR):
    # img = cv2.imread(imgpath, read_type)
    # if img is None:
    if not osp.exists(imgpath):
        return None
    img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), read_type)
    return img

def imwrite(img_path, img, ext='.png'):
    suffix = Path(img_path).suffix
    if suffix != '':
        img_path = img_path.replace(suffix, ext)
    else:
        img_path += ext
    cv2.imencode(ext, img)[1].tofile(img_path)


def show_img_by_dict(imgdicts):
    for keyname in imgdicts.keys():
        cv2.imshow(keyname, imgdicts[keyname])
    cv2.waitKey(0)

widths = [
  (126,  1), (159,  0), (687,   1), (710,  0), (711,  1),
  (727,  0), (733,  1), (879,   0), (1154, 1), (1161, 0),
  (4347,  1), (4447,  2), (7467,  1), (7521, 0), (8369, 1),
  (8426,  0), (9000,  1), (9002,  2), (11021, 1), (12350, 2),
  (12351, 1), (12438, 2), (12442,  0), (19893, 2), (19967, 1),
  (55203, 2), (63743, 1), (64106,  2), (65039, 1), (65059, 0),
  (65131, 2), (65279, 1), (65376,  2), (65500, 1), (65510, 2),
  (120831, 1), (262141, 2), (1114109, 1),
]
def get_width(o):
    global widths
    o = ord(o)

    if o == 0xe or o == 0xf:
        return 0
    for num, wid in widths:
        if o <= num:
            return wid
    return 1

def get_text_width(text):
    '''get text width'''
    # text = text.encode("utf-8")
    # text = text.decode("utf-8")
    width = 0
    for t in text:
        width += get_width(t) 
    return width

def text_is_empty(text) -> bool:
    if isinstance(text, str):
        if text.strip() == '':
            return True
    if isinstance(text, list):
        for t in text:
            t_is_empty = text_is_empty(t)
            if not t_is_empty:
                return False
        return True    
    elif text is None:
        return True

def bubdict_template(source_lang="en",
                     target_lang="chi",
                     seg_mode=0, 
                    #  inpaint_type=0, 
                     default_font_size=15,
                     default_font_family=-1,
                     vert_text=False,
                     line_spacing=85):
    template_dict = {  
                     "source_lang": source_lang,
                     "target_lang": target_lang,
                     "seg_mode": seg_mode,
                     "inner_rect": [-1, -1, -1, -1, -1],
                     "xywh": [-1, -1, -1, -1, -1],
                    #  "inpaint_type": inpaint_type,
                     "bgr": [0, 0, 0],
                     "bground_bgr":[255,255,255],
                     "translation": "",
                     "ocr": "",
                     "default_font_size": default_font_size,
                     "default_font_family": default_font_family,
                     "vert_text": vert_text,
                     "alignment": 0,
                     "line_spacing": line_spacing}
    return template_dict


# def gen_projconfig(proj_path):
#     config_path = osp.join(proj_path, osp.basename(proj_path) + "_proj.json")
#     imglist = find_all_imgs(proj_path)
    
#     config = {"stage":0, "new_img_num":0, "img_info":{}, "all_boxes":{}, "mask_ext":".bmp", "result_ext": ".png"}
#     for imgp in imglist:
#         config["img_info"][imgp] = {}

#     format_save(config_path, config)
#     return config




def load_proj_dict(proj_dict_path, check_img_update=True):
    proj_dict_path = proj_dict_path.replace("/", "\\")
    proj_path = proj_dict_path[:proj_dict_path.rfind("\\")]
    imglist = find_all_imgs(proj_path)
    assert osp.exists(proj_dict_path)
    with open(proj_dict_path, "r", encoding="utf-8") as f:
        proj_dict = json.loads(f.read())
        img_info_dict = proj_dict["img_info"]
        img_names = list(img_info_dict.keys())
        for name in img_names:
            if osp.exists(osp.join(proj_path, name)) == False:
                proj_dict["img_info"].pop(name)
        if check_img_update:
            proj_dict["new_img_num"] = 0
            for imgl in imglist:
                if imgl in img_names:
                    continue
                proj_dict["img_info"][imgl] = {}
                proj_dict["new_img_num"] += 1
    # else:
    #     proj_dict = gen_projconfig(proj_path)

    format_save(proj_dict_path, proj_dict)
    return proj_dict
        
def format_save(config_path, config_dict):
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(config_dict, ensure_ascii=False, indent=4, separators=(',', ':'), cls=TextBlkEncoder))
    # with open(config_path, "r", encoding="utf-8") as f:
    #     lines = f.readlines()
    #     wr = ""
    #     skip_ln, lrst = False, 0
    #     for l in lines:
    #         if re.findall("\[", l):
    #             skip_ln, lrst = True, -1
    #         if re.findall("\]", l):
    #             skip_ln = False
    #             l = l.lstrip()
    #         if skip_ln:
    #             l = l.replace("\n", "")
    #             if lrst == 0:
    #                 l = l.lstrip()
    #             wr += l
    #             if lrst == -1:
    #                 lrst = 0
    #         else: wr += l

    # with open(config_path, "w", encoding="utf-8") as f:
    #     f.write(wr)



