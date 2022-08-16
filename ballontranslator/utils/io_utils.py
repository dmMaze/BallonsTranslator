import json, os, cv2
import os.path as osp
import numpy as np
from pathlib import Path

IMG_EXT = ['.bmp', '.jpg', '.png', '.jpeg']
NP_BOOL_TYPES = (np.bool_, np.bool8)
NP_FLOAT_TYPES = (np.float_, np.float16, np.float32, np.float64)
NP_INT_TYPES = (np.int_, np.int8, np.int16, np.int32, np.int64, np.uint, np.uint8, np.uint16, np.uint32, np.uint64)

def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__, ensure_ascii=False))

def json_dump_nested_obj(obj):
    return json.dumps(obj, default=lambda o: o.__dict__, ensure_ascii=False)

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
    for filename in os.listdir(img_dir):
        file_suffix = Path(filename).suffix
        if file_suffix.lower() not in IMG_EXT:
            continue
        if abs_path:
            imglist.append(osp.join(img_dir, filename))
        else:
            imglist.append(filename)
    return imglist

def imread(imgpath, read_type=cv2.IMREAD_COLOR):
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

