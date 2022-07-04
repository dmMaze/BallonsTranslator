import numpy as np
from utils.io_utils import imread, imwrite

from qtpy.QtCore import Qt, Signal, QPoint, QSize, QThread
from qtpy.QtGui import QImage


class ImgSaveThread(QThread):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.im_save_list = []
        self.job = None

    def saveImg(self, save_path: str, img: QImage):
        self.im_save_list.append((save_path, img))
        if self.job is None:
            self.job = self._save_img
            self.start()

    def _save_img(self):
        while True:
            if len(self.im_save_list) == 0:
                break
            save_path, img = self.im_save_list.pop(0)
            if isinstance(img, QImage):
                img.save(save_path)
            elif isinstance(img, np.ndarray):
                imwrite(save_path, img)

    def run(self):
        if self.job is not None:
            self.job()
        self.job = None