import sys, os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from ui.imgtrans_proj import ProjImgTrans
from ui.constants import PROGRAM_PATH
os.chdir(PROGRAM_PATH)

if __name__ == '__main__':
    proj_path = r'data/testpacks/manga'
    proj = ProjImgTrans(proj_path)
    proj.dump_doc()