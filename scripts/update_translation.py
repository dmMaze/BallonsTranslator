import os
import os.path as osp
from glob import glob

from qtpy.QtCore import QLocale
SYSLANG = QLocale.system().name()

if __name__ == '__main__':
    program_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    translate_dir = osp.join(program_dir, 'resources', 'translate')
    translate_path = osp.join(translate_dir, SYSLANG+'.ts')

    cmd = 'pylupdate5 -verbose '+ \
          ' '.join(glob(osp.join(program_dir, 'ballontranslator/ui/*.py'))) + \
          ' -ts ' + translate_path
    
    print('target language: ', SYSLANG)
    os.system(cmd)
    print(f'Saved to {translate_path}')
