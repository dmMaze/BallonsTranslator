import sys, os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))


from test_translators import dosth




if __name__ == '__main__':
    print(dosth())