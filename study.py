import math
import os

from adjustText import adjust_text
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from scipy.stats import linregress

from process import DicomProcess

case = 2
site = None # None for all except 1

dir_loc = './data/to_process/'

test_file = os.path.join(dir_loc, str(case), f'RS.zzsrs_outlining{case}.GTV{site if site else ""} O{case}.dcm')


def get_files():
    for (dirpath, dirnames, filenames) in os.walk(dir_loc):
        for filename in filenames:
            if filename.startswith('RS.') and 'GTV' in filename and filename.endswith('.dcm'):
                yield os.path.join(dirpath, filename)


USERS = {
    1: {
        'original': [f'GTV{site} 2', f'GTV{site} 5', f'GTV{site} 7', f'GTV{site} 8', f'GTV{site} 9', f'GTV{site} 10', f'GTV{site} 12', f'GTV{site} 20', f'GTV{site} 14', f'GTV{site} 16', f'GTV{site} 18', f'GTV{site} 22', f'GTV{site} 21', f'GTV{site} 13', f'GTV{site} 11', f'GTV{site} 17', f'GTV{site} 19'],
    },
    2: {
        'original': ['GTV 6 outlier', 'GTV 4', 'GTV 2 outlier', 'GTV 1', 'GTV 5', 'GTV 3', 'GTV 7', 'GTV 8', 'GTV 9', 'GTV 10', 'GTV 11', 'GTV 12', 'GTV 13', 'GTV 14', 'GTV 16 outlier', 'GTV 17', 'GTV 18 outlier', 'GTV 19', 'GTV 20 outlier', 'GTV 21', 'GTV 22 outlier'],
        'good': ['GTV 4', 'GTV 1', 'GTV 5', 'GTV 3', 'GTV 7', 'GTV 8', 'GTV 9', 'GTV 10', 'GTV 11', 'GTV 12', 'GTV 13', 'GTV 14', 'GTV 17', 'GTV 19', 'GTV 20 outlier'],
        'resub': ['GTV 4', 'GTV 1', 'GTV 5', 'GTV 3', 'GTV 7', 'GTV 8', 'GTV 9', 'GTV 10', 'GTV 11', 'GTV 12', 'GTV 13', 'GTV 14', 'GTV 17', 'GTV 18R', 'GTV 19', 'GTV 20R', 'GTV 21', 'GTV 22R'] ,
        'outliers': [],
    },
    3: {
        'original': ['GTV 6', 'GTV 1', 'GTV 2', 'GTV 3', 'GTV 5', 'GTV 17', 'GTV 7', 'GTV 9', 'GTV 11', 'GTV 12', 'GTV 13', 'GTV 14', 'GTV 18', 'GTV 19', 'GTV 20', 'GTV 21', 'GTV 22'],
    },
    4: {
        'original': ['GTV 6', 'GTV 2', 'GTV 1', 'GTV 3', 'GTV 8', 'GTV 9', 'GTV 11', 'GTV 12', 'GTV 13', 'GTV 14 outlier', 'GTV 16 outlier', 'GTV 17', 'GTV 18', 'GTV 19', 'GTV 20 outlier', 'GTV 22'],
        'good': ['GTV 6', 'GTV 2', 'GTV 1', 'GTV 3', 'GTV 11', 'GTV 12', 'GTV 13', 'GTV 18', 'GTV 19', 'GTV 22'],
        'resub': ['GTV 6', 'GTV 2', 'GTV 1', 'GTV 3', 'GTV 8', 'GTV 9', 'GTV 11', 'GTV 12', 'GTV 13', 'GTV 14R', 'GTV 16R', 'GTV 17', 'GTV 18', 'GTV 19', 'GTV 20R', 'GTV 22'],
        'outliers': [],
    },
}
