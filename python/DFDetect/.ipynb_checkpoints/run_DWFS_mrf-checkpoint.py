#Simple multiprocessing set up to run tiles
import numpy as np
import pandas as pd
import os
import DWFS_mrf
from multiprocessing import Pool
from astropy.coordinates import SkyCoord
import astropy.units as u

tile_cat = pd.read_csv('/netb/dokkum/data/DWFS/tile_cat_v1.csv')
tile_names = tile_cat['tile_name'][:]

def run_tile_mp(tile_name):
    mrf_dir = DWFS_mrf.output_dir + tile_name + '/'
    band = 'g'
    if not os.path.exists(mrf_dir):
        os.mkdir(mrf_dir)
    os.chdir(mrf_dir)
    res_G = DWFS_mrf.run_mrf_tile(tile_name, mrf_dir, band)
    #res_R = DWFS_mrf.run_mrf_tile(tile_name, mrf_dir, band)
    return 1

p = Pool(1)
print (p.map(run_tile_mp, tile_names[:10]) )
