import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

from DFDetect import DWFS_mrf, detect
import os

band = 'G'
tile_name = 'tile_223.647_-1.424_SloanG'
DF_file = DWFS_mrf.tile_dir + tile_name + '_g.fits'
config_file = '/home/tbm33/projects/packages/DragonflyWideSurvey/df_decals_g.yaml'
decals_g_file = DWFS_mrf.decals_dir + tile_name + '_decals_cutout_g.fits'
decals_r_file = DWFS_mrf.decals_dir + tile_name + '_decals_cutout_r.fits'

#You don't need the mast catalog but might be worth downloading if you plan on running on multiple tiles
res = DWFS_mrf.run_mrf_tile(DF_file, band, config_file, './output/', hires_g_file = decals_g_file, hires_r_file = decals_r_file, mast_file = DWFS_mrf.master_mast_catalog, use_two_bands = True,save_fig = True)