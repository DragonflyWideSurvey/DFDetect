import matplotlib
matplotlib.use('Agg')
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.sdss import SDSS
from astroquery.gama import GAMA
from astropy.table import Table,Column

import shutil
from mrf.task import MrfTask

import gc
import argparse
import sep
from astropy import wcs
import numpy.ma as ma

master_mast_catalog = '/netb/dokkum/data/DWFS/master_mast_catalog.csv'
data_dir = '/netb/dokkum/tbm33/projects/df_SMF/data/'
output_dir = '/netb/dokkum/data/DWFS/mrf_frames/'
final_dir = '/netb/dokkum/data/DWFS/final_frames/'
decals_dir = '/netb/dokkum/data/DWFS/decals_frames/'
tile_dir = '/netb/dokkum/data/DWFS/tile_frames/'
all_fields = ['G09_139.5_1', 'G12_175.5_m1','G12_178.5_m1', 'G15_213_1', 'G15_222_1','G15_222_m1', 'G09_130.5_1', 'G09_136.5_1', 'G12_175.5_1','G09_139.5_m1']

df_as_per_pix = 2.5 #dragonfly arcseconds per pixel
decals_as_per_pix = 0.262 #Standard decals pixel scale
    
def run_mrf_tile(tile_name, mrf_dir, band, yaml_file = None, use_two_bands = True, display_result = False, skip_mast = True,save_fig = True,verbose = True, copy_to_final = True, mrf_task_kwargs = {}):
    '''
    Run mrf on a specified tile
    tile_name (string): name of DWFS tile
    mrf_dir (string): Directory for save mrf results
    band ('G' or 'R'): which bands to run mrf on
    '''
    if yaml_file == None:
        yaml_file = '/netb/dokkum/data/DWFS/df_decals_%s.yaml'%band.lower()
        if verbose == True: print ('Using default yaml file: %s'%yaml_file)
    
    #Get DF cutout
    #Currently they are all g band but will need to add additional identifier for which band in future
    img_lowres = tile_dir + tile_name+'_%s.fits'%band.lower()
    
    if use_two_bands == False:
        img_hires_r = decals_dir + tile_name + '_decals_%s_cutout.fits'%band.lower()
        img_hires_b = decals_dir + tile_name +  '_decals_%s_cutout.fits'%band.lower()
    else:
        img_hires_r = decals_dir + tile_name +  '_decals_cutout_r.fits'
        img_hires_b = decals_dir + tile_name +  '_decals_cutout_g.fits'
    
    #Set up MAST catalog
    #if mast catalog exists use mrf in skip_mast mode
    if skip_mast:
        ps1_cat = Table.read(master_mast_catalog, format = 'csv')
        df_wcs = WCS(fits.getheader(img_lowres))
        ps1_cat.add_columns([Column(data = df_wcs.wcs_world2pix(ps1_cat['raMean'], ps1_cat['decMean'], 0)[0], name='x_ps1'), 
        Column(data = df_wcs.wcs_world2pix(ps1_cat['raMean'], ps1_cat['decMean'], 0)[1], name='y_ps1')])
            
        ps1_cat.write(os.getcwd() + '/_ps1_cat.fits', overwrite=True)
        print ('using master MAST catalog')
    else:
        print ('Will download MAST catalog')
    
    if not os.path.exists(mrf_dir):
        os.makedir(mrf_dir)
    
    #run mrf
    task = MrfTask(yaml_file)
    results = task.run(img_lowres, img_hires_b, img_hires_r, None, output_name= mrf_dir +'%s'%band.lower(), verbose=verbose, wide_psf = True, skip_mast = skip_mast, **mrf_task_kwargs)
        
    if display_result or save_fig:
        from mrf.display import display_single
        fig, [ax1, ax2, ax3,ax4] = plt.subplots(1, 4, figsize=(30, 12))
        ax1 = display_single(results.lowres_input.image, ax=ax1, pixel_scale=2.0, 
            scale_bar_length=300, scale_bar_y_offset=0.3, add_text='Low res')
        ax2 = display_single(results.hires_img.image, ax=ax2, scale_bar=False, add_text='Hi res')

        ax3 = display_single(results.lowres_model.image, ax=ax3, scale_bar=False, add_text='Model')
        ax4 = display_single(results.lowres_final.image, ax=ax4, scale_bar=False, add_text='Residual')

        plt.subplots_adjust(wspace=0.05)
        if save_fig:
            plt.savefig(mrf_dir + 'mrf_result_%s.png'%band.lower(),bbox_inches = 'tight')
            
        if not display_result:
            plt.clf()
        else:
            plt.show()
        
    del task
    plt.close()
    gc.collect()
    if copy_to_final:
        shutil.copy(mrf_dir + '%s_final.fits'%band.lower(), final_dir + '%s_%s_final.fits'%(tile_name,band.lower()) )
        print ("Copied mrf'd frame to final_dir")
    return results, 

def set_up_mrf_field(fied_name, save_dir, size = 1, overlap = 0.5):
    '''
    Set up directories and tile catalog in prepration for mrf run
    field_name (string): name of DWFS frame
    bands ('G' and/or 'R'): which bands to run mrf on
    size (float): size in deg of tile to run mrf on
    overlap (float): amount of overlap, in deg, between tiles
    '''
    try:
        df_frame_file = DWFS_mrf.data_dir + 'WFS_frames/coadd_SloanG_%s_pcp_pcr.fits'%(field)
        field_file = DWFS_mrf.data_dir + '/WFS_frames/coadd_SloanG_%s_pcp_pcr.fits'%field
        field_wcs = WCS(fits.getheader(field_file))
        fp = field_wcs.calc_footprint()
    except:
        df_frame_file = DWFS_mrf.data_dir + 'WFS_frames/coadd_SloanG_%s_pcp_pcr.fits'%(field)
        print ('Could not load DWFS field: ', df_frame_file)
        return 0
    
    #Find footprint of field
    ra_min,ra_max = np.min(fp[:,0]),np.max(fp[:,0])
    dec_min,dec_max = np.min(fp[:,1]),np.max(fp[:,1])
    
    #Calculate number of tiles needed
    n_ra = np.ceil( ((ra_max - ra_min) - overlap)/(size - overlap) )
    n_dec = np.ceil( ((dec_max - dec_min) - overlap)/(size - overlap) )     
    
    ra_cent = []
    for i in range(int(n_ra)):
        if i==0:
            ra_cent.append(ra_min + size/(2.) )
        else:
            ra_cent.append(ra_cent[i-1] +  size- overlap)
    
    dec_cent = []
    for i in range(int(n_dec)):
        if i==0:
            dec_cent.append(dec_min + size/(2.) )
        else:
            dec_cent.append(dec_cent[i-1] +  size - overlap)
    
    print ('Preparing a total of %i tiles to run mrf'%(n_ra*n_dec))
    
    #Write tile catalog and make directories
    index = 0
    file = open(save_dir + 'tile_cat.txt', 'w+')
    file.write('-1\t %.3f\t %.3f \n'%(size, overlap))
    for i in range(int(n_ra)):
        for j in range(int(n_dec)):
            file.write('%i\t %.5f\t %.5f \n'%(index, ra_cent[i], dec_cent[j]))
            os.mkdir(save_dir + 'mrf_tile_%i/'%index)
            index+=1
    file.close()
    return 1

#Function to get decals cutout based on location and size
def get_decals_cutout(field, band, coord, save_dir, size_df_pixels = 1000, sub_bkg = True):
    cutout_file = save_dir + '_decals_cutout_%s.fits'%band
        
    if os.path.exists(cutout_file):
        print ( 'Cutout exists, overwriting: %s'%cutout_file )
        
    decals_frame_file = data_dir + 'WFS_hires/%s_decals_%s_%s.fits'%(field, band.lower(), band.lower() )
        
    decals_frame_fits = fits.open(decals_frame_file)
    decals_frame_wcs = WCS(decals_frame_fits[0].header)
                
    size_cutout = size_df_pixels * df_as_per_pix/decals_as_per_pix
        
    cutout = Cutout2D(decals_frame_fits[0].data, coord, size_cutout, wcs = decals_frame_wcs)
        
    cutout_hdu = fits.PrimaryHDU()
        
    if sub_bkg:
        temp_data = cutout.data.byteswap().newbyteorder()
        bkg = sep.Background(temp_data, bw = 256,bh = 256)
            
        data_to_save = temp_data - bkg.back()
    else:
        data_to_save = np.copy(cutout.data)
            
    cutout_hdu.data = data_to_save
    cutout_hdu.header.update(cutout.wcs.to_header() )
        
    #Make sure all relavent info is passed on to cutout
    to_update_header = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
    for to_update in to_update_header:
        cutout_hdu.header[to_update] = decals_frame_fits[0].header[to_update]
        
    cutout_hdu.writeto(cutout_file,overwrite = True)
        
    del cutout_hdu
    decals_frame_fits.close()
    gc.collect()
    return 1

#Function to get DF cutout based on location and size
def get_df_cutout(field, band, coord, save_dir, size_df_pixels = 1000):
        
    #Band options either 'G' or 'R'
    cutout_file = save_dir +'_DF_cutout_%s.fits'%band
    df_frame_file = data_dir + 'WFS_frames/coadd_Sloan%s_%s_pcp_pcr.fits'%(band,field)
        
    df_frame_fits = fits.open(df_frame_file)
    df_frame_wcs = WCS(df_frame_fits[0].header)
                                
    cutout_df = Cutout2D(df_frame_fits[0].data, coord, size_df_pixels, wcs = df_frame_wcs)
        
    cutout_hdu = fits.PrimaryHDU()
    cutout_hdu.data = cutout_df.data
    cutout_hdu.header.update(cutout_df.wcs.to_header() )
        
    #Make sure all relavent info is passed on to cutout
    to_update_header = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2','BACKVAL','REFZP','MEDIANZP']
    for to_update in to_update_header:
        cutout_hdu.header[to_update] = df_frame_fits[0].header[to_update]
        
    cutout_hdu.writeto(cutout_file,overwrite = True)
        
    del cutout_hdu
    df_frame_fits.close()
    gc.collect()
    return 1