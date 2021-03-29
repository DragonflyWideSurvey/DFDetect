import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from astropy.wcs import WCS
import astropy.units as u
from astropy.wcs.utils import pixel_to_skycoord,skycoord_to_pixel
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord

import sep
import astropy.units as u
import pandas as pd

from mrf import celestial,display

from unagi import hsc,task

pdr2 = hsc.Hsc(dr='pdr2',rerun = 'pdr2_wide')


def detect_sources(data,snr, band, wcs_cur, mask = None, search_sdss = True, search_radius = 4.,sep_bkg_kwargs = {}, sep_extract_kwargs = {}, plot_sources = True, save_plot = None,photo_z = True):
    """
    Function used to detect sources in DF images and search for counterparts in the SDSS database
    
    Paramaters
    ----------
        data: 2D Array
            Image to be searched, typically a mrf'ed DF tile
        snr: float
            SNR cut, above which to detect sources
        band:
            Observed band 'G' or 'R'
        wcs_cur: Astropy WCS object
            WCS information of data
        
        mask: 2D Array (optional)
            2D array containg pixels to be masked
        
        
        search_sdss: bool (optional)
            Whether to search sdss database for counterparts to detects sources, default True.
            if False then speeds up function call
        photo_z: bool (optional)
            Wether to search SDSS photoz database as backup for galaxies without specz
        search_radius: float (optional)
            Radius to search for counterparts in arcsec, defualt 4
        sep_bkg_kwargs: dict (optional)
            Keywords to pass to sep.Background call
        sep_extract_kwargs: dict (optional)
            Keywords to pass to sep.extract call
        
        plot_sources: bool (optional)
            Whether to make a figure diplaying detected sources, defualt True
        save plot: Str (optional)
            String specifying location of where figure should be saved, default None
    Returns
    -------
        obj_pd: Pandas DataFrame
            Pandas DataFrame containing all of information about the detected sources
"""
    #####
    # Run sep to detect sources
    #####
    bkg = sep.Background(data, mask=mask, **sep_bkg_kwargs)
    obj_tab = sep.extract(data - bkg.back(), snr, err=bkg.rms(), **sep_extract_kwargs )
    obj_pd = pd.DataFrame(obj_tab)
    obj_pd['x']
    obj_pd['y']
    
    #####
    # Run simple cuts to remove unwanted sources 
    #####
    #remove objects near edge
    obj_pd = obj_pd.query('x < 1002 and x > 5 and y < 1002 and y > 5')
    
    if mask is not None:
        #Check if pixels are masked near source
        mask_phot,_,_ = sep.sum_circle(mask, obj_pd['x'], obj_pd['y'], [2.]*len(obj_pd))
        obj_pd['mask_phot'] = mask_phot
        obj_pd = obj_pd.query('mask_phot < 1')

    obj_pd = obj_pd.reset_index(drop = True)
    coord_all =  pixel_to_skycoord(obj_pd['x']+1, obj_pd['y']+1, wcs_cur)
    
    obj_pd['ra'] = coord_all.ra.deg
    obj_pd['dec'] = coord_all.dec.deg
    
    #####
    # Search SDSS database for nearby objects
    #####
    if search_sdss:
        star_near = []
        gal_near = []
        
        star_mag = []
        gal_z = []
        gal_log_ms = []
        gal_photoobjid = []
        gal_specobjid = []
        
        for i, obj in obj_pd.iterrows():
            coord = pixel_to_skycoord(obj['x']+1, obj['y']+1, wcs_cur )
            tab = SDSS.query_region(coord, radius = search_radius*u.arcsec,photoobj_fields = ['ra','dec','objid','mode','type', 'psfMag_g', 'psfMag_r'])
    
            if tab is None:
                star_near.append(False)
                star_mag.append(-99)
                
                gal_near.append(False)
                gal_z.append(-99)
                gal_log_ms.append(-99)
                gal_photoobjid.append(-99)
                gal_specobjid.append(-99)                
                continue

            pd_cur = tab.to_pandas() 
            gal_pd = pd_cur.query('mode == 1 and type == 3').reset_index()
            star_pd = pd_cur.query('mode == 1 and type == 6').reset_index()
            
            if len(star_pd) == 0:
                star_near.append(False)
                star_mag.append(-99)
            else:
                star_near.append(True)
                star_mag.append(np.min(star_pd['psfMag_%s'%band.lower()]))
                
            if len(gal_pd) == 0:
                gal_near.append(False)
                gal_z.append(-99)
                gal_log_ms.append(-99)
                gal_photoobjid.append(-99)
                gal_specobjid.append(-99)
            
            else:
                gal_sep_as = coord.separation( SkyCoord(ra = gal_pd['ra']*u.deg, dec = gal_pd['dec']*u.deg)).to(u.arcsec).value
                gal_near.append(True)
                gal_photoobjid.append(gal_pd['objid'][np.argmin(gal_sep_as)])
                dx = search_radius/60./60.
                SM_tab = SDSS.query_sql('Select ra,dec,z,mstellar_median,specObjID from stellarMassPCAWiscBC03 where ra between %.6f and %.6f and dec between %.6f and %.6f'%(coord.ra.deg - dx,coord.ra.deg + dx, coord.dec.deg - dx, coord.dec.deg + dx ))
                if SM_tab is None:
                    gal_z.append(-99)
                    gal_log_ms.append(-99)
                    gal_specobjid.append(-99)
                else:
                    sm_use = SM_tab[np.argmin(coord.separation( SkyCoord(ra = SM_tab['ra']*u.deg, dec = SM_tab['dec']*u.deg)).to(u.arcsec).value )]
                    gal_z.append(sm_use['z'])
                    gal_log_ms.append(sm_use['mstellar_median'])
                    gal_specobjid.append(sm_use['specObjID'])
        
        obj_pd['star_near'] = star_near
        obj_pd['gal_near'] = gal_near
        obj_pd['star_mag'] = star_mag
        obj_pd['gal_spec_z'] = gal_z
        obj_pd['gal_log_ms'] = gal_log_ms
        obj_pd['gal_photoobjid'] =  gal_photoobjid
        obj_pd['gal_specobjid'] = gal_specobjid
    
    
    if photo_z and search_sdss:
        to_query = 'Select objID as gal_photoobjid,z as gal_phot_z, zErr as gal_phot_z_err from Photoz where objID in ('
        to_add = ['%i ,'%id_cur for id_cur in obj_pd.query('gal_near == True').reset_index()['gal_photoobjid']]
        to_query = to_query + ''.join(to_add)[:-1] + ')'
        phz_tab = SDSS.query_sql(to_query)
        obj_pd = obj_pd.join( phz_tab.to_pandas().set_index('gal_photoobjid'), on = 'gal_photoobjid')
        obj_pd['gal_phot_z'] = obj_pd['gal_phot_z'].replace(np.nan,int(-99))
        obj_pd['gal_phot_z_err'] = obj_pd['gal_phot_z_err'].replace(np.nan,int(-99))

    if plot_sources:
        fig, ax = plt.subplots(figsize = (15,15))
        data_sub = data - bkg.back()
        
        if mask is not None: data_sub[np.where(mask == 1)] = 0
            
        m, s = np.mean(data_sub), np.std(data_sub)
        im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
               vmin=m-s, vmax=m+s, origin='lower')

        # plot an ellipse for each object
        for i,obj in obj_pd.iterrows():
            e = Ellipse(xy=(obj['x'], obj['y']),
                width=6*obj['a'],
                height=6*obj['b'],
                angle=obj['theta'] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)
        if save_plot is not None: plt.savefig(save_plot)
    return obj_pd



def plot_cutouts(res, obj_pd, decals_fits, save_name = None, df_size = 50, hr_size = 20):
    """
    Function used to plot cutouts of detected sources for exploratory purposes
    
    Paramaters
    ----------
        res: mrf results objects
        
        obj_pd: Pandas Dataframe
            DataFrame containing information on sources, generally the output of detect.detect_sources
        
        decals_fits: Astropy fits HDU
            Fits file containing the un-magnifying DECALS images
        
        save_name: str (optional)
            Location to save figure
        
        df_size: float (optional)
            size in arcsec for DF cutout, default 50
        
        hr_size: float (optional)
            size in arcsec for Decals and HSC cutouts, defualt 20
    
    returns:
        Figure
    
"""
    num_obj = len(obj_pd)
    fig, axes = plt.subplots(num_obj,5, figsize = (14,num_obj*3 ) )
    i = 0
    
    for i,obj in obj_pd.iterrows():
        
        if num_obj == 1:
            ax_df = axes[0]
            ax_hrfl = axes[1]
            ax_lrfl = axes[2]
            ax_dec = axes[3]
            ax_hsc = axes[4]        
        else:
            ax_df = axes[i][0]
            ax_hrfl = axes[i][1]
            ax_lrfl = axes[i][2]
            ax_dec = axes[i][3]
            ax_hsc = axes[i][4]
        
        
        #Calculate properties of object
        coord = pixel_to_skycoord(obj['x'], obj['y'], res.lowres_final.wcs )
        
        
        #Get df_cutout
        df_img = celestial.img_cutout(res.lowres_final.image, res.lowres_final.wcs, coord.ra/u.deg *u.deg,coord.dec/u.deg *u.deg, size = [df_size,df_size], pixel_scale = 2.5, save = False)
    
        #Get mrf outputs
        hr_flux_mod = celestial.img_cutout(res.hires_fluxmod,res.hires_img.wcs, coord.ra/u.deg *u.deg,coord.dec/u.deg *u.deg, size = [df_size,df_size], pixel_scale = 0.833, save = False) 
        lr_flux_mod = celestial.img_cutout(res.lowres_model.image,res.lowres_model.wcs, coord.ra/u.deg *u.deg,coord.dec/u.deg *u.deg, size = [df_size,df_size], pixel_scale = 2.5, save = False) 
        #Get decals cutout
        dec_img = celestial.img_cutout(decals_fits[0].data, WCS(decals_fits[0].header), coord.ra.deg,coord.dec.deg, size = [hr_size,hr_size],pixel_scale = 0.262, save = False)

        #Download hsc_cutout
        hsc_img = task.hsc_cutout(coord, cutout_size = hr_size/2.*u.arcsec, filters = 'g',archive = pdr2,dr = 'pdr2', use_saved=False, verbose=True, variance=False, mask=False, save_output=False)
        hsc_wcs = WCS(hsc_img[1].header)
    
        #Plot images
        display.display_single(df_img[0].data, pixel_scale = 2.5,ax = ax_df,contrast = 0.001, scale_bar_y_offset  = 2)
        
        
        f_hr = np.sum(hr_flux_mod[0].data == 0) / (hr_flux_mod[0].data.shape[0]*hr_flux_mod[0].data.shape[1]) * 100
        if f_hr == 100:
            f_hr -= 1
       
        display.display_single(hr_flux_mod[0].data, pixel_scale = 0.8333,ax = ax_hrfl,contrast = 0.001, lower_percentile=98, upper_percentile = 100, scale_bar_y_offset  = 2)
        display.display_single(lr_flux_mod[0].data, pixel_scale = 2.5,ax = ax_lrfl,contrast = 0.001, scale_bar_y_offset  = 2)
        display.display_single(hsc_img[1].data, ax = ax_hsc,contrast = 0.03, scale_bar_y_offset  = 2, pixel_scale = 0.168)
        display.display_single(dec_img[0].data, pixel_scale = 0.262, ax = ax_dec,contrast = 0.03, scale_bar_y_offset  = 2)

    
        #Plot outline of smaller cutouts on larger cutouts
        lr_box_loc = df_img[0].wcs.world_to_pixel_values(dec_img[0].wcs.calc_footprint()*u.deg )
        lr_box_loc = np.vstack([lr_box_loc,lr_box_loc[0]])

        hr_box_loc = hr_flux_mod[0].wcs.world_to_pixel_values(dec_img[0].wcs.calc_footprint()*u.deg )
        hr_box_loc = np.vstack([hr_box_loc,hr_box_loc[0]])
    
        ax_df.plot(lr_box_loc[:,0], lr_box_loc[:,1], 'w-', lw = 2)
        ax_lrfl.plot(lr_box_loc[:,0], lr_box_loc[:,1], 'w-', lw = 2)
        ax_hrfl.plot(hr_box_loc[:,0], hr_box_loc[:,1], 'w-', lw = 2)
    
        #Define all panels
        ax_df.set_title('obj %i, mrf_final - 50`` x 50`` '%(i) )
        ax_lrfl.set_title('mrf low res flux model - 50`` x 50 ``')
        ax_hrfl.set_title('mrf high res flux model - 50`` x 50 ``')
        ax_dec.set_title('DeCALS - 20`` x 20``')
        ax_hsc.set_title('HSC - 20`` x 20``')
    
        #Query and plot source from SDSS Photoobj catalog
        #Blue are galaxies, red a stars
        tab = SDSS.query_region(coord, radius = hr_size/1.5*u.arcsec,photoobj_fields = ['ra','dec','objid','mode','type'])
        for src in tab:
            if src['mode'] != 1: continue
        
            x,y = skycoord_to_pixel(SkyCoord(ra = src['ra']*u.deg,dec = src['dec']*u.deg), hsc_wcs)
            e = Ellipse(xy=(x, y), width=16, height=16, angle=0,lw = 2)
            e.set_facecolor('none')
            if src['type'] == 3:
                e.set_edgecolor('blue')
            else:
                e.set_edgecolor('red')
            ax_hsc.add_artist(e)
        
            x1,y1 = skycoord_to_pixel(SkyCoord(ra = src['ra']*u.deg,dec = src['dec']*u.deg), dec_img[0].wcs)
            e1 = Ellipse(xy=(x1, y1), width=16*0.168/0.262, height = 16*0.168/0.262, angle=0,lw = 2)
            e1.set_facecolor('none')
            if src['type'] == 3:
                e1.set_edgecolor('blue')
            else:
                e1.set_edgecolor('red')
            ax_dec.add_artist(e1)
        
        #Plot geometry of detected source
        obj_x,obj_y = skycoord_to_pixel(coord, df_img[0].wcs, origin = 0)
        e = Ellipse(xy=(obj_x, obj_x),
                width=2*obj['a'],
                height=2*obj['b'],
                angle=obj['theta'] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('k')
        ax_df.add_artist(e)
    
        obj_x,obj_y = skycoord_to_pixel(coord, dec_img[0].wcs)
        e = Ellipse(xy=(obj_x, obj_y),
            width=2*obj['a']*2.5/0.262,
            height=2*obj['b']*2.5/0.262,
            angle=obj['theta'] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('k')
        ax_dec.add_artist(e)
    
    
        obj_x,obj_y = skycoord_to_pixel(coord, hsc_wcs)
        e = Ellipse(xy=(obj_x, obj_y),
            width=2*obj['a']*2.5/0.168,
            height=2*obj['b']*2.5/0.168,
            angle=obj['theta'] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('k')
        ax_hsc.add_artist(e)
    
    #fig.subplots_adjust(wspace = -1,hspace = 0 )
    if save_name is not None: 
        plt.savefig(save_name, bbox_inches = 'tight')
    return fig