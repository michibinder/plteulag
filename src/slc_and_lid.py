import sys
import os
import glob
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal, integrate

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from matplotlib.ticker import MaxNLocator

# - for reloading libraries/modules - #
import importlib
import multiprocessing as mp

# import subroutines
# import vis_eulag
#importlib.reload(subroutines)
#importlib.reload(vis_eulag)
#from vis_eulag import *

import warnings
warnings.filterwarnings('ignore')
# with warnings.catch_warnings():
#     warnings.filterwarnings('ignore', r'invalid value encountered in sqrt')

import cmaps, plt_helper

"""Config"""
pltsurf = 0
pbar_interval = 5 # %
data_folder = "/scratch/b/b309199"
animation_folder = "../data/animation_slices"
# data_folder = "/work/bd0620/b309199"

if os.path.exists('latex_default.mplstyle'):
    plt.style.use('latex_default.mplstyle')


def slc_and_lid(t, var, fpath, image_folder, pbar):
    """Visualize u,v,th or other vars in vertical profiles, different cross sections and virtual lidar time-height diagrams"""

    ds, ds_env, ds_xzslices, ds_yzslices, ds_xyslices, ds_lidars, ds_full = plt_helper.preprocess_eulag_output(fpath)
    
    global xlim, ylim, zlim, xlim_lid, tstamp, zsponge, thlev, surf_factor

    """Lidar locations"""
    ds_lid = ds_lidars[1]
    xlim_lid = [0,ds_lid.time.max().values]
    
    """Slices"""
    dsxz = ds_xzslices[1]
    dsyz = ds_yzslices[0]
    dsxz['zcr'] = dsxz['zcr']/1000
    dsyz['zcr'] = dsyz['zcr']/1000
    
    """Limits and timestamp and thlev"""
    tstamp  = t * ds.nslice * ds.dt00 / 3600
    zsponge = [ds.zab/1000, dsxz.zcr.max().values]
    if  ds.attrs["itopo"] == 0:
        if t == 0:
            print("[i]  itopo: ", ds.attrs["itopo"], ". Using idealized topography.")
    else:
        if t == 0:
            print("[i]  itopo: ", ds.attrs["itopo"], ". Using realistic topography.")
    if pltsurf == 1:
        surf_factor = 1
        zlim  = [0,20]
        thlev = np.exp(1+0.03*np.arange(1,100,0.5)) 
        xlim  = [-600, 600] # km
        ylim  = [-600, 600] # km
    else:
        surf_factor = 5
        zlim  = [0,dsxz.zcr.max().values]
        thlev = np.exp(4+0.03*np.arange(1,350,5))
        #thlev=np.exp(5+0.03*np.arange(1,250,10))
        xlim  = [ds.xcr.min().values,ds.xcr.max().values]
        ylim  = [ds.ycr.min().values,ds.ycr.max().values]
    

    """Momentum flux - gaussian filter"""
    global lambdax, lambdaz
    lambdax = 200 * 1000 # m 
    lambdaz =  20 * 1000 # m 

    """Plot parameter"""
    global lw1, lw_sponge, csponge
    csponge = 'lightgrey'
    lw1 = 2
    lw_sponge = 1.5
    alpha_box = 0.9
    alpha_sponge = 0.7

    """Colormap"""
    global cmap, clev, norm
    if var == "th":
        cmap   = cmaps.get_wave_cmap()
        # clev   = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32]
        # clev_l = [-16,-4,-1,1,4,16]
        clev, clev_l = plt_helper.get_colormap_bins_and_labels(max_level=64)
        norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
        cbar_label = r"$\Theta'$ / K"
    elif var == "mf":
        cmap = plt.get_cmap('PiYG_r')
        clev, clev_l = plt_helper.get_colormap_bins_and_labels(max_level=32)
        cbar_label = r"MF$_x$ / mPa"
    elif var == "w":
        cmap = plt.get_cmap('seismic')
        clev, clev_l = plt_helper.get_colormap_bins_and_labels(max_level=8)
        cbar_label = r"w' / m$\,$s$^{-1}$"
    elif var == "u":
        cmap = plt.get_cmap('bwr')
        clev, clev_l = plt_helper.get_colormap_bins_and_labels(max_level=32)
        cbar_label = r"u' / m$\,$s$^{-1}$"

    norm = BoundaryNorm(boundaries=clev, ncolors=cmap.N, clip=True)

    """Labels"""
    global xpp, ypp
    xpp = 0.88
    ypp = 0.93
    ipp = 0
    numb_str = ['a','b','c','d','e','f','g','h','i','j']

    """Figure stuff"""
    gskw  = {'hspace':0.05, 'wspace':0.03, 'height_ratios': [7,7,0.4,1], 'width_ratios': [3,3,3]} #  , 'width_ratios': [5,5]}
    # gskw2 = {'hspace':0.14, 'wspace':0.06, 'height_ratios': [7,7,0.1,2], 'width_ratios': [1,1,1,3,3]}
    fig, axes = plt.subplots(4,3,figsize=(15,10), gridspec_kw=gskw)
    for ax in axes[-1,0:3]:
        ax.set_axis_off()
    for ax in axes[-2,0:3]:
        ax.set_axis_off()

    # --- Replace top left axis --- #
    gs_topleft = axes[0,0].get_gridspec()
    axes[0,0].remove()
    
    gs_top2 = fig.add_gridspec(4,5, hspace=0.05, wspace=0.03, height_ratios=[7,7,0.4,1], width_ratios=[0.94,0.94,0.94,3,3])
    ax_wind = fig.add_subplot(gs_top2[0])
    ax_stab = fig.add_subplot(gs_top2[1])
    ax_t    = fig.add_subplot(gs_top2[2])
    
    ax0 = axes[0,1] # xz
    ax1 = axes[1,0] # xy
    ax2 = axes[1,1] # xy2
    ax3 = axes[1,2] # xy2
    axlid = axes[0,2] # lid

    ax0.grid(False)
    axlid.grid(False)

    # iprof = int(ds.nx/2) # CORAL
    # jprof = int(ds.ny/2 - 125) # CORAL
    iprof = int(ds_lidars[0].i)
    jprof = int(ds_lidars[0].j)
    # iprof = int(ds.nx/2 - 80)
    # jprof = int(ds.ny/2)
    if t==0:
        print(f"[i]  Vertical profiles: i={iprof}, j={jprof}")
        print(f"[i]  Vertical profiles: x={ds_lidars[0].xpos}km, j={ds_lidars[0].ypos}km")
    
    """Wind axis"""
    cu = 'darkorchid'
    cv = 'lightseagreen'
    # wind_lims = [-34,34]
    wind_lims = [-90,90]
    x0=0
    # print(np.max(dsxz['u'][t,:,iprof+15]-dsxz['ue'][t,:,iprof+15]))
    ax_wind.plot(dsxz['ue'][t,:,iprof], dsxz.zcr[t,:,iprof], lw=2, ls='--', color=cu, label=r'u$_{env}$')
    ax_wind.plot(np.mean(dsxz['u'][t,:,:],axis=1), dsxz.zcr[t,:,x0], lw=2, ls='dotted', color=cu, label=r'u$_{mean}$')
    # ax_wind.plot(ds_lidars[0]['u'][3*t,:], ds_lidars[0]['zcr'][3*t,:], lw=1, ls='dotted', color=cu, label='u')
    ax_wind.plot(dsxz['u'][t,:,iprof], dsxz.zcr[t,:,iprof], lw=1, ls='-', color=cu, label=r'u$_{mtn}$')
    ax_wind.plot(dsxz['ve'][t,:,iprof], dsxz.zcr[t,:,iprof], lw=2, ls='--', color=cv, label=r'v$_{env}$')
    ax_wind.plot(dsxz['v'][t,:,iprof], dsxz.zcr[t,:,iprof], lw=1, ls='-', color=cv, label=r'v$_{mtn}$')
    ax_wind.set_xlabel('(u,v) / m$\,$s$^{-1}$')
    ax_wind.set_ylabel('altitude z / km')
    ax_wind.set_ylim(zlim)
    ax_wind.set_xlim(wind_lims)
    
    ax_wind.xaxis.set_label_position('top')
    ax_wind.vlines(x=[0], ymin=0,ymax=zlim[1], colors="grey", lw=0.75, ls='-.')
    ##ax_wind.xaxis.set_major_locator(MultipleLocator(50))
    ax_wind.tick_params(axis='x', which='both', top=True, bottom=True, labelbottom=False, labeltop=True, labelleft=True)
    ax_wind.xaxis.set_minor_locator(AutoMinorLocator())
    ax_wind.yaxis.set_minor_locator(AutoMinorLocator())
    ax_wind.legend(loc="upper left", fontsize=7)
    ax_wind.grid()
    ax_wind.text(xpp, ypp, numb_str[ipp], transform=ax_wind.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1
    
    """N^2 axis"""
    u = dsxz['ue'][t,:,x0]
    th0 = dsxz['the'][t,:,x0].values + dsxz['th'][t,:,x0].values
    dthdz0 = np.gradient(th0, dsxz.zcr[t,:,x0].values*1000)
    bvf0 = (ds.g/th0*dthdz0)**0.5
    th = dsxz['the'][t,:,iprof].values + dsxz['th'][t,:,iprof].values
    dthdz = np.gradient(th, dsxz.zcr[t,:,iprof].values*1000)
    bvf = np.sqrt(ds.g/th*dthdz)
    ax_stab.plot(bvf, dsxz.zcr[t,:,iprof], lw=1.5, ls='-', color="firebrick")
    ax_stab.plot(bvf0, dsxz.zcr[t,:,x0], lw=1.5, ls='--', color="firebrick")
    ax_stab.set_xlabel('N / s$^{-1}$')
    ax_stab.xaxis.set_label_position('top')
    ax_stab.set_xlim(0,0.03)
    ax_stab.vlines(x=[0], ymin=0,ymax=zlim[1], colors="grey", lw=0.75, ls='-.')
    ##ax_wind.xaxis.set_major_locator(MultipleLocator(50))
    ax_stab.tick_params(which='both', top=True, bottom=True, labelbottom=False, labeltop=True, labelleft=False)
    ax_stab.xaxis.set_minor_locator(AutoMinorLocator())
    ax_stab.yaxis.set_minor_locator(AutoMinorLocator())
    ax_stab.set_ylim(zlim)
    ax_stab.grid()
    ax_stab.text(xpp, ypp, numb_str[ipp], transform=ax_stab.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1
    
    """Temperature and temperature gradient (mark altitudes with gradient below threshold)"""
    # ---- TEMPERATURE ------------------ #   
    # maybe use ppe??
    zcr = dsxz['zcr'][t,:,iprof].values
    the = dsxz['the'][t,:,iprof].values
    ppe = dsxz['ppe'][t,:,iprof].values
    thloc = the + dsxz['th'][t,:,iprof].values
    ploc = ppe + dsxz['p'][t,:,iprof].values
    tloc = thloc*(ploc/ds.pref00)**ds.cap
    tte = the*(ppe/ds.pref00)**ds.cap
    # tgrad = np.gradient(tloc,zcr)
    # tprime = tloc-tte

    # fill between for altitude bands with gradient below dry lapse rate (<-1K/100m) -> unstable
    # plot tgrad, too. second x axis
    # ---- TEMPERATURE ------------------ # 
    
    
    ax_t.plot(tloc, zcr, lw=1.5, ls='-', color="coral")
    ax_t.plot(tte, zcr, lw=1.5, ls='--', color="coral")
    ax_t.set_xlabel('T / K')
    ax_t.xaxis.set_label_position('top')
    ax_t.set_xlim([155,295])
    # ax_grad.vlines(x=[0], ymin=0,ymax=zlim[1], colors="grey", lw=0.75, ls='-.')
    ##ax_wind.xaxis.set_major_locator(MultipleLocator(50))
    ax_t.tick_params(which='both', top=True, bottom=True, labelbottom=False, labeltop=True, labelleft=False, labelright=False)
    ax_t.xaxis.set_minor_locator(AutoMinorLocator())
    ax_t.yaxis.set_minor_locator(AutoMinorLocator())
    ax_t.set_ylim(zlim)
    ax_t.grid()
    ax_t.text(xpp, ypp, numb_str[ipp], transform=ax_t.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1
    xpp = 0.96
    
    """Plot xz-slice"""
    jslc = int(dsxz.j)
    if var == "mf":
        if t==0:
            print(f"[i]  Filter with $\\lambda_x$ = {lambdax} and $\\lambda_z$={lambdaz}")
        uw   = dsxz['w'][t,:,:].values * (dsxz['u'][t,:,:].values-dsxz['ue'][t,:,:].values)
        mfx  = dsxz['rh0'][t,:,:].values * uw
        mfx = plt_helper.gaussian_filter_fft(mfx, lambdaz, lambdax, ds.dz00, ds.dx00)
        cvar = mfx*1000
    elif var == "u":
        cvar = dsxz["u"][t,:,:].values - dsxz["ue"][t,:,:].values
    else:
        cvar = dsxz[var][t,:,:]
    contf = ax0.contourf(ds.xcr.expand_dims({'z':dsxz.z},axis=0)[:,jslc,:], dsxz.zcr[t,:,:], cvar,
                            cmap=cmap, norm=norm, levels=clev, extend='both')

    isentropes = ax0.contour(ds.xcr.expand_dims({'z':dsxz.z},axis=0)[:,jslc,:], dsxz.zcr[t,:,:], dsxz['the'][t,:,:]+dsxz['th'][t,:,:], 
                             colors='k', alpha=0.7, levels=thlev)
    ax0.plot(ds.xcr[jprof], surf_factor*dsxz.zcr[t,0,:], lw=2, color='black')
    ax0.yaxis.set_major_locator(MultipleLocator(10))
    ax0.xaxis.set_minor_locator(AutoMinorLocator())
    ax0.yaxis.set_minor_locator(AutoMinorLocator())
    ax0.tick_params(labelbottom=False,labeltop=False)
    ax0.set_xlabel('streamwise x / km') # change to longitudes, latitude 10$^3$
    ax0.xaxis.set_label_position('top') 
    
    ##ax0.set_ylabel('altitude z-z$_{trp}$ / km')
    ax0.set_xlim(xlim)
    ax0.set_ylim(zlim)
    ax0.tick_params(which='both', top=True, right=True, bottom=False, labelbottom=False, labeltop=True, labelleft=False, labelright=False)

    # ax0.vlines(x=[xlid0], ymin=zlim[0],ymax=zlim[1], colors='black', lw=lw1, ls='--')
    ax0.axhline(y=ds_xyslices[0].zcr[0,0,0].values, color='black', lw=lw1, ls='--')
    ax0.axhline(y=ds_xyslices[1].zcr[0,0,0].values, color='black', lw=lw1, ls='--')
    ax0.axhline(y=ds_xyslices[2].zcr[0,0,0].values, color='black', lw=lw1, ls='--')
    ax0.text(1-xpp, ypp, f"y: {dsxz.ypos}", transform=ax0.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ax0.text(xpp, ypp, numb_str[ipp], transform=ax0.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1

    """Plot virtual lidars"""
    axlid = plot_virtual_lidar(axlid, ds_lid)
    axlid.text(xpp, ypp, numb_str[ipp], transform=axlid.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1

    # axlid1 = plot_virtual_lidar(axlid1, ds_lid1, tstamp, xlim_lid, zlim, thlev)
    # axlid1.tick_params(which='both', labeltop=False, labelbottom=True)
    # axlid1.xaxis.set_label_position('bottom')
    # axlid1.set_xlabel('time / h')
        
    """Plot xy-slices"""
    ax1 = plt_xyslc(ax1, ds_xyslices[0], ds, dsxz, ds_lid, t)
    ax2 = plt_xyslc(ax2, ds_xyslices[1], ds, dsxz, ds_lid, t)
    ax3 = plt_xyslc(ax3, ds_xyslices[2], ds, dsxz, ds_lid, t)

    ax2.tick_params(labelleft=False)
    ax3.tick_params(labelleft=False,labelright=True)
    ax3.yaxis.set_label_position('right')
    ax1.set_ylabel('spanwise y / km')
    ax3.set_ylabel('spanwise y / km')

    ax1.text(xpp, ypp, numb_str[ipp], transform=ax1.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1
    ax2.text(xpp, ypp, numb_str[ipp], transform=ax2.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1
    ax3.text(xpp, ypp, numb_str[ipp], transform=ax3.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1

    # - Sponge layer - #
    if zsponge[0] > 0:
        ax0.fill_between(xlim, [zsponge[1],zsponge[1]], [zsponge[0],zsponge[0]], facecolor=csponge, alpha=alpha_sponge)
        axlid.fill_between(xlim_lid, [zsponge[1],zsponge[1]], [zsponge[0],zsponge[0]], facecolor=csponge, alpha=alpha_sponge)
    ax2.axvspan(ds.xcr[0,0], ds.xcr[0,0]+ds.dxabL/1000, alpha=alpha_sponge, color=csponge)
    ax2.axvspan(ds.xcr[0,-1]-ds.dxabR/1000, ds.xcr[0,-1], alpha=alpha_sponge, color=csponge)
    xsponge = [ds.xcr[0,0] + ds.dxabL/1000, ds.xcr[0,-1] - ds.dxabR/1000]
    ax2.fill_between(xsponge, [ds.ycr[0,0]+ds.dyab/1000,ds.ycr[0,0]+ds.dyab/1000], [ds.ycr[0,0],ds.ycr[0,0]], facecolor=csponge, alpha=alpha_sponge)
    ax2.fill_between(xsponge, [ds.ycr[-1,0],ds.ycr[-1,0]], [ds.ycr[-1,0]-ds.dyab/1000,ds.ycr[-1,0]-ds.dyab/1000], facecolor=csponge, alpha=alpha_sponge)

    """Colorbar"""
    cbar = fig.colorbar(contf, ax=axes[-1,1:3], location='bottom', shrink=0.67, fraction=0.8, ticks=clev_l, pad=0, extend='both', aspect=33) #  pad=0.15 default
    cbar.set_label(cbar_label)

    """Save figure"""
    os.makedirs(image_folder,exist_ok=True)
    if t<10:
        buffer = "00"
    elif t<100:
        buffer = "0"
    else:
        buffer = ""
    fig_title = "slice_" + buffer + str(t) + ".png"
    fig.savefig(os.path.join(image_folder,fig_title), facecolor='w', edgecolor='w',
                    format='png', dpi=120, bbox_inches='tight')
    """Finish"""
    plt_helper.show_progress(pbar['progress_counter'], pbar['lock'], pbar["stime"], pbar['ntasks'])
    # sema.release()


def plt_xyslc(ax, dsxy, ds, dsxz, ds_lid, t):
    """Plot xy-slice"""

    if var == "mf":
        uw   = dsxy['w'][t,:,:].values * (dsxy['u'][t,:,:].values-dsxy['ue'][t,:,:].values)
        vw   = dsxy['w'][t,:,:].values * (dsxy['v'][t,:,:].values-dsxy['ve'][t,:,:].values)
        mfx  = dsxy['rh0'][t,:,:].values * uw
        mfy  = dsxy['rh0'][t,:,:].values * vw
        mfx = plt_helper.gaussian_filter_fft(mfx,lambdax, lambdax, ds.dx00, ds.dx00)
        mfy = plt_helper.gaussian_filter_fft(mfy,lambdax, lambdax, ds.dx00, ds.dx00)
        cvar = mfx*1000
        # efx = (dsxy['u'][t,:,:]-dsxy['ue'][t,:,:]) * ds['pprime'][t,:,:]
        # efy = (dsxy['v'][t,:,:]-dsxy['ve'][t,:,:]) * ds['pprime'][t,:,:]
        # efx = subroutines.fft_gaussian_xy(efx,nx_avg)
        # efy = subroutines.fft_gaussian_xy(efy,nx_avg)
        # efmax = np.max(np.sqrt(efx**2+efy**2))
        # print(efmax)
        # efx = efx/efmax
        # efy = efy/efmax
        
        # efx = np.where(np.abs(efy)>0.165,efx,np.nan) # 0.07
        # efy = np.where(np.abs(efy)>0.165,efy,np.nan)
    elif var == "u":
        cvar = dsxy["u"][t,:,:] - dsxy["ue"][t,:,:]
    else:
        cvar = dsxy[var][t,:,:]
    ax.contourf(ds.xcr, ds.ycr, cvar, cmap=cmap, norm=norm, levels=clev, extend='both')
    
    ### - Topography - ###
    if ds.itopo == 1:
        amp = dsxy.zcrtopo.max()
        amp = 1200
        topo_levels=np.linspace(20,surf_factor*amp,3)
    else:
        if ds.amp < 0:
            topo_levels=np.linspace(surf_factor*ds.amp,-surf_factor*ds.amp,12)
        else: 
            topo_levels=np.linspace(-2*surf_factor*ds.amp,2*surf_factor*ds.amp,24)
    ax.contour(ds.xcr, ds.ycr, surf_factor*dsxy.zcrtopo[t,:,:], colors='k', levels=topo_levels, linewidths=0.3)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('streamwise x / km')
    ax.set_xlim(xlim)
    ax.grid()

    # - Lidar and Slice positions - #
    ax.text(ds_lid.xpos, ds_lid.ypos, "e", weight='bold', fontsize=8, bbox={"boxstyle" : "circle", "lw":0.4, "facecolor":"white", "edgecolor":"black"})
    ax.axhline(y=dsxz.ypos, color='black', lw=lw1, ls='--')
    # ax.axvline(x=dsyz.xpos, color='black', lw=lw1, ls='--')
    ax.text(1-xpp, ypp, f"z: {dsxy.zpos}km", transform=ax.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    return ax


def plot_virtual_lidar(axlid, ds_lid):
    ## tref_lid = np.where(ds_lid.time.values[:,0] == tstamp_ref)[0]
    if var == "mf":
        uw   = ds_lid['w'].values * (ds_lid['u'].values-ds_lid['ue'].values)
        mfx  = ds_lid['rh0'].values * uw
        mfx = plt_helper.gaussian_filter_fft(mfx, 1, lambdaz, 1, ds.dz00)
        cvar = mfx*1000
    elif var == "u":
        cvar = ds_lid["u"] - ds_lid["ue"]
    else:
        cvar = ds_lid[var]
    axlid.contourf(ds_lid.time, ds_lid.zcr, cvar, levels=clev,
                        cmap=cmap, norm=norm, extend='both')

    isentropes = axlid.contour(ds_lid.time, ds_lid.zcr, ds_lid.the + ds_lid.th, colors='k', levels=thlev)
    # ax.clabel(isentropes, thlev[1::], fontsize=8, fmt='%1.0f K', inline_spacing=1, inline=True, 
    #             manual=[(8,ds.zcr[t,10,0,x]), (8,ds.zcr[t,-15,0,x])]) # ha='left', thlev[1::3]

    axlid.plot(ds_lid.time, surf_factor*ds_lid.zcr[:,0], lw=2, color='black')
    
    axlid.yaxis.set_major_locator(MultipleLocator(10))
    axlid.xaxis.set_minor_locator(AutoMinorLocator())
    axlid.yaxis.set_minor_locator(AutoMinorLocator())
    axlid.tick_params(labelbottom=False,labeltop=True, labelleft=False, labelright=True)
    axlid.xaxis.set_label_position('top')
    axlid.yaxis.set_label_position('right')
    axlid.set_xlabel('time / h')
    axlid.set_ylabel('altitude z / km')
    axlid.set_xlim(xlim_lid)
    axlid.set_ylim(zlim)
    axlid.vlines(x=[tstamp], ymin=zlim[0],ymax=zlim[1], colors='black', lw=2, ls='--')
    # axlid.spines['bottom'].set_color(ds_lid.color)
    # axlid.spines['top'].set_color(ds_lid.color) 
    # axlid.spines['right'].set_color(ds_lid.color)
    # axlid.spines['left'].set_color(ds_lid.color)
    # lw_axlid = 1.5
    # axlid.spines['bottom'].set_linewidth(lw_axlid)
    # axlid.spines['top'].set_linewidth(lw_axlid) 
    # axlid.spines['right'].set_linewidth(lw_axlid)
    # axlid.spines['left'].set_linewidth(lw_axlid)

    axlid.text(1-xpp, ypp, f"x: {ds_lid.xpos}km, y: {ds_lid.ypos}km", transform=axlid.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    return axlid


if __name__ == '__main__':
    """Generate animation of EULAG simulation based on NETCDF slice and lidar output"""

    """Example: 
        >> python3 slc_and_lid.py t pata_tide <50>
        >> python3 slc_and_lid.py mf pata_tide <50>
    """
    
    """Try changing working directory for Crontab"""
    try:
        os.chdir(os.path.dirname(sys.argv[0]))
    except:
        print('[i]  Working directory already set!')
    
    var = sys.argv[1]
    simulation = sys.argv[2]
    if len(sys.argv) > 3:
        ncpus = int(sys.argv[3])
    else: 
        ncpus = mp.cpu_count()-2 # use maximum here but check number of tasks --> ntasks
    fpath = os.path.join(data_folder,simulation)
    image_folder = os.path.join(animation_folder, simulation)
    if var != "th":
        image_folder = image_folder + "_" + var
    ds, ds_env, ds_xzslices, ds_yzslices, ds_xyslices, ds_lidars, ds_full = plt_helper.preprocess_eulag_output(fpath)

    """Parallel processing"""
    progress_counter = mp.Manager().Value('i', 0)
    lock = mp.Manager().Lock()
    stime = time.time()
    pbar = {"progress_counter": progress_counter, "lock": lock, "stime": stime}

    args_list = []
    for t in range(0,np.shape(ds_xzslices[0]['th'])[0]):
        args = (t, var, fpath, image_folder, pbar)
        args_list.append(args)
    pbar['ntasks'] = len(args_list)

    ncpus = np.min([ncpus, pbar['ntasks']])
    print(f"[i]  CPUs available: {mp.cpu_count()}")
    print(f"[i]  CPUs for visualization: {ncpus}")

    with mp.Pool(processes=ncpus) as pool:
        pool.starmap(slc_and_lid, args_list)

    plt_helper.create_animation(image_folder, "animation_" + simulation + "_" + var + ".mp4")

    # sema = mp.Semaphore(config.getint("General","ncpus"))
    # running_procs = []
    # for args in args_list:
    #     for p in running_procs[:]:
    #         if not p.is_alive():
    #             p.join()
    #             running_procs.remove(p)
    #     sema.acquire()
    #     proc = mp.Process(target=vis_slice_and_lid, args=args)
    #     running_procs.append(proc)
    #     proc.start()
    # for proc in running_procs:
    #     proc.join()