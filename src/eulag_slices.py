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

import subroutines
import vis_eulag
#importlib.reload(subroutines)
#importlib.reload(vis_eulag)
#from vis_eulag import *

import cmaps, plt_helper

"""Config"""
pbar_interval = 5 # %
surf_factor = 5
data_folder = "/scratch/b/b309199"
# data_folder = "/work/bd0620/b309199"

if os.path.exists('latex_default.mplstyle'):
    plt.style.use('latex_default.mplstyle')


def vis_slice_and_lid(t, fpath, image_folder, pbar):
    """Visualize u,v,th or other vars in vertical profiles, different cross sections and virtual lidar time-height diagrams"""

    ds, ds_env, ds_xzslices, ds_yzslices, ds_xyslices, ds_lidars, ds_full = preprocess_eulag_output(fpath)

    """Add two simulations"""
    """
    fpath  = os.path.join(folder,"twomtns")
    fpath1 = os.path.join(folder,"twomtns1")
    fpath2 = os.path.join(folder,"twomtns2")
    ds, ds_env, ds_xzslices, ds_yzslices, ds_xyslices, ds_lidars, ds_full = preprocess_eulag_output(fpath)
    ds1, ds_env1, ds_xzslices1, ds_yzslices1, ds_xyslices1, ds_lidars1, ds_full1 = preprocess_eulag_output(fpath1)
    ds2, ds_env2, ds_xzslices2, ds_yzslices2, ds_xyslices2, ds_lidars2, ds_full2 = preprocess_eulag_output(fpath2)
    for ds00, ds11, ds22 in zip(ds_xzslices,ds_xzslices1,ds_xzslices2):
        ds00["th"] = ds00["th"] - (ds11["th"] + ds22["th"])
    for ds00, ds11, ds22 in zip(ds_yzslices,ds_yzslices1,ds_yzslices2):
        ds00["th"] = ds00["th"] - (ds11["th"] + ds22["th"])
    for ds00, ds11, ds22 in zip(ds_xyslices,ds_xyslices1,ds_xyslices2):
        ds00["th"] = ds00["th"] - (ds11["th"] + ds22["th"])
    for ds00, ds11, ds22 in zip(ds_lidars,ds_lidars1,ds_lidars2):
        ds00["th"] = ds00["th"] - (ds11["th"] + ds22["th"])
    """
    
    """Lidar locations"""
    ilid0 = 0
    ilid1 = 1
    if ilid0 != None:
        ds_lid0 = ds_lidars[ilid0]
        xlim_lid = [0,ds_lid0.time.max().values]
    if ilid1 != None:
        ds_lid1 = ds_lidars[ilid1]
    
    """Slices"""
    dsxz = ds_xzslices[1]
    dsyz = ds_yzslices[1]
    dsxy = ds_xyslices[1]
    dsxz['zcr'] = dsxz['zcr']/1000
    dsyz['zcr'] = dsyz['zcr']/1000
    
    """Limits"""
    # tref = 28
    # tstamp_ref = tref * (ds_lid.time.max().values / (np.shape(ds['th'])[0]-1))
    # tstamp     = t * (ds_lid.time.max().values / (np.shape(ds['th'])[0]-1))
    # tstamp_ref = tref * ds.nslice * ds.dt00 / 3600
    tstamp     = t    * ds.nslice * ds.dt00 / 3600

    zlim      = [-2,dsxz.zcr.max().values]
    xlim      = [ds.xcr.min().values,ds.xcr.max().values]
    ylim      = [ds.ycr.min().values,ds.ycr.max().values]
    zsponge   = [ds.zab/1000, zlim[1]]
    
    """Plot parameter"""
    c2 = 'forestgreen'
    c3 = 'lightgrey'
    c4 = 'darkorchid'
    c5 = 'lightseagreen'
    lw_1 = 2
    lw_2 = 1.5
    lw_sponge = 1.5
    alpha_box = 0.9
    alpha_sponge = 0.7

    """Colormap"""
    cmap   = cmaps.get_wave_cmap()
    # clev   = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32]
    # clev_l = [-16,-4,-1,1,4,16]
    clev, clev_l = subroutines.get_colormap_bins_and_labels(max_level=32)
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)

    """Labels"""
    global xpp, ypp
    xpp = 0.9
    ypp = 0.93
    ipp = 0
    numb_str = ['a','b','c','d','e','f','g','h','i','j']
    
    """Theta levels"""
    # if config.getint("General", "stratos") == 1:
    #    thlev=np.exp(5+0.03*np.arange(1,250,10)) # for N=0.02
        # thlev=np.exp(5+0.03*np.arange(1,300,30)) # for N=0.02
    # else:
    #     thlev=np.exp(5+0.03*np.arange(1,100)) # 90 is needed for atmosphere up to 100km
        # thlev=np.exp(5+0.03*np.arange(1,100,0.5)) # more isentropes for troposphere visualization

    #thlev=np.exp(5+0.03*np.arange(1,250,10)) # N=0.02
    #thlev=np.exp(5+0.03*np.arange(1,100))    # N=0.01
    thlev=np.exp(1+0.03*np.arange(1,350,5)) # N=0.02


    """Figure stuff"""
    gskw  = {'hspace':0.1, 'wspace':0.04, 'height_ratios': [7,7,0.4,2], 'width_ratios': [3,3,3]} #  , 'width_ratios': [5,5]}
    # gskw2 = {'hspace':0.14, 'wspace':0.06, 'height_ratios': [7,7,0.1,2], 'width_ratios': [1,1,1,3,3]}
    fig, axes = plt.subplots(4,3,figsize=(15,10), gridspec_kw=gskw)
    for ax in axes[-1,0:3]:
        ax.set_axis_off()
    for ax in axes[-2,0:3]:
        ax.set_axis_off()

    # --- Replace top left axis --- #
    gs_topleft = axes[0,0].get_gridspec()
    axes[0,0].remove()
    
    gs_top2 = fig.add_gridspec(4,5, hspace=0.1, wspace=0.04, height_ratios=[7,7,0.4,2], width_ratios=[0.94,0.94,0.94,3,3])
    ax_wind = fig.add_subplot(gs_top2[0])
    ax_stab = fig.add_subplot(gs_top2[1])
    ax_t    = fig.add_subplot(gs_top2[2])
    
    ax0 = axes[0,1] # xz
    ax1 = axes[1,1] # xy
    ax2 = axes[1,0] # xy2
    axlid0 = axes[0,2] # lid
    axlid1 = axes[1,2] # lid2
    ax0.grid(False)
    axlid0.grid(False)
    axlid1.grid(False)

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
    # wind_lims = [-34,34]
    wind_lims = [-90,90]
    x0=0
    # print(np.max(dsxz['u'][t,:,xi]))
    ax_wind.plot(dsxz['ue'][t,:,x0], dsxz.zcr[t,:,x0], lw=2, ls='--', color=c4)
    # ax_wind.plot(np.mean(dsxz['u'][t,:,:],axis=1), dsxz.zcr[t,:,x0], lw=1, ls='-', color=c4, label='u')
    ax_wind.plot(dsxz['u'][t,:,iprof], dsxz.zcr[t,:,iprof], lw=1, ls='-', color=c4, label='u')
    ax_wind.plot(dsxz['ve'][t,:,x0], dsxz.zcr[t,:,x0], lw=2, ls='--', color=c5)
    ax_wind.plot(dsxz['v'][t,:,iprof], dsxz.zcr[t,:,iprof], lw=1, ls='-', color=c5, label='v')
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
    bvf = (ds.g/th*dthdz)**0.5
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
    var = dsxz.th[t,:,:]
    jslc = int(dsxz.j)
    pcMesh0 = ax0.contourf(ds.xcr.expand_dims({'z':dsxz.z},axis=0)[:,jslc,:], dsxz.zcr[t,:,:], var,
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

    # ax0.vlines(x=[xlid0], ymin=zlim[0],ymax=zlim[1], colors='black', lw=lw_1, ls='--')
    ax0.axhline(y=dsxy.zcr[0,0,0].values, color='black', lw=lw_1, ls='--')
    ax0.text(1-xpp, ypp, f"y: {dsxz.ypos}", transform=ax0.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ax0.text(xpp, ypp, numb_str[ipp], transform=ax0.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1

    """Plot virtual lidars"""
    if ilid0 != None:
        axlid0 = plot_virtual_lidar(axlid0, ds_lid0, tstamp, xlim_lid, zlim, thlev, clev, cmap, norm)
        axlid0.text(xpp, ypp, numb_str[ipp], transform=axlid0.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
        ipp += 1
    else:
        axlid0.set_axis_off()
    if ilid1 != None:
        axlid1 = plot_virtual_lidar(axlid1, ds_lid1, tstamp, xlim_lid, zlim, thlev, clev, cmap, norm)
        axlid1.tick_params(which='both', labeltop=False, labelbottom=True)
        axlid1.xaxis.set_label_position('bottom')
        axlid1.set_xlabel('time / h')
        
    else:
        axlid1.set_axis_off()
        
    """Plot yz-slice"""
    var = dsyz.th[t,:,:]
    islc = int(dsyz.i)
    ax2.contourf(ds.ycr.expand_dims({'z':dsyz.z},axis=0)[:,:,0], dsyz.zcr[t,:,:], var,
                            cmap=cmap, norm=norm, levels=clev, extend='both')
    isentropes = ax2.contour(ds.ycr.expand_dims({'z':dsyz.z},axis=0)[:,:,0], dsyz.zcr[t,:,:], dsyz['the'][t,:,:]+dsyz['th'][t,:,:], 
                             colors='k', alpha=0.7, levels=thlev)
    ax2.plot(ds.ycr[:,islc], surf_factor*dsyz.zcr[t,0,:], lw=2, color='black')
    
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_xlabel('spanwise y / km')
    ax2.set_ylabel('altitude z / km')
    ax2.set_ylim(zlim)
    ax2.grid()
    ax2.text(1-xpp, ypp, f"x: {dsyz.xpos}km", transform=ax2.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ax2.text(xpp, ypp, numb_str[ipp], transform=ax2.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1

    """Plot xy-slice"""
    var = dsxy.th[t,:,:]
    ax1.contourf(ds.xcr, ds.ycr, var,
                            cmap=cmap, norm=norm, levels=clev, extend='both')
    ### - Topography - ###
    if ds.amp < 0:
        topo_levels=np.linspace(surf_factor*ds.amp,-surf_factor*ds.amp,12)
    else: 
        topo_levels=np.linspace(-surf_factor*ds.amp,surf_factor*ds.amp,12)
    isentropes = ax1.contour(ds.xcr, ds.ycr, surf_factor*dsxy.zcrtopo[t,:,:], colors='k', levels=topo_levels, linewidths=1)
    ##isentropes = ax2.contour(ds.xcr, ds.ycr, -surf_factor*ds_env.zcr[-1,0,:,:], colors='k', levels=topo_levels, linewidths=1)
    
    ###@cmb include sponge in x and y
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis="y",direction="in", pad=-25)
    ax1.set_xlabel('streamwise x / km')
    ax1.set_ylabel('spanwise y / km', labelpad=-20)
    ax1.set_xlim(xlim)
    ax1.grid()
    ## include horizontal line in xz plot and vice versa in xy 
    ax1.scatter(ds_lid0.xpos, ds_lid0.ypos, s=80, c=ds_lid0.color, marker = "D")
    ax1.scatter(ds_lid1.xpos, ds_lid1.ypos, s=80, c=ds_lid1.color, marker = "D")
    ax1.axhline(y=0, color='black', lw=lw_1, ls='--')
    ax1.axvline(x=dsyz.xpos, color='black', lw=lw_1, ls='--')
    ax1.text(1-xpp, ypp, f"z: {dsxy.zpos}km", transform=ax1.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ax1.text(xpp, ypp, numb_str[ipp], transform=ax1.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1

    """Sponge layer"""
    #ax0.axhline(y=48, lw=lw_2,ls='--',color='grey')
    #ax1.axhline(y=48, lw=lw_2,ls='--',color='grey')
    if zsponge[0] > 0:
        ax0.fill_between(xlim, [zsponge[1],zsponge[1]], [zsponge[0],zsponge[0]], facecolor=c3, alpha=alpha_sponge)
        ax2.fill_between(ylim, [zsponge[1],zsponge[1]], [zsponge[0],zsponge[0]], facecolor=c3, alpha=alpha_sponge)
        if ilid0 != None:
            axlid0.fill_between(xlim_lid, [zsponge[1],zsponge[1]], [zsponge[0],zsponge[0]], facecolor=c3, alpha=alpha_sponge)
        if ilid1 != None:
            axlid1.fill_between(xlim_lid, [zsponge[1],zsponge[1]], [zsponge[0],zsponge[0]], facecolor=c3, alpha=alpha_sponge)
    ax1.fill_between(xlim, [ds.ycr[0,0]+ds.dyab/1000,ds.ycr[0,0]+ds.dyab/1000], [ds.ycr[0,0],ds.ycr[0,0]], facecolor=c3, alpha=alpha_sponge)
    ax1.fill_between(xlim, [ds.ycr[-1,0],ds.ycr[-1,0]], [ds.ycr[-1,0]-ds.dyab/1000,ds.ycr[-1,0]-ds.dyab/1000], facecolor=c3, alpha=alpha_sponge)

    ##sponge_label = r'$\uparrow$ sponge layer $\uparrow$'
    ##axes[0,0].text(0.6, 0.72, sponge_label, transform=axes[0,0].transAxes, weight='bold', color='grey')

    """Lidar 2 label"""
    axlid1.text(xpp, ypp, numb_str[ipp], transform=axlid1.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1

    """Colorbar"""
    cbar = fig.colorbar(pcMesh0, ax=axes[-1,:], location='bottom', shrink=0.67, fraction=1, ticks=clev_l, pad=0, extend='both', aspect=28) #  pad=0.15 default
    cbar.set_label(r"$\Theta'$ / K")

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

    
def plot_virtual_lidar(axlid, ds_lid, tstamp, xlim_lid, zlim, thlev, clev, cmap, norm):
    ## tref_lid = np.where(ds_lid.time.values[:,0] == tstamp_ref)[0]
    ## var = ds_lid.th-ds_lid.th[tref_lid]
    var = ds_lid.th
    pcMesh1 = axlid.contourf(ds_lid.time, ds_lid.zcr, var, levels=clev,
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
    axlid.spines['bottom'].set_color(ds_lid.color)
    axlid.spines['top'].set_color(ds_lid.color) 
    axlid.spines['right'].set_color(ds_lid.color)
    axlid.spines['left'].set_color(ds_lid.color)
    lw_axlid = 1.5
    axlid.spines['bottom'].set_linewidth(lw_axlid)
    axlid.spines['top'].set_linewidth(lw_axlid) 
    axlid.spines['right'].set_linewidth(lw_axlid)
    axlid.spines['left'].set_linewidth(lw_axlid)

    axlid.text(1-xpp, ypp, f"x: {ds_lid.xpos}km, y: {ds_lid.ypos}km", transform=axlid.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    return axlid


def preprocess_eulag_output(fpath):
    """Process EULAG output"""
    env_path   = os.path.join(fpath, "env.nc")
    tapes_path = os.path.join(fpath, "tapes.nc")
    grid_path  = os.path.join(fpath, "grd.nc")
    ds_full = xr.open_dataset(tapes_path)
    ds_env  = xr.open_dataset(env_path)
    ds      = xr.open_dataset(grid_path)
    ds      = ds.assign_coords({'xcr':ds['xcr']/1000, 'ycr':ds['ycr']/1000, 'zcr':ds['zcr']/1000})
    
    # ---- Sim parameters -------------- # 
    ds.attrs['bv'] = ds.attrs['bv'].round(3)
    ds.attrs['nx'] = np.shape(ds_full['w'])[3]
    ds.attrs['ny'] = np.shape(ds_full['w'])[2]
    ds.attrs['nz'] = np.shape(ds_full['w'])[1]
    
    ds.attrs['cp']=3.5*ds.rg # Earth
    ds.attrs['cap']=ds.rg/ds.cp
    ds.attrs['pref00']=101325.
        
    """Slice outputs"""
    # dsxy['zcr'] = dsxy['zcr'] / 1000
    xzslices = sorted(glob.glob(os.path.join(fpath, "xzslc_*")))
    yzslices = sorted(glob.glob(os.path.join(fpath, "yzslc_*")))
    xyslices = sorted(glob.glob(os.path.join(fpath, "xyslc_*")))
    ds_xzslices = []
    ds_yzslices = []
    ds_xyslices = []
    for slc in xzslices:
        ds_slc = xr.open_dataset(slc)
        ds_slc.attrs['j'] = int(slc.split("/")[-1][-8:-3])
        ds_slc.attrs['ypos'] = (ds_slc.j - ds.ny/2) * ds.dy00/1000
        # ds.attrs['pref00'] = ds_slc['pr0'].max()
        
        # ds_slc = ds_slc.assign_coords({ds_slc.zcr:ds_slc.z})
        ds_xzslices.append(ds_slc)
    for slc in yzslices:
        ds_slc = xr.open_dataset(slc)
        ds_slc.attrs['i'] = int(slc.split("/")[-1][-8:-3])
        if ds.ibcx == 0:
            ds_slc.attrs['xpos'] = (ds_slc.i - ds.nx/2)  * ds.dx00/1000
        else:
            ds_slc.attrs['xpos'] = ds_slc.i  * ds.dx00/1000
        ds_yzslices.append(ds_slc)
    for slc in xyslices:
        ds_slc = xr.open_dataset(slc)
        ds_slc['zcr'] = ds_slc['zcr'] / 1000 #km
        ds_slc.attrs['k'] = int(slc.split("/")[-1][-8:-3])
        ds_slc.attrs['zpos'] = ds_slc.k * ds.dz00/1000
        ds_xyslices.append(ds_slc)
    
    """Lidar outputs with high temporal resolution"""
    lid_colors = ["purple", "forestgreen"]
    lidars = sorted(glob.glob(os.path.join(fpath, "lid_*")))
    ds_lidars = []
    for i, lid_file in enumerate(lidars):
        ds_lid = xr.open_dataset(lid_file)
        ds_lid['time'] = ds_lid.t * ds.nlid * ds.dt00/3600
        ds_lid['time'] = ds_lid['time'].expand_dims({'z':ds_lid.z}, axis=1)
        ds_lid['zcr'] = ds_lid['zcr']/1000
        
        loc_str = lid_file.split("/")[-1][:-3]
        ds_lid.attrs['i'] = int(str(loc_str)[4:9])
        ds_lid.attrs['j'] = int(str(loc_str)[-5:])
        if ds.ibcx == 0:
            ds_lid.attrs['xpos'] = (ds_lid.i - ds.nx/2)  * ds.dx00/1000
        else:
            ds_lid.attrs['xpos'] = ds_lid.i * ds.dx00/1000
        ds_lid.attrs['ypos'] = (ds_lid.j - ds.ny/2) * ds.dy00/1000
        ds_lid.attrs['color'] = lid_colors[i]
        ds_lidars.append(ds_lid)
        
    return ds, ds_env, ds_xzslices, ds_yzslices, ds_xyslices, ds_lidars, ds_full



if __name__ == '__main__':
    """Generate animation of EULAG simulation based on NETCDF slice and lidar output"""

    """Example: 
        >> python3 eulag_slices.py pata01
    """
    
    """Try changing working directory for Crontab"""
    try:
        os.chdir(os.path.dirname(sys.argv[0]))
    except:
        print('[i]  Working directory already set!')
    
    simulation = sys.argv[1]
    ncpus = int(sys.argv[2])
    fpath = os.path.join(data_folder,simulation)
    image_folder = os.path.join("./data/animation_slices", simulation)
    ds, ds_env, ds_xzslices, ds_yzslices, ds_xyslices, ds_lidars, ds_full = preprocess_eulag_output(fpath)

    """Parallel processing"""
    progress_counter = mp.Manager().Value('i', 0)
    lock = mp.Manager().Lock()
    stime = time.time()
    pbar = {"progress_counter": progress_counter, "lock": lock, "stime": stime}

    args_list = []
    for t in range(0,np.shape(ds_xzslices[0]['th'])[0]):
        args = (t, fpath, image_folder, pbar)
        args_list.append(args)
    pbar['ntasks'] = len(args_list)

    # ncpus = np.min([mp.cpu_count()-2, pbar['ntasks']])
    print(f"[i]  CPUs available: {mp.cpu_count()}")
    print(f"[i]  CPUs for visualization: {ncpus}")

    with mp.Pool(processes=ncpus) as pool:
        pool.starmap(vis_slice_and_lid, args_list)

    plt_helper.create_animation(image_folder)

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