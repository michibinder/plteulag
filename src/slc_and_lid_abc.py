import sys
import os
import glob
import time
from datetime import datetime
import shutil 

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, FuncFormatter
from matplotlib.ticker import MaxNLocator

# - for reloading libraries/modules - #
import importlib
import multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')

import cmaps, plt_helper, filter

# USAGE
# python3 slc_and_lid.py vortex darwin_240718_400m <notest>

"""Config"""
data_folder = "/scratch/b/b309199"
data_folder = "/work/bd0620/b309199/patagonia"
data_folder = "/work/bd0620/b309199/scratch"
pbar_interval = 5 # %
animation_folder = "../data/animation_slices"

VERTICAL_CUTOFF = 15 # km (LAMBDA_CUT)
TEMPORAL_CUTOFF = 8*60 # min (TAU_CUT)

if os.path.exists('latex_default.mplstyle'):
    plt.style.use('latex_default.mplstyle')

"""Colormap"""
mycmap = {}
mynorm = {}
myclev = {}
myclevl = {}
mylabel = {}

xvar = "vorticity"
# mycmap[xvar] = plt.get_cmap('RdBu')
mycmap[xvar] = plt.get_cmap('PiYG')
# clev_vort, clev_l_vort = plt_helper.get_colormap_bins_and_labels(max_level=0.2)
myclev[xvar] = np.linspace(-90,90,100) * 10**(-3)
myclevl[xvar] = [np.min(myclev[xvar]), np.max(myclev[xvar])]
mynorm[xvar] = BoundaryNorm(boundaries=myclev[xvar], ncolors=mycmap[xvar].N, clip=True)
mylabel[xvar] = "Vorticity / s$^{-1}$"

xvar = "t"
# mycmap[xvar] = plt.get_cmap('seismic')
mycmap[xvar] = cmaps.get_wave_cmap()
# clev_t, clev_l_t = plt_helper.get_colormap_bins_and_labels(max_level=32)
myclev[xvar] = np.linspace(-25,25,100)
# myclev[xvar] = np.linspace(-30,30,100)
# myclev[xvar] = np.linspace(-8,8,100)
myclevl[xvar] = [np.min(myclev[xvar]), np.max(myclev[xvar])]
mynorm[xvar] = BoundaryNorm(boundaries=myclev[xvar], ncolors=mycmap[xvar].N, clip=True)
mylabel[xvar] = r"T' / K"

xvar = "u"
mycmap[xvar] = cmaps.get_wave_cmap()
# mycmap[xvar] = plt.get_cmap('RdBu_r')
myclev[xvar] = np.linspace(-40,40,100)
# myclev[xvar] = np.linspace(-5,5,100)
myclevl[xvar] = [np.min(myclev[xvar]), np.max(myclev[xvar])]
mynorm[xvar] = BoundaryNorm(boundaries=myclev[xvar], ncolors=mycmap[xvar].N, clip=True)
mylabel[xvar] = r"u' / m$\,$s$^{-1}$"

xvar = "w"
# mycmap[xvar] = plt.get_cmap('PIYG')
# mycmap[xvar] = plt.get_cmap('RdBu_r')
mycmap[xvar] = cmaps.get_wave_cmap()
myclev[xvar] = np.linspace(-20,20,100)
# myclev[xvar] = np.linspace(-45,45,100)
myclevl[xvar] = [np.min(myclev[xvar]), np.max(myclev[xvar])]
mynorm[xvar] = BoundaryNorm(boundaries=myclev[xvar], ncolors=mycmap[xvar].N, clip=True)
mylabel[xvar] = r"w / m$\,$s$^{-1}$"

# cmap = cmaps.get_wave_cmap()
# # clev   = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32]
# # clev_l = [-16,-4,-1,1,4,16]
# clev, clev_l = plt_helper.get_colormap_bins_and_labels(max_level=2)
# norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
# cbar_label = r"P' / Pa"

# cmap = plt.get_cmap('YlOrRd')
# # clev = [8,16,32,64,128,256,512,1024]
# clev = [32,64,128,256,512,1024,2048,4096]
# clev_l = clev
# norm = BoundaryNorm(boundaries=clev, ncolors=cmap.N, clip=True)
# cbar_label = r"E$_{pm}$ / J$\,$kg$^{-1}$"

# cmap = plt.get_cmap('PiYG_r')
# cmap = plt.get_cmap('RdBu_r')
# cmap = cmaps.get_wave_cmap()
# clev, clev_l = plt_helper.get_colormap_bins_and_labels(max_level=64)
# cbar_label = r"MF$_x$ / mPa"


def slc_and_lid(t, var, fpath, slices, dslid, image_folder, pbar, sema):
    """Visualize u,v,th or other vars in vertical profiles, different cross sections and virtual lidar time-height diagrams"""
    
    global xlim, ylim, zlim, zsponge, thlev, surf_factor, topo_levels, wind_levels, zcut_mf
    ds, dsxz, dsyz, ds_xyslices = plt_helper.preprocess_eulag_tstep(fpath, t, slices=slices)

    """Limits and timestamp and thlev"""
    zsponge = [ds.zab/1000, dsxz.zcr.max().values]
    if  ds.attrs["itopo"] == 0:
        if t == 0:
            print("[i]  itopo: ", ds.attrs["itopo"], ". Using idealized topography.")
    else:
        if t == 0:
            print("[i]  itopo: ", ds.attrs["itopo"], ". Using realistic topography.")
    
    if region == 'debeto' and var == 'vortex':
        xlim = [-80,100]
        ylim  = [-80,80]
    elif region == 'pata' and var == 'vortex':
        xlim = [-400,-100]
        xlim = [-600, 100] # Fitz roy
        ylim  = [100,400]
    elif region=="darwin" and var == 'vortex':
        xlim = [-50,150]
        xlim = [-150,225]
        ylim  = [-100,100]
    else:
        xlim  = [ds.xcr.min().values,ds.xcr.max().values]
        ylim  = [ds.ycr.min().values,ds.ycr.max().values]

    zcut_mf = 75000
    surf_factor = 5
    thlev = np.exp(4+0.03*np.arange(1,350,5))
    zlim = [0,dsxz.zcr.max().values]
    if var == 'vortex':
        zlim = [45,85]
        zlim = [0,85]
        zlim = [0,110]
        var1 = 'w'
        var1 = 'u'
        # var2 = 'vorticity'
        # var3 = 't'
    elif var == 'amtm':
        zlim = [0,dsxz.zcr.max().values]
        var1 = 'u'
        var2 = 't'
        var3 = 't'
    elif var == 'alima_x' or var == 'alima_y':
        # xlim  = [-200,200]
        # ylim  = [-200,200]
        # zlim = [45,85]
        var1 = 't'
        var2 = 't'
        var3 = 't'
    elif var == "surf":
        xlim  = [-100,100]
        ylim  = [-100,100]
        zlim = [0,6]
        var1 = 'u'
        var2 = 't'
        var3 = 't'
        surf_factor = 1
        thlev=np.exp(4+0.03*np.arange(40,100,0.2))
    else:
        zlim = [0,dsxz.zcr.max().values]
        var1 = 'u'
        var2 = 'vorticity'
        var3 = 't'

    if ds.itopo == 1:
        amp = int(np.max(ds_xyslices[0].zcrtopo))
        topo_levels=np.linspace(20, surf_factor*0.5*amp, 2)
    else:
        if ds.amp < 0:
            topo_levels=np.linspace(surf_factor*ds.amp,-surf_factor*ds.amp,12)
        else: 
            topo_levels=np.linspace(-2*surf_factor*ds.amp,2*surf_factor*ds.amp,24)

    wind_levels = np.arange(-120,120,10)

    """Momentum flux - gaussian filter"""
    global lambdax, lambdaz
    # lambdax = 40 * 1000 # m 
    lambdax = 50 * 1000 # m
    lambdaz =  15 * 1000 # m 

    """Plot parameter"""
    global lw1, lw2, lw_sponge, csponge
    csponge = 'lightgrey'
    lw1 = 1.2
    lw2 = 0.8
    lw_sponge = 1.5
    alpha_box = 0.9
    alpha_sponge = 0.7

    """Labels"""
    global xpp, ypp
    xpp = 0.88
    ypp = 0.93
    ipp = 0
    numb_str = ['a','b','c','d','e','f','g','h','i','j']

    """Figure stuff (panels a, b, c only)"""
    fig = plt.figure(figsize=(12,5.5))
    gs_main = fig.add_gridspec(
        2, 3,
        hspace=0.05, wspace=0.08,
        height_ratios=[7, 0.45],
        width_ratios=[1.6, 1.6, 3.0]
    )

    # Use one top row for panels a, b, c so they have the same height.
    ax_wind = fig.add_subplot(gs_main[0, 0])                 # panel a
    ax_t    = fig.add_subplot(gs_main[0, 1], sharey=ax_wind) # panel b
    ax0     = fig.add_subplot(gs_main[0, 2], sharey=ax_wind) # panel c
    ax_cbar = fig.add_subplot(gs_main[1, 2])                 # colorbar for panel c / var1
    ax_pad  = fig.add_subplot(gs_main[1, 0:2])               # keep balance in lower row
    ax_pad.set_axis_off()
    ax_cbar.set_axis_off()

    ax0.grid(False)

    if 'i' in dslid.attrs and 'j' in dslid.attrs:
        iprof = int(dslid.i)
        # jprof = int(dslid.j)
    else:
        iprof = int(np.shape(dsxz['w'].values)[-1]/2)
        if var == 'alima_y':
            dslid.attrs['xpos'] = dsyz.xpos
            dslid.attrs['ypos'] = alima_locs[t] / 1000
        else:
            dslid.attrs['xpos'] = alima_locs[t] / 1000
            dslid.attrs['ypos'] = dsxz.ypos
    dslid.attrs['kprof'] = ds_xyslices[0].k
        
    """Wind axis"""
    cu = 'darkorchid'
    cv = 'lightseagreen'
    lw_wind = 2
    lws_wind = 0.4
    # wind_lims = [-34,34]
    wind_lims = [-99,99]
    wind_lims = [-177,177]
    wind_lims = [-53,153]
    x0=0
    # print(np.max(dsxz['u'][:,iprof+15]-dsxz['ue'][:,iprof+15]))
    ax_wind.plot(dsxz['ue'][:,iprof], dsxz.zcr[:,iprof], lw=lw_wind, ls='--', color='black', label=r'u$_{env}$')
    
    # ax_wind.plot(np.mean(dsxz['u'][:,1024:-100],axis=1), dsxz.zcr[:,x0], lw=lw_wind, ls='dotted', color='red', label=r'u$_{mean}$')
    umean = dsxz['ue'][:,iprof] - 2.5 * (dsxz['ue'][:,iprof] - np.mean(dsxz['u'][:,:],axis=1))
    umean[187:] = np.mean(dsxz['u'][:,:],axis=1)[187:]
    ax_wind.plot(umean, dsxz.zcr[:,x0], lw=lw_wind, ls='dotted', color='red', label=r'u$_{mean}$')
    
    ax_wind.plot(dsxz['u'][:,iprof], dsxz.zcr[:,iprof], lw=lws_wind, ls='-', color=cu, label=r'u$_{vlidar}$')
    
    #cmb ax_wind.plot(dsxz['ve'][:,iprof], dsxz.zcr[:,iprof], lw=lw_wind, ls='--', color=cv, label=r'v$_{env}$')
    #cmb ax_wind.plot(dsxz['v'][:,iprof], dsxz.zcr[:,iprof], lw=lws_wind, ls='-', color=cv, label=r'v$_{mtn}$')
    
    ax_wind.set_xlabel('(u,v) / m$\,$s$^{-1}$')
    ax_wind.set_ylabel('altitude z / km')
    ax_wind.set_ylim(zlim)
    ax_wind.set_xlim(wind_lims)
    # ax_wind.yaxis.set_major_locator(MultipleLocator(10))
    
    ax_wind.xaxis.set_label_position('top')
    ax_wind.vlines(x=[0], ymin=0,ymax=zlim[1], colors="grey", lw=0.75, ls='-.')
    ##ax_wind.xaxis.set_major_locator(MultipleLocator(50))
    ax_wind.tick_params(axis='x', which='both', top=True, bottom=True, labelbottom=False, labeltop=True, labelleft=True)
    ax_wind.xaxis.set_minor_locator(AutoMinorLocator())
    ax_wind.yaxis.set_minor_locator(AutoMinorLocator())
    ax_wind.legend(loc="lower left", fontsize=10)
    ax_wind.grid()
    ax_wind.text(xpp, ypp, numb_str[ipp], transform=ax_wind.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1
    
    
    """Temperature and temperature gradient (mark altitudes with gradient below threshold)""" 
    zcr = dsxz['zcr'][:,iprof].values
    if var == "mf":
        zcr = dsxz['zcr'][:,0].values
        uw  = dsxz['w'].values * (dsxz['u'].values-dsxz['ue'].values)
        mfx = dsxz['rh0'].values * uw
        mfx = np.mean(mfx, axis=1)
        mfx = filter.gaussian_filter_fft_1D(mfx, lambdaz, ds.dz00)
        mfx = mfx * 10**6
        ax_t.plot(mfx, zcr, lw=1.5, ls='-', color="royalblue")
        # ax_t.plot(tenv, zcr, lw=1.5, ls='--', color="coral")
        ax_t.set_xlabel('MF$_x$ / Pa $\cdot 10^{-6}$')
        ax_t.set_xlim([-10**6,10**3])
        ax_t.set_xscale("symlog")
        ax_t.set_xticks([-10**4,-10**2, 10**2])
    elif var == "ep":
        zcr = dsxz['zcr'][:,0].values
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxz['thprime'].values, dsxz['the'].values, dsxz['pprime'].values, dsxz['ppe'].values, ds.cap, ds.pref00)
        epm = 1/2*(ds.g/ds.bv)**2 * ((tloc-tenv)/tenv)**2
        # epm = plt_helper.gaussian_filter_fft(epm, lambdaz, lambdax, ds.dz00, ds.dx00)
        epm = np.mean(epm, axis=1)
        epm = filter.gaussian_filter_fft_1D(epm, lambdaz, ds.dz00)
        ax_t.plot(epm, zcr, lw=1.5, ls='-', color="red")
        # ax_t.plot(tenv, zcr, lw=1.5, ls='--', color="coral")
        ax_t.set_xlabel(r"E$_{pm}$ / J$\,$kg$^{-1}$")
        ax_t.set_xlim([0,2400])
    else:
        zcr = dsxz['zcr'][:,iprof].values
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxz['thprime'][:,iprof].values, dsxz['the'][:,iprof].values, dsxz['pprime'][:,iprof].values, dsxz['ppe'][:,iprof].values, ds.cap, ds.pref00)
        ax_t.plot(tenv, zcr, lw=1.5, ls='--', color="coral", label=r'T$_{env}$')
        ax_t.plot(tloc, zcr, lw=lw2, ls='-', color="coral", label=r'T$_{vlidar}$')
        ax_t.set_xlabel('T / K')
        ax_t.set_xlim([155,295])

        if var != "surf":
            x = np.linspace(200, 270, 70)
            yref = 62
            y0 = -1/9.8 * (x-200) + yref
            y1 = -1/9.8 * (x-200) + yref + 12
            ax_t.plot(x,y0,lw=1.5,ls='--',color='k')
            ax_t.plot(x,y1,lw=1.5,ls='--',color='k')
            ax_t.text(x[0]-15, y0[0]+2, "-9.8 K/km", horizontalalignment='center') # weight='bold'

    ax_t.legend(loc="lower left", fontsize=10)

    ## tgrad = np.gradient(tloc,zcr)
    ## tprime = tloc-tte

    # ---- TEMPERATURE ------------------ # 
    ax_t.xaxis.set_label_position('top')
    # ax_grad.vlines(x=[0], ymin=0,ymax=zlim[1], colors="grey", lw=0.75, ls='-.')
    ##ax_wind.xaxis.set_major_locator(MultipleLocator(50))
    ax_t.tick_params(which='both', top=True, bottom=True, labelbottom=False, labeltop=True, labelleft=False, labelright=False)
    ax_t.xaxis.set_minor_locator(AutoMinorLocator())
    ax_t.yaxis.set_minor_locator(AutoMinorLocator())
    ax_t.set_ylim(zlim)
    # ax_t.yaxis.set_major_locator(MultipleLocator(10))
    ax_t.grid()
    ax_t.text(xpp, ypp, numb_str[ipp], transform=ax_t.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1
    xpp = 0.96
    
    ############## PANEL C ##############
    """Plot xz-slice"""
    ax0, contf_1 = plt_xzslc(ax0, dsxz, ds, var=var1)
    ax0.axvline(x=dslid.xpos, color='black', lw=lw1, ls='--')
    if var == 'alima_x' or var == 'alima_y':
        ax0.text(dslid.xpos, zlim[0] + 7, '✈', fontsize=30, fontname='DejaVu Sans', ha='center', va='center')
    ax0.text(xpp, ypp, numb_str[ipp], transform=ax0.transAxes, horizontalalignment='right', weight='bold', bbox={"boxstyle" : "circle", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    hrs, rem = divmod(dslid.time[t].values*3600, 3600)
    mins, secs = divmod(rem, 60)
    ax0.text(1-xpp, ypp, f"Time: {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}s ({t:03d})", transform=ax0.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    ipp += 1
    ############## PANEL C ##############

    # - Sponge layer - #
    # if zsponge[0] > 0:
    #     ax0.fill_between(xlim, [zsponge[1],zsponge[1]], [zsponge[0],zsponge[0]], facecolor=csponge, alpha=alpha_sponge)
    #     axlid.fill_between(xlim_lid, [zsponge[1],zsponge[1]], [zsponge[0],zsponge[0]], facecolor=csponge, alpha=alpha_sponge)
    # ax2.axvspan(ds.xcr[0,0], ds.xcr[0,0]+ds.dxabL/1000, alpha=alpha_sponge, color=csponge)
    # ax2.axvspan(ds.xcr[0,-1]-ds.dxabR/1000, ds.xcr[0,-1], alpha=alpha_sponge, color=csponge)
    # xsponge = [ds.xcr[0,0] + ds.dxabL/1000, ds.xcr[0,-1] - ds.dxabR/1000]
    # ax2.fill_between(xsponge, [ds.ycr[0,0]+ds.dyab/1000,ds.ycr[0,0]+ds.dyab/1000], [ds.ycr[0,0],ds.ycr[0,0]], facecolor=csponge, alpha=alpha_sponge)
    # ax2.fill_between(xsponge, [ds.ycr[-1,0],ds.ycr[-1,0]], [ds.ycr[-1,0]-ds.dyab/1000,ds.ycr[-1,0]-ds.dyab/1000], facecolor=csponge, alpha=alpha_sponge)

    """Colorbar (only first variable)"""
    fig.colorbar(
        contf_1,
        ax=ax_cbar,
        location='bottom',
        shrink=0.85,
        fraction=1,
        # pad=0.08,
        ticks=myclevl[var1],
        label=mylabel[var1],
        extend='both'
    )

    """Save figure"""
    os.makedirs(image_folder,exist_ok=True)
    if t<10:
        buffer = "00"
    elif t<100:
        buffer = "0"
    else:
        buffer = ""
    fig_title = "slice_" + buffer + str(t) + ".png"
    dpi = 200
    if test:
        dpi = 300
    fig.savefig(os.path.join(image_folder,fig_title), facecolor='w', edgecolor='w',
                    format='png', dpi=dpi, bbox_inches='tight')
    """Finish"""
    plt_helper.show_progress(pbar['progress_counter'], pbar['lock'], pbar["stime"], pbar['ntasks'])
    sema.release()


def plt_xzslc(ax, dsxz, ds, var='w'):
    jslc = int(dsxz.j)

    if var == "vorticity":
        cvar = plt_helper.vorticity_xz(dsxz['u'].values, dsxz['w'].values, ds.dx00, ds.dz00)

    elif var == "mf":
        if t==0:
            print(f"[i]  Filter with $\\lambda_x$ = {lambdax} and $\\lambda_z$={lambdaz}")
        uw   = dsxz['w'].values * (dsxz['u'].values-dsxz['ue'].values)
        mfx  = dsxz['rh0'].values * uw
        mfx = filter.gaussian_fft_smoothing(mfx, lambdaz, lambdax, ds.dz00, ds.dx00)
        izcut = int(zcut_mf / ds.dz00)
        mfx[izcut:,:] = mfx[izcut:,:] * 1000
        cvar = mfx*1000
            
    elif var == "ep":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxz['thprime'].values, dsxz['the'].values, dsxz['pprime'].values, dsxz['ppe'].values, ds.cap, ds.pref00)
        epm = 1/2*(ds.g/ds.bv)**2 * ((tloc-tenv)/tenv)**2
        epm = filter.gaussian_fft_smoothing(epm, lambdaz, lambdax, ds.dz00, ds.dx00)
        epm = np.where(epm>clev[0],epm,np.nan)
        cvar = epm
    elif var == "u":
        cvar = dsxz["u"].values - dsxz["ue"].values
    elif var == "t":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxz['thprime'].values, dsxz['the'].values, dsxz['pprime'].values, dsxz['ppe'].values, ds.cap, ds.pref00)
        cvar = tloc - tenv
    else:
        cvar = dsxz[var].values
        
    cmap = mycmap[var]
    norm = mynorm[var]
    clev = myclev[var]
    contf = ax.contourf(ds.xcr.expand_dims({'z':dsxz.z},axis=0)[:,jslc,:], dsxz.zcr, cvar,
                            cmap=cmap, norm=norm, levels=clev, extend='both')
    ax.contour(ds.xcr.expand_dims({'z':dsxz.z},axis=0)[:,jslc,:], dsxz.zcr, dsxz['the']+dsxz['thprime'], 
                            colors='k', alpha=0.7, levels=thlev, lw=lw2)
        
    ax.plot(ds.xcr[jslc], surf_factor*dsxz.zcr[0,:], lw=1.5, color='black')
    #ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(labelbottom=False,labeltop=False)
    ax.set_xlabel('streamwise x / km') # change to longitudes, latitude 10$^3$
    ax.xaxis.set_label_position('top') 
    ax.grid()

    ##ax.set_ylabel('altitude z-z$_{trp}$ / km')
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.tick_params(which='both', top=True, right=True, bottom=False, labelbottom=False, labeltop=True, labelleft=False, labelright=False)
    # ax.text(1-xpp, ypp, f"y: {dsxz.ypos}km", transform=ax.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    return ax, contf


def plt_yzslc(ax, dsyz, ds, var="w"):
    """Plot xy-slice"""

    if var == "mf":
        uw   = dsyz['w'].values * (dsyz['u'].values-dsyz['ue'].values)
        vw   = dsyz['w'].values * (dsyz['v'].values-dsyz['ve'].values)
        mfx  = dsyz['rh0'].values * uw
        mfy  = dsyz['rh0'].values * vw
        mfx = filter.gaussian_fft_smoothing(mfx,lambdax, lambdax, ds.dx00, ds.dx00)
        mfy = filter.gaussian_fft_smoothing(mfy,lambdax, lambdax, ds.dx00, ds.dx00)
        cvar = mfx*1000
    elif var == "ep":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsyz['thprime'].values, dsyz['the'].values, dsyz['pprime'].values, dsyz['ppe'].values, ds.cap, ds.pref00)
        epm = 1/2*(ds.g/ds.bv)**2 * ((tloc-tenv)/tenv)**2
        epm = filter.gaussian_fft_smoothing(epm,lambdax, lambdax, ds.dx00, ds.dx00)
        epm = np.where(epm>clev[0],epm,np.nan)
        cvar = epm
    elif var == "u":
        cvar = dsyz["u"] - dsyz["ue"]
    elif var == "t":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsyz['thprime'].values, dsyz['the'].values, dsyz['pprime'].values, dsyz['ppe'].values, ds.cap, ds.pref00)
        cvar = tloc - tenv
    else:
        cvar = dsyz[var].values
    islc = int(dsyz.i)
    cmap = mycmap[var]
    norm = mynorm[var]
    clev = myclev[var]
    contf = ax.contourf(ds.ycr.expand_dims({'z':dsyz.z},axis=0)[:,:,islc], dsyz.zcr, cvar, cmap=cmap, norm=norm, levels=clev, extend='both')
    # ax.contour(ds.ycr.expand_dims({'z':dsyz.z},axis=0)[:,:,islc], dsyz.zcr, dsyz['ue'].values, 
    #                 colors='black', norm=norm, levels=wind_levels, linewidths=0.5, extend='both')

    ### - Topography - ###
    ax.plot(ds.ycr[:,islc], surf_factor*dsyz.zcr[0,:], lw=1.5, color='black')

    # ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('spanwise y / km')

    ax.set_xlim(ylim)
    ax.set_ylim(zlim)
    ax.grid()
    ax.text(1-xpp, ypp, f"x: {dsyz.xpos}km", transform=ax.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})


    return ax, contf


def plt_amtm(ax, dsxy_list, ds, dslid):
    """Plot AMTM measurement (variable is t)"""

    weights = [0.0625,0.25,0.375,0.25,0.0625] # Pascal's triangle
    # weights = [1, 8, 28, 56, 70, 56, 28, 8, 1] / 256
    # dsxy = sum(w * ds for w, ds in zip(weights, dsxy_list))
    # print(dsxy)

    var = "t"

    tlocs = []
    tenvs = []
    for dsxy in dsxy_list:
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxy['thprime'].values, dsxy['the'].values, dsxy['pprime'].values, dsxy['ppe'].values, ds.cap, ds.pref00)
        tlocs.append(tloc)
        tenvs.append(tenv)
    tloc = sum(w * layer for w, layer in zip(weights, tlocs))
    tenv = sum(w * layer for w, layer in zip(weights, tenvs))
    cvar = tloc - tenv

    cmap = mycmap[var]
    norm = mynorm[var]
    clev = myclev[var]
    contf = ax.contourf(ds.xcr, ds.ycr, cvar, cmap=cmap, norm=norm, levels=clev, extend='both')
    dsxy = dsxy_list[0]

    ### - Topography - ###
    ax.contour(ds.xcr, ds.ycr, surf_factor*dsxy.zcrtopo, colors='k', levels=topo_levels, linewidths=0.3)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('streamwise x / km')

    amtm_domain = 200 
    ax.set_xlim(dslid.xpos-amtm_domain/2, dslid.xpos+amtm_domain/2)
    ax.set_ylim(dslid.ypos-amtm_domain/2, dslid.ypos+amtm_domain/2)
    ax.grid()

    # - Lidar and Slice positions - #
    ax.text(dslid.xpos, dslid.ypos, "d", weight='bold', fontsize=8, bbox={"boxstyle" : "circle", "lw":0.4, "facecolor":"white", "edgecolor":"black"})
    ax.text(1-xpp, ypp, f"AMTM (80-88km)", transform=ax.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    return ax, contf


def plt_xyslc(ax, dsxy, ds, var="w"):
    """Plot xy-slice"""

    if var == "vorticity":
        cvar = 5*plt_helper.vorticity_xz(dsxy['v'].values, dsxy['u'].values, ds.dy00, ds.dx00) # dv/dx - du/dy

    elif var == "mf":
        uw   = dsxy['w'].values * (dsxy['u'].values-dsxy['ue'].values)
        vw   = dsxy['w'].values * (dsxy['v'].values-dsxy['ve'].values)
        mfx  = dsxy['rh0'].values * uw
        mfy  = dsxy['rh0'].values * vw
        mfx = filter.gaussian_fft_smoothing(mfx,lambdax, lambdax, ds.dx00, ds.dx00)
        mfy = filter.gaussian_fft_smoothing(mfy,lambdax, lambdax, ds.dx00, ds.dx00)
        if dsxy.zpos > zcut_mf / 1000:
            mfx = mfx*1000
        cvar = mfx*1000
        # efx = (dsxy['u']-dsxy['ue']) * ds['pprime']
        # efy = (dsxy['v']-dsxy['ve']) * ds['pprime']
        # efx = subroutines.fft_gaussian_xy(efx,nx_avg)
        # efy = subroutines.fft_gaussian_xy(efy,nx_avg)
        # efmax = np.max(np.sqrt(efx**2+efy**2))
        # print(efmax)
        # efx = efx/efmax
        # efy = efy/efmax
        
        # efx = np.where(np.abs(efy)>0.165,efx,np.nan) # 0.07
        # efy = np.where(np.abs(efy)>0.165,efy,np.nan)
    elif var == "ep":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxy['thprime'].values, dsxy['the'].values, dsxy['pprime'].values, dsxy['ppe'].values, ds.cap, ds.pref00)
        epm = 1/2*(ds.g/ds.bv)**2 * ((tloc-tenv)/tenv)**2
        epm = filter.gaussian_fft_smoothing(epm,lambdax, lambdax, ds.dx00, ds.dx00)
        epm = np.where(epm>clev[0],epm,np.nan)
        cvar = epm
    elif var == "u":
        cvar = dsxy["u"] - dsxy["ue"]
    elif var == "t":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dsxy['thprime'].values, dsxy['the'].values, dsxy['pprime'].values, dsxy['ppe'].values, ds.cap, ds.pref00)
        cvar = tloc - tenv
    else:
        cvar = dsxy[var].values

    cmap = mycmap[var]
    norm = mynorm[var]
    clev = myclev[var]
    contf = ax.contourf(ds.xcr, ds.ycr, cvar, cmap=cmap, norm=norm, levels=clev, extend='both')
    
    ### - Topography - ###
    ax.contour(ds.xcr, ds.ycr, surf_factor*dsxy.zcrtopo, colors='k', levels=topo_levels, linewidths=0.3)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('streamwise x / km')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid()

    if region=="darwin":
        ax.plot(-8.8, -61.6, marker='^', mfc='none', mec='black', markersize=10)

    ax.text(1-xpp, ypp, f"z: {dsxy.zpos}km", transform=ax.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    return ax, contf


def plot_vlidar(axlid, dslid, ds, var="t", t=0):

    time_res = (dslid.time[-1] - dslid.time[-2]).values * 60 # min
    if var == "mf":
        uw   = dslid['w'].values * (dslid['u'].values-dslid['ue'].values)
        mfx  = dslid['rh0'].values * uw
        mfx = filter.gaussian_fft_smoothing(mfx, 2*60, lambdaz, time_res, ds.dz00)
        izcut = int(zcut_mf / ds.dz00)
        mfx[:,izcut:] = mfx[:,izcut:] * 1000
        cvar = mfx*1000
    elif var == "ep":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dslid['thprime'].values, dslid['the'].values, dslid['pprime'].values, dslid['ppe'].values, ds.cap, ds.pref00)
        epm = 1/2*(ds.g/ds.bv)**2 * ((tloc-tenv)/tenv)**2
        epm = filter.gaussian_fft_smoothing(epm, 1, lambdaz, 1, ds.dz00)
        epm = np.where(epm>clev[0],epm,np.nan)
        cvar = epm
    elif var == "u":
        cvar = dslid["u"] - dslid["ue"]
    elif var == "t":
        tloc, tenv = plt_helper.get_eulag_t_and_tenv(dslid['thprime'].values, dslid['the'].values, dslid['pprime'].values, dslid['ppe'].values, ds.cap, ds.pref00)
        vert_res = (dslid.zcr[-1,-1] - dslid.zcr[-1,-2]).values
        tprime_bwf15, tbg15 = filter.butterworth_filter(tloc, cutoff=1/VERTICAL_CUTOFF, fs=1/vert_res, order=5, mode='both')
        tprime_bwf_time, tbg_time = filter.butterworth_filter(tprime_bwf15.T, cutoff=1/(0.05*60), fs=1/time_res, order=5, mode='both')
        # tprime_bwf_time, tbg_time = filter.butterworth_filter(tprime_bwf15.T, cutoff=1/(0.25*60), fs=1/time_res, order=5, mode='both')
        
        # cvar = tprime_bwf15 # only vertical bwf
        # cvar = tbg_time.T # vertical and temporal bwf
        cvar = tloc - tenv # no bwf

        ### Second temperature axis ####
        axt = axlid.twinx()
        axt.plot(dslid.time, cvar[:,dslid.kprof], lw=1.5, color='red')
        axt.tick_params(labelbottom=False,labeltop=False, labelleft=True, labelright=False, left=True, right=False)

        axt.tick_params(axis='y', color='red', labelcolor='red', direction="in", pad=-7) # pad=-27
        axt.spines['left'].set_color('red')
        axt.spines['left'].set_linewidth(1.5)
        axt.set_ylim([-15,110])
        axt.yaxis.set_major_formatter(FuncFormatter(kelvin_formatter))
        axt.set_yticks([-10,0,10,20]) # labels=["-10 K", "0 K", "10 K", "20 K"]
        [label.set_horizontalalignment('left') for label in axt.get_yticklabels()]
        # axt.set_ylabel(r"T' / K", color='red')
    else:
        cvar = dslid[var].values
    
    cmap = mycmap[var]
    norm = mynorm[var]
    clev = myclev[var]
    contf = axlid.contourf(dslid.time.expand_dims({'z':dslid.z},axis=1), dslid.zcr, cvar, levels=clev,
                        cmap=cmap, norm=norm, extend='both')

    # isentropes = axlid.contour(dslid.time, dslid.zcr, dslid.the + dslid.th, levels=thlev, colors='k', lw=lw2)
    # ax.clabel(isentropes, thlev[1::], fontsize=8, fmt='%1.0f K', inline_spacing=1, inline=True, 
    #             manual=[(8,ds.zcr[10,0,x]), (8,ds.zcr[-15,0,x])]) # ha='left', thlev[1::3]

    axlid.plot(dslid.time, surf_factor*dslid.zcr[:,0], lw=2, color='black')
    
    # axlid.yaxis.set_major_locator(MultipleLocator(10))
    axlid.xaxis.set_minor_locator(AutoMinorLocator())
    axlid.yaxis.set_minor_locator(AutoMinorLocator())
    axlid.tick_params(which='both', labelbottom=False,labeltop=True, labelleft=False, labelright=True, left=False, right=True)
    axlid.xaxis.set_label_position('top')
    axlid.yaxis.set_label_position('right')
    axlid.set_xlabel('time / h')
    axlid.set_ylabel('altitude z / km')
    axlid.set_xlim(xlim_lid)
    axlid.set_ylim(zlim)
    axlid.grid()
    
    hrs, rem = divmod(dslid.time[t].values*3600, 3600)
    mins, secs = divmod(rem, 60)
    axlid.axvline(x=[dslid.time[t].values], color='black', lw=lw1, ls='--') # ymin=zlim[0],ymax=zlim[1]
    axt.text(1-xpp, ypp-0.1, f"Time: {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}s ({t:03d})", transform=axt.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})
    if 'xpos' in dslid.attrs:
        axt.text(1-xpp, ypp, f"x: {dslid.xpos:.1f}km, y: {dslid.ypos:.1f}km", transform=axt.transAxes, weight='bold', bbox={"boxstyle" : "round", "lw":0.67, "facecolor":"white", "edgecolor":"black"})

    return axlid, contf


def kelvin_formatter(x, pos):
    return f"{x:.0f}K"
    # return f"{x:.0f}\u2009K"


def emulate_airplane_measurement(ds, speed=220, xrange=None, dim='xcr'):
    """
    Emulate an airplane flying along one slice (x-axis) with dimensions (time, z, x).
    
    Parameters:
    - ds: xarray.Dataset with dimensions (time, z, x)
    - speed: speed of the airplane in m/s
    - range: Boundaries for airplane to turn around
    
    Returns:
    - xarray.DataArray with dimensions (time, z), representing the vertical
      profile at the airplane's position at each timestep.
    """

    ## Format dataset
    # ds = ds.set_coords('time')
    if dim == 'ycr':
        ds['y'] = ds['ycr'][:,0]
        ds = ds.set_coords('y')
        ds = ds.set_index({'y':'y'})
    else:
        ds['x'] = ds['xcr'][0,:]
        ds = ds.set_coords('x')
        ds = ds.set_index({'x':'x'})
    ds = ds.reset_coords('zcr', drop=False)

    ## Get x-locations of airplane for each timestep
    if xrange is None:
        xrange  = [ds[dim].min().values, ds[dim].max().values]
    xrange = [x * 1000 for x in xrange] # meter
    
    T = (xrange[1] - xrange[0]) / speed  # time for one leg
    tau = np.mod(ds.time.values * 3600, 2 * T)
    xlocs = np.where(tau < T, # tau < T -> forward flying
                    xrange[0] + speed * tau,
                    xrange[1] - speed * (tau - T))
    
    # Extract vertical profiles for each x-location
    if dim == 'ycr':
        vertical_profiles = [
            ds.isel(t=tstep).sel(y=xloc/1000, method='nearest')
            for tstep, xloc in enumerate(xlocs)
        ]
    else:
        vertical_profiles = [
            ds.isel(t=tstep).sel(x=xloc/1000, method='nearest')
            for tstep, xloc in enumerate(xlocs)
        ]
    result = xr.concat(vertical_profiles, dim='time')
    return result, xlocs


if __name__ == '__main__':
    """Generate animation of EULAG simulation based on NETCDF slice and lidar output"""

    """Example: 
        >> python3 slc_and_lid.py <var> <simulation> <notest>
        >> python3 slc_and_lid.py vortex <simulation> <notest>
        >> python3 slc_and_lid.py amtm <simulation> <notest>
        >> python3 slc_and_lid.py surf <simulation> <notest>
        >> python3 slc_and_lid.py alima_x ideal_topo_L20 <notest>
        >> python3 slc_and_lid.py alima_y ideal_topo_L20 <notest>
    """    
    global region, xlim_lid, test
    
    var = sys.argv[1]
    simulation = sys.argv[2]
    test = True
    if len(sys.argv) > 3:
        if sys.argv[3] == "notest":
            test = False
    region = simulation.split("_")[0]
    ncpus = mp.cpu_count()-2 # use maximum here but check number of tasks --> ntasks
    ncpus = 75
    fpath = os.path.join(data_folder, simulation)
    image_folder = os.path.join(animation_folder, simulation) + "_" + var
    if os.path.isdir(image_folder):
        for png_file in glob.glob(os.path.join(image_folder, "*.png")):
            try:
                os.remove(png_file)
            except OSError:
                pass

    """Slices"""
    if region=="debeto":
        slices = {"x": 1, "y": 1, "z": [2]} # 7
        xlid = 11
        # xlid = 15.4
        xlim_lid = [4, 6.3] #h
    elif region=="darwin":
        xlid = 96 # CORAL - fake
        xlid = 25 # darwin slice
        # xlid = 114 # CORAL
        # xlid = 40 # DARWIN
        xlim_lid = [2.5, 4] #h
        slices = {"x": 0, "y": 0, "z": [2]} # 7
    elif region=="pata":
        xlid = -270
        slices = {"x": 0, "y": 0, "z": [5]} # 7
        xlim_lid = [0, 12] #h
    else:
        xlid = 0
        slices = {"x": 0, "y": 0, "z": [5]} # 7
        xlim_lid = [3, 7.5] #h

    if var=='vortex':
        xlim_lid = [3.5, 6.25] #h
        xlim_lid = [3.25, 5.5] #h
        # xlim_lid = [0,5] # fitz roy
        # slices = {"x": 0, "y": 1, "z": [8]} # 8 is good, 12 highest
        slices = {"x": 0, "y": 0, "z": [8]} # 8 is good, 12 highest
        # slices = {"x": 0, "y": 1, "z": [1]} # fitz roy

    elif var=='surf':
        xlim_lid = [0, 4] #h
    if var=='amtm':
        xlid = 114 # CORAL
        # xlim_lid = [3.5, 7.5] #h
        xlim_lid = [5.5, 7.5] #h
        xlim_lid = [0, 7] #h
        slices = {"x": 1, "y": 1, "z": [0,1,3,5,7,9]} # AMTM
    elif var=='alima_x' or var == 'alima_y':
        xlid = None # CORAL
        # xlim_lid = [2.5, 4] #h
        xlim_lid = [2.5, 5] #h
        slices = {"x": 0, "y": 0, "z": [0,4]}

    stime = time.time()
    print(f"[i]  Test run (one timestamp): Opening NETCDF files for simulation: {fpath}")
    print(slices)

    dsxz, dsyz = plt_helper.preprocess_eulag_xzyz(fpath, slices=slices)
    elapsed = time.time() - stime
    print(f"[i]  Files loaded in {elapsed / 60:.2f}min.")

    """Lidar location in xz slice"""
    if xlid is not None:
        ilid = int((xlid - dsxz.xcr.min()) / (dsxz.xcr.max() - dsxz.xcr.min()) * np.shape(dsxz.xcr)[-1])
        xlid = dsxz.xcr[0,ilid].values
        dslid = dsxz.isel(x=ilid) # x location in km
        dslid.attrs['i'] = ilid 
        dslid.attrs['xpos'] = xlid
        print(f"[i]  Virtual ground-based lidar profile: i={dslid.i}, j={dslid.j}, x={dslid.xpos}km, y={dslid.ypos}km")
        
    else:
        global alima_locs
        if var == 'alima_y':
            dslid, alima_locs = emulate_airplane_measurement(dsyz, speed=220, xrange=[-200,200], dim='ycr')
        else:
            dslid, alima_locs = emulate_airplane_measurement(dsxz, speed=220, xrange=[-200,200], dim='xcr')
        print(f"[i]  Virtual airplane-based profile (ALIMA)!")

    # xlim_lid = [0,ds_xz.time.max().values]
    trange = [np.abs(dsxz.time.values - xlim_lid[0]).argmin(), np.abs(dsxz.time.values - xlim_lid[1]).argmin()]
    # trange = [700,900]

    trange = np.arange(trange[0], trange[1])
    if test:
        # trange = [647]
        # trange = [900]
        trange = [len(dsxz.time.values)-1]
        trange = [985]
        trange = [932]
        trange = [580]
    print(f"Plotting time steps {trange[0]} to {trange[-1]}")

    """Parallel processing"""
    progress_counter = mp.Manager().Value('i', 0)
    lock = mp.Manager().Lock()
    stime = time.time()
    pbar = {"progress_counter": progress_counter, "lock": lock, "stime": stime, "ntasks": len(trange)}
    ncpus = np.min([ncpus, pbar['ntasks']])
    sema = mp.Semaphore(ncpus)

    print(f"[i]  CPUs available: {mp.cpu_count()}")
    print(f"[i]  CPUs for visualization: {ncpus}")

    running_procs = []
    for t in trange:
        for p in running_procs[:]:
            if not p.is_alive():
                p.join()
                running_procs.remove(p)
        sema.acquire()
        args = (t, var, fpath, slices, dslid, image_folder, pbar, sema)
        proc = mp.Process(target=slc_and_lid, args=args)
        running_procs.append(proc)
        proc.start()
    
    for proc in running_procs:
        proc.join()

    plt_helper.create_animation(image_folder, "anime_abc_" + simulation + "_" + var + ".mp4")
