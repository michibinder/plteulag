import os
import sys
# import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter, AutoMinorLocator
# import statsmodels.api as sm 
# from statsmodels.graphics.gofplots import qqplot_2samples

# import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

from subroutines import *

import warnings
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)

# import matplotlib
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = ['Tahoma']

CLEV_100= [-100,-50,-30,-20,-10,-5,-3,-2,-1,-0.2,0.2,1,2,3,5,10,20,30,50,100]
CLEV_30 = [-30,-20,-10,-7,-5,-3,-2,-1,-0.7,-0.5,0.5,0.7,1,2,3,5,7,10,20,30]
CLEV_20 = [-20.,-7.,-5.,-3.,-2.,-1.,-0.7,-0.5,-0.2,-0.1,0.1,0.2,0.5,0.7,1.,2.,3.,5.,7.,20.]
CLEV_10 = [-10.,-7.,-5.,-3.,-2.,-1.,-0.7,-0.5,-0.2,-0.1,0.1,0.2,0.5,0.7,1.,2.,3.,5.,7.,10.]
CLEV_2 = [-5.,-2.,-1.,-0.7,-0.5,-0.3,-0.2,-0.1,-0.07,-0.05,0.05,0.07,0.1,0.2,0.3,0.5,0.7,1.,2.,5.]

FIGSIZE = (10,7)
FIGSIZE_2 = (10,8)
FIGSIZE_QQ_PLOT = (5,5)

def plot_uprime_xz(ds, SETTINGS, t, y=0):
    fig, ax0 = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap('RdBu_r')
    clev= CLEV_30
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    
    pcMesh = ax0.pcolormesh(ds.xcr[y], ds.zcr[:,y,:], ds.uprime[t,:,y,:],
                            cmap=cmap, norm=norm, shading='nearest')
    
    ax0 = format_xz_plot(ds, SETTINGS, ax0, t, y=0)
    
    # Labels
    cbar = fig.colorbar(pcMesh, ax=ax0, orientation='horizontal')
    cbar.set_label("u' / $m s^{-1}$")
    ax0.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax0.set_ylabel('altitude / km')
    
    fig.tight_layout()
    fig_name = 'uprime_y' + str(y) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'


def plot_wprime_xz(ds, SETTINGS, t, y=0):
    "w equals w'"
    fig, ax0 = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap('RdBu_r')
    clev= CLEV_20
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    
    pcMesh = ax0.pcolormesh(ds.xcr[y], ds.zcr[:,y,:], ds.w[t,:,y,:],
                            cmap=cmap, norm=norm, shading='nearest')
    
    ax0 = format_xz_plot(ds, SETTINGS, ax0, t, y=0)
    
    # Labels
    cbar = fig.colorbar(pcMesh, ax=ax0, orientation='horizontal')
    cbar.set_label("w' / $m s^{-1}$")
    ax0.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax0.set_ylabel('altitude / km')
    
    fig.tight_layout()
    fig_name = 'wprime_y' + str(y) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'
    

def plot_tprime_xz(ds, SETTINGS, t, y=0):
    fig, ax0 = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap('RdBu_r')
    clev=CLEV_20
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    
    pcMesh = ax0.pcolormesh(ds.xcr[y], ds.zcr[:,y,:], ds.tprime[t,:,y,:],
                            cmap=cmap, norm=norm, shading='nearest')
    
    ax0 = format_xz_plot(ds, SETTINGS, ax0, t, y=0)
    
    # Labels
    cbar = fig.colorbar(pcMesh, ax=ax0, orientation='horizontal')
    cbar.set_label("T' / $K$")
    ax0.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax0.set_ylabel('altitude / km')
    
    # Save fig
    fig.tight_layout()
    fig_name = 'tprime_y' + str(y) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'
    

def plot_thprime_xz(ds, SETTINGS, t, y=0):
    fig, ax0 = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap('RdBu_r')
    clev=CLEV_20
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    
    pcMesh = ax0.pcolormesh(ds.xcr[y], ds.zcr[:,y,:], ds.th[t,:,y,:],
                            cmap=cmap, norm=norm, shading='nearest')
    
    ax0 = format_xz_plot(ds, SETTINGS, ax0, t, y=0)
    
    # Labels
    cbar = fig.colorbar(pcMesh, ax=ax0, orientation='horizontal')
    cbar.set_label("$ \Theta $' / $K$")
    ax0.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax0.set_ylabel('altitude / km')
    
    # Save fig
    fig.tight_layout()
    fig_name = 'thprime_y' + str(y) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'
    

def plot_pprime_xz(ds, SETTINGS, t, y=0):
    fig, ax0 = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap('RdBu_r')
    clev=CLEV_30
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    
    pcMesh = ax0.pcolormesh(ds.xcr[y], ds.zcr[:,y,:], ds.h12[t,:,y,:],
                            cmap=cmap, norm=norm, shading='nearest')
    
    ax0 = format_xz_plot(ds, SETTINGS, ax0, t, y=0)
    
    # Labels
    cbar = fig.colorbar(pcMesh, ax=ax0, orientation='horizontal')
    cbar.set_label("p' / $K$")
    ax0.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax0.set_ylabel('altitude / km')
    
    # Save fig
    fig.tight_layout()
    fig_name = 'pprime_y' + str(y) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'
    

def plot_pprime2_xz(ds, SETTINGS, t, y=0):
    fig, ax0 = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap('RdBu_r')
    clev=CLEV_30
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    
    pcMesh = ax0.pcolormesh(ds.xcr[y], ds.zcr[:,y,:], ds.h7[t,:,y,:],
                            cmap=cmap, norm=norm, shading='nearest')
    
    ax0 = format_xz_plot(ds, SETTINGS, ax0, t, y=0)
    
    # Labels
    cbar = fig.colorbar(pcMesh, ax=ax0, orientation='horizontal')
    cbar.set_label("p' / $K$")
    ax0.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax0.set_ylabel('altitude / km')
    
    # Save fig
    fig.tight_layout()
    fig_name = 'pprime2_y' + str(y) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'


def plot_efx1_xz(ds, SETTINGS, t, y=0):
    fig, ax0 = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap('RdBu_r')
    clev = CLEV_10
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    
    pcMesh = ax0.pcolormesh(ds.xcr[y], ds.zcr[:,y,:], ds.efx1[t,:,y,:],
                            cmap=cmap, norm=norm, shading='nearest')
    
    ax0 = format_xz_plot(ds, SETTINGS, ax0, t, y=0)
    
    # Labels
    cbar = fig.colorbar(pcMesh, ax=ax0, orientation='horizontal')
    cbar.set_label('EF$_x$ / W $m^{-2}$')
    ax0.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax0.set_ylabel('altitude / km')
    
    # Save fig
    fig.tight_layout()
    fig_name = 'EFx1_y' + str(y) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'
    

def plot_efz1_xz(ds, SETTINGS, t, y=0):
    fig, ax0 = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap('RdBu_r')
    clev = CLEV_10
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    
    pcMesh = ax0.pcolormesh(ds.xcr[y], ds.zcr[:,y,:], ds.efz1[t,:,y,:],
                            cmap=cmap, norm=norm, shading='nearest')
    
    ax0 = format_xz_plot(ds, SETTINGS, ax0, t, y=0)
    
    # Labels
    cbar = fig.colorbar(pcMesh, ax=ax0, orientation='horizontal')
    cbar.set_label('EF$_z$ / W $m^{-2}$')
    ax0.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax0.set_ylabel('altitude / km')
    
    # Save fig
    fig.tight_layout()
    fig_name = 'EFz1_y' + str(y) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'


def plot_mfx_xz(ds, SETTINGS, t, y=0):
    fig, ax0 = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap('RdBu_r')
    clev = CLEV_100
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    
    pcMesh = ax0.pcolormesh(ds.xcr[y], ds.zcr[:,y,:], 1000*ds.mfx[t,:,y,:],
                            cmap=cmap, norm=norm, shading='nearest')
    
    ax0 = format_xz_plot(ds, SETTINGS, ax0, t, y=0)
    
    # Labels
    cbar = fig.colorbar(pcMesh, ax=ax0, orientation='horizontal')
    cbar.set_label('MF$_x$ / $mPa$')
    ax0.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax0.set_ylabel('altitude / km')
    
    # Save fig
    fig.tight_layout()
    fig_name = 'MFx_y' + str(y) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'