import os
import sys

import numpy as np
import xarray as xr
import scipy
# import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter, AutoMinorLocator
# from matplotlib.colors import BoundaryNorm
# import matplotlib.dates as mdates
# from mpl_toolkits.axes_grid1 import make_axes_locatable

import xrft.xrft as xrft

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)


def average_horizontal(data,nx_avg):
    "average over about one horizontal wavelength"
    nx=np.shape(data)[1]
    padded = data.pad(x=nx_avg, mode="edge")
    data_avg = padded.rolling(x=nx_avg, center=True).construct("new").mean("new",skipna=True)[:,nx_avg:nx+nx_avg]
    return data_avg
    
def average_vertical(data, nz_avg):
    "average over about one vertical wavelength"
    nz=np.shape(data)[0]
    padded = data.pad(z=nz_avg, mode="edge") # padded with NAN for interpolated grid
    data_avg = padded.rolling(z=nz_avg, center=True).construct("new").mean("new",skipna=True)[nz_avg:nz+nz_avg,:]
    # Rolling contruct: construct(window_dim=None, stride=1, fill_value=<NA>)
    return data_avg


def fft_gaussian_filter(data,nz_avg,nx_avg=None,usey=0,pad_mode="symmetric"):
    "Gaussian filter for individual horizontal and vertical wavelengths - required for different horiz/vert resolutions"
    nz=np.shape(data)[0]
    # nx=np.shape(data)[1]
    
    # mask = np.isfinite(data)
    # data = data.fillna(0)

    ## REPLACE NANs AND PAD ##
    data = data.ffill(dim='z').bfill(dim='z')
    # data = data.pad(z=nz_avg, mode="edge")
    data = data.pad(z=nz_avg, mode=pad_mode) # "symmetric", "reflect" - difference??
    
    # data = data.pad(x=nx_avg, mode="edge")

    ## FFT ##
    data=data.drop(['zcr','xcr','ycr','zs','zh','gi','z_coord'], errors='ignore')
    da_fft = xrft.fft(data) # Fourier Transform w/ numpy.fft-like behavior   
    # da_fft = xrft.dft(data, true_phase=True, true_amplitude=True)
    
    ## GAUSSIAN RESPONSE FUNCTION ##
    response_func_z = np.exp(-da_fft.freq_z**2 * nz_avg**2)
    da_fft_low = da_fft * response_func_z

    if usey:
        response_func_y = np.exp(-da_fft.freq_y**2 * nx_avg**2)
        da_fft_low = da_fft_low * response_func_y
    else:
        if (nx_avg != None): # include horizontal averaging
            nx_avg = 2*nx_avg
            response_func_x = np.exp(-da_fft.freq_x**2 * nx_avg**2)
            da_fft_low = da_fft_low * response_func_x
    
    ## INVERSE FFT ##
    if len(np.shape(data))==1:
        data_filtered = xrft.ifft(da_fft_low)[nz_avg:nz+nz_avg]
    else:
        data_filtered = xrft.ifft(da_fft_low)[nz_avg:nz+nz_avg,:]
    # data_filtered = xrft.ifft(da_fft_low)[nz_avg:nz+nz_avg,nx_avg:nx+nx_avg]
    # data_filtered = xrft.idft(da_fft_low, true_phase=True, true_amplitude=True)
    return data_filtered.real


def fft_gaussian_xy(data,nx_avg,ny_avg=None):
    "Gaussian filter for horizontal 2D plane - eventually split filter for x and y direction"
    nz=np.shape(data)[0]
    nx=np.shape(data)[1]
    # mask = np.isfinite(data)
    # data = data.fillna(0)

    ## REPLACE NANs AND PAD ##
    # data = data.ffill(dim='x').bfill(dim='x')
    # data = data.pad(x=nx_avg, mode="edge")
    # data = data.pad(x=nx_avg, mode="edge")

    ## FFT ##
    data=data.drop(['zcr','xcr','ycr','zs','zh','gi'])
    da_fft = xrft.fft(data) # Fourier Transform w/ numpy.fft-like behavior   
    
    ## GAUSSIAN RESPONSE FUNCTION ##
    response_func_x = np.exp(-da_fft.freq_x**2 * nx_avg**2)
    da_fft_low = da_fft * response_func_x

    if (ny_avg != None):
        n_avg = ny_avg
    else:
        n_avg = nx_avg

    # n_avg = 2*n_avg
    response_func_y = np.exp(-da_fft.freq_y**2 * n_avg**2)
    da_fft_low = da_fft_low * response_func_y
    
    ## INVERSE FFT ##
    data_filtered = xrft.ifft(da_fft_low)

    return data_filtered.real


def filter_2D(data,nx_avg,nz_avg,mode=0):
    if int(mode)==1:
        "Gaussian filter with Fast Fourier Transform (FFT)"
        data_filtered = fft_gaussian_filter(data,nz_avg,nx_avg=nx_avg)
    else:
        "Running mean"
        data_filtered = average_vertical(data, nz_avg)
        data_filtered = average_horizontal(data_filtered, nx_avg)
    return data_filtered


def filter_1Dz(data,nz_avg,mode=0,pad_mode="symmetric"):
    "only use vertical filter before applying horizontal average"
    if int(mode)==1:
        "Gaussian filter with Fast Fourier Transform (FFT)"
        data_filtered = fft_gaussian_filter(data,nz_avg,pad_mode=pad_mode)
    else:
        "Running mean"
        data_filtered = average_vertical(data, nz_avg)
    return data_filtered
    

def interp_elev_to_z(data,elev,z):
    "2D Berg fuer xz Schnitt y level egal, fuer 3D Berg wichtig!"
    # old_shape = np.shape(data)
    # new_shape = (len(z),old_shape[1])
    # data_i=np.zeros(new_shape)
    data_i=data # aequidistant vertical grid has same number of grid points as terrain following grid
    UNDEF=np.nan
    # UNDEF=-99. 
    for ix in range(0,np.shape(data)[1]):
        data_i[:,ix] = np.interp(z[:],elev[:,ix],data[:,ix],left=UNDEF)
    # data_i=np.ma.masked_array(data_i,np.isnan(data_i))
    
    return data_i


def filt_2dx(f,ifil):
    """ 
    Smooth pressure perturbations locally
    ifil: how often apply filter
    """
    b=f.copy()
    a=f.copy()
    i = np.arange(1,len(f.x)-1)
    # k = np.arange(0,len(ds.z)-1)
    
    for ifi in range(0,ifil):
        a[:,:,:,i]=0.25*(b[:,:,:,i-1]+2*b[:,:,:,i]+b[:,:,:,i+1])
        b=a.copy()
    return b


def dzdx_topo(ds):
    "calculates derivative of topo with centered difference"
    i = np.arange(1,len(ds.zs[0,:])-1)
    
    ds['dzdx_surf'] = ds.zs.copy()
    for y in range(0,len(ds.zs[:,0])):
        # Iterate over all y
        # ds['dzdx_surf'][y,i] = 1000*(ds.zs[y,i+1] - ds.zs[y,i-1]) / (2*ds.dx00)
        # ds['dzdx_surf'][y,i] = 1000*(ds.zs[y,i+1] - ds.zs[y,i-1]) / (ds.xcr[y][i+1]-ds.xcr[y][i-1])
        
        # Use topo of last time step
        ds['dzdx_surf'][y,i] = (ds.zcr[-1,0,y,i+1] - ds.zcr[-1,0,y,i-1]) / (ds.xcr[y][i+1]-ds.xcr[y][i-1])
    return ds


def dthdz_prof(ds,t=0,y=0,x=0):
    "calculates derivative of theta in vertical with centered difference"
    i = np.arange(1,len(ds['th'][0])-1)
    # print(i)
    dthdz = ds['th'][0,:,0,0].copy()
    ## TEMPERATURE CALCULATION##                             
    thloc = ds['the'][t,:,y,x] + ds['th'][t,:,y,x] # Theta
    # print(dthdz)
    dthdz[i] = (thloc[i+1] - thloc[i-1]) / (1000*(ds.zcr[t,i+1,y,x]-ds.zcr[t,i-1,y,x]))
    # dthdz[i] = (ds['thloc'][t,i+1,y,x] - ds['thloc'][t,i-1,y,x]) / (1000*(ds.zcr[t,i+1,y,x]-ds.zcr[t,i-1,y,x]))
    dthdz[0] = dthdz[1]
    dthdz[-1] = dthdz[-2]
    return dthdz


def dudz_prof(ds,t=0,y=0,x=0):
    "calculates derivative of u in vertical with centered difference"
    i = np.arange(1,len(ds['u'][0])-1)
    dudz = ds['u'][0,:,0,0].copy()
    # dudz[i] = ((ds['u'][t,i+1,y,x]+ds['ue'][t,i+1,y,x]) - (ds['u'][t,i-1,y,x]+ds['ue'][t,i-1,y,x])) / (1000*(ds.zcr[t,i+1,y,x]-ds.zcr[t,i-1,y,x]))
    dudz[i] = ((ds['u'][t,i+1,y,x]) - (ds['u'][t,i-1,y,x])) / (1000*(ds.zcr[t,i+1,y,x]-ds.zcr[t,i-1,y,x]))
    dudz[0] = dudz[1]
    dudz[-1] = dudz[-2]
    
    # dudz^2
    dudz2 = dudz.copy()
    dudz2[i] = (dudz[i+1] - dudz[i-1]) / (1000*(ds.zcr[t,i+1,y,x]-ds.zcr[t,i-1,y,x]))
    dudz2[0] = dudz2[1]
    dudz2[-1] = dudz2[-2]
    return dudz, dudz2


def dvdz_prof(ds,t=0,y=0,x=0):
    "calculates derivative of v in vertical with centered difference"
    i = np.arange(1,len(ds['u'][0])-1)
    # print(i)
    dudz = ds['v'][0,:,0,0].copy()
    # print(dthdz)
    dudz[i] = ((ds['v'][t,i+1,y,x]+ds['ve'][t,i+1,y,x]) - (ds['v'][t,i-1,y,x]+ds['ve'][t,i-1,y,x])) / (1000*(ds.zcr[t,i+1,y,x]-ds.zcr[t,i-1,y,x]))
    # dudz[i] = ((ds['ue'][t,i+1,y,x]) - (ds['ue'][t,i-1,y,x])) / (1000*(ds.zcr[t,i+1,y,x]-ds.zcr[t,i-1,y,x]))
    dudz[0] = dudz[1]
    dudz[-1] = dudz[-2]
    return dudz


def format_xz_plot(ds, SETTINGS, ax, t, y=0):
    if int(SETTINGS['STRATOS']):
        # thlev=np.exp(5+0.03*np.arange(1,120,5)) # every fifth theta in stratosphere with higher stability N=0.02 1/s
        thlev=np.exp(5+0.03*np.arange(1,120,10))
        surf_factor = 5 # 10 or 5
    else: # TROPOSPHERE
        thlev=np.exp(5+0.03*np.arange(1,100)) # 90 is needed for atmosphere up to 100km
        # thlev=np.exp(5+0.03*np.arange(1,100,0.5)) # more isentropes for troposphere visualization
        # thlev=260 + 1*(40*0.5) + 0.5*np.arange(0,40) # only lower atmosphere -> linear
        surf_factor = 10 # 10 or 5

    y = int(ds.ny/2)
    x = 10
    z = 10
    # isentropes = ax.contourf(ds.xcr[y], ds.zcr[t,:,y,:], ds.thloc[t,:,y,:], colors='k', levels=thlev)
    # print(ds.xcr.expand_dims({'z':ds.z}))
    # print(ds.xcr.expand_dims({'z':ds.z})[:,y,:])
    isentropes = ax.contour(ds.xcr.expand_dims({'z':ds.z},axis=0)[:,y,:], ds.zcr[t,:,y,:], ds['the'][t,:,y,:]+ds['th'][t,:,y,:], colors='k', alpha=0.7, levels=thlev)
    if int(SETTINGS['STRATOS']):
        th_z_pos = np.linspace(5,36,5)
        th_label = []
        for pos in th_z_pos:
            th_label.append((4000,pos))
        ax.clabel(isentropes, fmt='%1.0f K', inline_spacing=1, inline=True, fontsize=8, manual=th_label)
    else:
        th_z_pos = np.linspace(2,20,4)
        th_label = []
        for pos in th_z_pos:
            th_label.append((-600,pos))
        ax.clabel(isentropes, fmt=' %1.0f K ', inline_spacing=2, inline=True, fontsize=8, manual=th_label)
        
    # - TOPO - #
    # ax.plot(ds.xcr[y], surf_factor*ds.zs[y,:], lw=2, color='red')
    ax.plot(ds.xcr[y], surf_factor*ds.zcr[t,0,y,:], lw=2, color='black')
    # ax.plot(ds.xcr[y], 10000*ds.dzdx_surf[y,:], lw=2, color='red') # Verification of surf derivative
    # ax.plot(ds.xcr[y],10*ds.ELEVATION[t,0,y,:]/1000, lw=2, color='black')
    
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    return ax


def format_yz_plot(ds, SETTINGS, ax, t, x=0):
    if int(SETTINGS['STRATOS']):
        # thlev=np.exp(5+0.03*np.arange(1,120,5)) # every fifth theta in stratosphere with higher stability N=0.02 1/s
        thlev=np.exp(5+0.03*np.arange(1,120,10))
        surf_factor = 5 # 10 or 5
    else: # TROPOSPHERE
        thlev=np.exp(5+0.03*np.arange(1,100)) # 90 is needed for atmosphere up to 100km
        # thlev=np.exp(5+0.03*np.arange(1,100,0.5)) # more isentropes for troposphere visualization
        # thlev=260 + 1*(40*0.5) + 0.5*np.arange(0,40) # only lower atmosphere -> linear
        surf_factor = 10 # 10 or 5

    # t=0
    # y=int(ds.ny/2)
    # x=0

    # isentropes = ax.contourf(ds.xcr[y], ds.zcr[t,:,y,:], ds.thloc[t,:,y,:], colors='k', levels=thlev)
    # print(ds.xcr.expand_dims({'z':ds.z}))
    # print(ds.xcr.expand_dims({'z':ds.z})[:,y,:])
    isentropes = ax.contour(ds.ycr.expand_dims({'z':ds.z},axis=0)[:,:,x], ds.zcr[t,:,:,x], ds['the'][t,:,:,x]+ds['th'][t,:,:,x],
                            colors='k', alpha=0.7, levels=thlev)
    th_z_pos = np.linspace(5,36,5)
    th_label = []
    for pos in th_z_pos:
        th_label.append((2000,pos))
    ax.clabel(isentropes, fmt='%1.0f K', inline_spacing=0, inline=True, fontsize=8, manual=th_label)

    # - TOPO - #
    surf_factor = 3 # 10 or 5
    # ax.plot(ds.xcr[y], surf_factor*ds.zs[y,:], lw=2, color='red')
    ax.plot(ds.ycr[:,x], surf_factor*ds.zcr[t,0,:,x], lw=2, color='black')
    # ax.plot(ds.xcr[y], 10000*ds.dzdx_surf[y,:], lw=2, color='red') # Verification of surf derivative
    # ax.plot(ds.xcr[y],10*ds.ELEVATION[t,0,y,:]/1000, lw=2, color='black')
    
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    return ax


def format_xy_plot(ds, SETTINGS, ax, t, z=0):
    if ds.amp < 0:
        topo_levels=np.linspace(ds.amp,0,6) # for depression
        isentropes = ax.contour(ds.xcr, ds.ycr, 1000*ds.zcr[t,0,:,:], colors='k', levels=topo_levels)
    else: 
        topo_levels=np.linspace(-10*ds.amp,0,5) # for mountain
        isentropes = ax.contour(ds.xcr, ds.ycr, -10*1000*ds.zcr[t,0,:,:], colors='k', levels=topo_levels)
    
    # ax.clabel(isentropes, thlev[1::], fontsize=8, fmt='%1.0f K', inline_spacing=1, inline=True, 
    #             manual=[(ds.xcr[0,90],ds.zcr[t,10,0,90]), (ds.xcr[0,90],ds.zcr[t,-15,0,90])]) # ha='left', thlev[1::3]
    # try:
    #     ax.clabel(isentropes, topo_levels[1::], fontsize=6, fmt='%1.0f', inline_spacing=1, inline=True)
    # except:
    #     print('XY-plot: no levels available! (probably due to rising topography)')

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    return ax


def format_tz_plot(ds_lid, ds, SETTINGS, ax):
    "!!!adjust ds.zcr due to what zcr could be used for Lidar data!!!!!!"
    t=0
    y=int(ds.ny/2)
    x=0
    if int(SETTINGS['STRATOS']):
        # thlev=np.exp(5+0.03*np.arange(1,120,5)) # every fifth theta in stratosphere with higher stability N=0.02 1/s
        thlev=np.exp(5+0.03*np.arange(1,120,10))
    else: # TROPOSPHERE
        thlev=np.exp(5+0.03*np.arange(1,100)) # 90 is needed for atmosphere up to 100km

    isentropes = ax.contour(ds_lid.time, ds_lid.zcr, ds_lid.the + ds_lid.th, colors='k', levels=thlev)
    # ax.clabel(isentropes, thlev[1::], fontsize=8, fmt='%1.0f K', inline_spacing=1, inline=True, 
    #             manual=[(8,ds.zcr[t,10,0,x]), (8,ds.zcr[t,-15,0,x])]) # ha='left', thlev[1::3]

    # ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    surf_factor = 5 # 10 or 5
    # ax.plot(ds.xcr[y], surf_factor*ds.zs[y,:], lw=2, color='red')
    ax.plot(ds_lid.time, surf_factor*ds_lid.zcr[:,0], lw=2, color='black')

    return ax