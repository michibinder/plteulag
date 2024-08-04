################################################################################
# Copyright 2023 German Aerospace Center                                       #
################################################################################
# This is free software you can redistribute/modify under the terms of the     #
# GNU Lesser General Public License 3 or later: http://www.gnu.org/licenses    #
################################################################################

import os
import glob
import matplotlib.dates as mdates
import time
from datetime import datetime
import imageio.v2 as imageio

import numpy as np
import xarray as xr
import scipy
import pywt

from waveletFunctions import wave_signif, wavelet
# from statistics import linear_regression

# import logging
# import warnings
# warnings.simplefilter("ignore", RuntimeWarning)
# warnings.filterwarnings('ignore', category=UserWarning, module='imageio_ffmpeg')
# logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

"""Config"""
pbar_interval = 5 # %

mother = 'MORLET'
lag1 = 0.72  # lag-1 autocorrelation for red noise background
sig_lev = 1

def wavelet_1D_decomp(data, dx, dim=0):
    """1D Wavelet analysis along given dim of 2D array"""

    shape = np.shape(data)
    lambdax = np.zeros(shape)
    variance = np.std(data, ddof=1) ** 2

    # pad = 1    # pad the time series with zeroes (recommended)
    dj = 0.0625  # this will do 4 sub-octaves per octave
    s0 = 2 * dx  # this says start at a scale of 6 months
    j1 = 5 / dj  # this says do 7 powers-of-two with dj sub-octaves each

    n = shape[0]
    m = shape[1]
    if dim == 0:
        n0 = n
        n1 = m
    else:
        n0 = m 
        n1 = n
    for y in range(0,n1):
        if dim == 0:
            narray = data[y,:]
        else:
            narray = data[:,y]
        wave, period, scale, coi = wavelet(narray, dx, dj=dj, s0=s0, J1=j1, mother=mother)
        power = (np.abs(wave)) ** 2  # / variance # wavelet power spectrum normalized by variance

        # - Significance levels - #
        signif = wave_signif(([variance]), dt=dx, sigtest=0, scale=scale, lag1=lag1, mother=mother)
        # expand signif --> (J+1)x(N) array
        sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
        sig95 = power / sig95  # where ratio > 1, power is significant

        # jvec holds indices of maximum power for each x-location
        # indices refer to certain wavelengths
        jvec = np.argmax(power,axis=0)
        xvec = np.arange(0,len(jvec),1)

        if dim == 0:
            lambdax[y,:] = np.where(sig95[jvec,xvec] > sig_lev, period[jvec], np.nan)
        else:
            lambdax[:,y] = np.where(sig95[jvec,xvec] > sig_lev, period[jvec], np.nan)
        # lambdax[y,:] = np.where(power[jvec,xvec] > E_THRESHOLD, period[jvec], np.nan) # E_THRESHOLD = 25
    return lambdax


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


def gaussian_filter_fft(arr, wavelength_x, wavelength_y, res_x, res_y):
    """
    Apply a Gaussian filter to a 2D array using FFT with different resolutions and cutoff wavelengths for each dimension.

    Parameters:
    arr (numpy.ndarray): 2D numpy array to be smoothed.
    wavelength_x (float): Cutoff wavelength in the x dimension.
    wavelength_y (float): Cutoff wavelength in the y dimension.
    res_x (float): Resolution in the x dimension.
    res_y (float): Resolution in the y dimension.

    Returns:
    numpy.ndarray: Smoothed 2D array.
    """
    # Get the dimensions of the input array
    nx, ny = arr.shape

    # Create the grid in the frequency domain
    kx = np.fft.fftfreq(nx, d=res_x)[:, None]
    ky = np.fft.fftfreq(ny, d=res_y)[None, :]
    
    # Calculate the Gaussian filter in the frequency domain using wavelengths
    sigma_x = wavelength_x / (2.0 * np.pi)
    sigma_y = wavelength_y / (2.0 * np.pi)
    gaussian = np.exp(-0.5 * ((kx**2) * (sigma_x**2) + (ky**2) * (sigma_y**2)))
    
    # Perform FFT of the input array
    arr_fft = np.fft.fft2(arr)
    
    # Apply the Gaussian filter in the frequency domain
    arr_fft_filtered = arr_fft * gaussian
    
    # Perform inverse FFT to get the smoothed array
    arr_smoothed = np.fft.ifft2(arr_fft_filtered).real
    
    return arr_smoothed


def preprocess_eulag_output(fpath):
    """Process EULAG output"""
    env_path   = os.path.join(fpath, "env.nc")
    tapes_path = os.path.join(fpath, "tapes.nc")
    grid_path  = os.path.join(fpath, "grd.nc") # --> ds
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


def timelab_format_func(value, tick_number):
    dt = mdates.num2date(value)
    if dt.hour == 0:
        return "{}\n{}".format(dt.strftime("%Y-%b-%d"), dt.strftime("%H"))
    else:
        return dt.strftime("%H")


def major_formatter_lon(x, pos):
    """Using western coordinates"""
    return "%.f°W" % abs(x)
    ##return "%.f°E" % abs(x)


def major_formatter_lat(x, pos):
    return "%.f°S" % abs(x)


def create_animation(image_folder, animation_name):
    """Create animation (mp4) from pngs"""

    filenames        = sorted(os.listdir(image_folder))
    fps              = 10
    macro_block_size = 16

    # Increase the probesize to give FFmpeg more data to estimate the rate
    # writer_options = {'ffmpeg_params': ['-probesize', '100M']}  # Increase probesize to 5MB
    # writer_options = {'ffmpeg_params': ['-probesize', '5000000', '-analyzeduration', '5000000']}

    with imageio.get_writer(os.path.join(image_folder, animation_name), fps=fps) as writer: # duration=1000*1/fps
        for filename in filenames:
            if filename.endswith(".png"):
                image = imageio.imread(os.path.join(image_folder, filename))
                image = resize_to_macro_block(image, macro_block_size)
                writer.append_data(image)

    # imageio.mimsave(image_folder + "/era5_sequence.gif", images, duration=1/fps, palettesize=256/2)  # loop=0, quantizer="nq", palettesize=256
    print("[i]  MP4 Video created successfully!")


def resize_to_macro_block(image, macro_block_size):
    """Function to make image dimensions divisible by macro block size"""
    height, width = image.shape[:2]
    new_height = (height + macro_block_size - 1) // macro_block_size * macro_block_size
    new_width = (width + macro_block_size - 1) // macro_block_size * macro_block_size
    if (new_height != height) or (new_width != width):
        image = np.pad(image, ((0, new_height - height), (0, new_width - width), (0, 0)), 'constant')
    return image


def show_progress(progress_counter, lock, stime, total_tasks):
    with lock:
        progress_counter.value += 1
        if total_tasks <= 100/pbar_interval:
            print(f"[p]  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Number of tasks below progress bar limit.")
        else:
            if (progress_counter.value % (total_tasks // (100/pbar_interval))) == 0 or progress_counter.value == total_tasks or progress_counter.value == 1:
                progress = progress_counter.value / total_tasks
                elapsed = time.time() - stime
                eta = (elapsed / progress) * (1 - progress)

                # Convert elapsed and ETA to hours, minutes, and seconds
                elapsed_hrs, elapsed_rem = divmod(elapsed, 3600)
                elapsed_min, elapsed_sec = divmod(elapsed_rem, 60)
                eta_hrs, eta_rem = divmod(eta, 3600)
                eta_min, eta_sec = divmod(eta_rem, 60)

                # Progress bar
                total_hashtags = int(100/pbar_interval)
                hashtag_str = "#" * int(np.ceil(progress * total_hashtags))
                minus_str = "-" * int((1 - progress) * total_hashtags)

                print(f"[p]  |{hashtag_str}{minus_str}| Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Progress: {progress*100:05.2f}% - Elapsed: {int(elapsed_hrs):02d}:{int(elapsed_min):02d}:{int(elapsed_sec):02d} - ETA: {int(eta_hrs):02d}:{int(eta_min):02d}:{int(eta_sec):02d} (hh:mm:ss)", flush=True)


def get_colormap_bins_and_labels(max_level=64,linear=False):
    """
    Get different bin and label settings for varying colormap ranges.
    The bins are used for norm and later within pcolormesh / contourf / ...

    Good matplotlib colormaps:
    - 'turbo' better than 'jet'?
    - Plasma
    - Spectral
    - RdYlBu_r
    - RdBu_r -> perturbations of simulations
    
    cmap = plt.get_cmap('RdBu_r')

    """
    # - Logarithmic - #
    CLEV_64 = [-64,-32,-16,-8,-4,-2,-1,1,2,4,8,16,32,64]
    CLEV_64_LABELS = [-32,-8,-2,2,8,32]

    CLEV_32 = [-32,-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16,32]
    CLEV_32_LABELS = [-16,-4,-1,1,4,16]

    CLEV_16 = [-16,-8,-4,-2,-1,-0.5,-0.25,-0.125,0.125,0.25,0.5,1,2,4,8,16]
    CLEV_16_LABELS = [-8,-2,-0.5,-0.125,0.125,0.5,2,8]

    CLEV_8 = [-8,-4,-2,-1,-0.5,-0.25,-0.125,-0.0625,0.0625,0.125,0.25,0.5,1,2,4,8]
    CLEV_8_LABELS = [-4,-1,-0.25,-0.0625,0.0625,0.25,1,4]

    CLEV_2 = [-2,-1,-0.5,-0.25,-0.125,-0.0625,-0.03125,-0.015625,0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2]
    CLEV_2_LABELS = [-1,-0.25,-0.0625,-0.015625,0.015625,0.0625,0.25,1]

    CLEV_08 = np.array(CLEV_8) / 10
    CLEV_08_LABELS = np.array(CLEV_8_LABELS) / 10

    CLEV_02 = np.array(CLEV_2) / 10
    CLEV_02_LABELS = np.array(CLEV_2_LABELS) / 10

    # CLEV_100 = np.array([-100,-50,-20,-10,-7,-5,-3,-1,1,3,5,7,10,20,50,100])
    # CLEV_100_LABELS = np.array([-50,-10,-5,-1,1,5,10,50])

    # CLEV_30 = [-30,-10,-7,-5,-3,-2,-1,-0.5,0.5,1,2,3,5,7,10,30]
    # CLEV_30_LABELS = [-10,-5,-2,-0.5,0.5,2,5,10]

    # CLEV_20 = [-20.,-7.,-5.,-3.,-2.,-1.,-0.7,-0.5,-0.2,-0.1,0.1,0.2,0.5,0.7,1.,2.,3.,5.,7.,20.]

    # CLEV_10 = np.array([-10.,-5.,-2.,-1.,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,1.,2.,5.,10.])
    # CLEV_10_LABELS = np.array([-5.,-1.,-0.5,-0.1,0.1,0.5,1.,5.])

    # CLEV_5 = [-5.,-2.,-1.,-0.7,-0.5,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.5,0.7,1.,2.,5.]
    # CLEV_5_LABELS = [-2.,-0.7,-0.3,-0.1,0.1,0.3,0.7,2.]

    # CLEV_2 = [-2.,-1.,-0.5,-0.3,-0.2,-0.1,-0.07,-0.05,0.05,0.07,0.1,0.2,0.3,0.5,1.,2.]
    # CLEV_2_LABELS = [-1.,-0.3,-0.1,-0.05,0.05,0.1,0.3,1.]

    # CLEV_05 = [-0.5,-0.3,-0.1,-0.05,-0.03,-0.01,-0.005,-0.002,0.002,0.005,0.01,0.03,0.05,0.1,0.3,0.5]
    # CLEV_05_LABELS = [-0.3,-0.05,-0.01,-0.002,0.002,0.01,0.05,0.3]

    # CLEV_01 = CLEV_100/1000
    # CLEV_01_LABELS = CLEV_100_LABELS/1000

    # - Linear - #
    CLEV_22 = [-22,-18,-14,-10,-6,-2,2,6,10,14,18,22]
    CLEV_11 = [-11,-9,-7,-5,-3,-1,1,3,5,7,9,11]
    CLEV_11_LABELS = [-9,-5,-1,1,5,9]

    if linear:
        if max_level==64:
            CLEV = CLEV_64
            CLEV_LABELS = CLEV_64_LABELS
    else:    
        if max_level==64:
            CLEV = CLEV_64
            CLEV_LABELS = CLEV_64_LABELS
        elif max_level==32:
            CLEV = CLEV_32
            CLEV_LABELS = CLEV_32_LABELS
        elif max_level==16:
            CLEV = CLEV_16
            CLEV_LABELS = CLEV_16_LABELS
        elif max_level==8:
            CLEV = CLEV_8
            CLEV_LABELS = CLEV_8_LABELS
        elif max_level==2:
            CLEV = CLEV_2
            CLEV_LABELS = CLEV_2_LABELS
        elif max_level==0.8:
            CLEV = CLEV_08
            CLEV_LABELS = CLEV_08_LABELS
        elif max_level==0.2:
            CLEV = CLEV_02
            CLEV_LABELS = CLEV_02_LABELS

    return CLEV, CLEV_LABELS