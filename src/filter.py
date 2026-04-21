import numpy as np
from scipy import signal

def get_wave_cmap():
    """Create a custom colormap with smooth transitions between the given colors."""
    c0 = 'darkslateblue'
    c1 = 'royalblue'
    c2 = 'cornflowerblue'
    #c2 = 'lightblue'
    c3 = 'lavender'
    c4 = 'white' # 'whitesmoke'
    c5 = 'palegoldenrod'
    c55 = '#EEE600'
    c6 = 'goldenrod'
    c7 = 'indianred' # firebrick
    c8 = 'darkred'

    # colors = [(0.0, c0), (0.1, c1), (0.2, c2), (0.4, c3), (0.48, c4), (0.52, c4), (0.55, c5), (0.65, c55), (0.8, c6), (0.9, c7), (1.0, c8)]
    colors = [(0.0, c0), (0.1, c1), (0.2, c2), (0.4, c3), (0.48, c4), (0.52, c4), (0.6, c5), (0.75, c6), (0.85, c7), (1.0, c8)]
    # colors = [(0.0, c0), (0.32, c0), (0.4, c1), (0.6, c1), (0.68, c2), (0.73, c2), (0.78, c3), (1.0, c3)]
    
    cmap = LinearSegmentedColormap.from_list('wave', colors=colors, N=256)
    return cmap


def fil2dx(data2d, ifil=1):
    """ 
    Smooth pressure perturbations of a 2D field with a 2 dx filter in both dimensions
    ifil: how often apply filter
    """
    a = data2d.copy()
    i = np.arange(1,np.shape(data2d)[1]-1)
    j = np.arange(1,np.shape(data2d)[0]-1)
    
    for ifi in range(0,ifil):
        b=a.copy()
        a[:,i]=0.25*(b[:,i-1]+2*b[:,i]+b[:,i+1])
        b=a.copy()
        a[j,:]=0.25*(b[j-1,:]+2*b[j,:]+b[j+1,:])
    return a


def gaussian_filter_fft_1D(arr, wavelength_x, res_x):
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
    # Pad    
    mirrored_arr = np.pad(arr, (0, len(arr)), mode='reflect')

    # Create the grid in the frequency domain
    kx = np.fft.fftfreq(len(mirrored_arr), d=res_x)

    # Calculate the Gaussian filter in the frequency domain using wavelengths
    # sigma_x = wavelength_x / (2.0 * np.pi)
    sigma_x = wavelength_x
    gaussian = np.exp(-(kx**2) * (sigma_x**2))
    
    # Perform FFT of the input array
    arr_fft = np.fft.fft(mirrored_arr)
    
    # Apply the Gaussian filter in the frequency domain
    arr_fft_filtered = arr_fft * gaussian
    
    # Perform inverse FFT to get the smoothed array
    arr_smoothed = np.fft.ifft(arr_fft_filtered).real
    
    return arr_smoothed[:len(arr)]

def gaussian_fft_smoothing(arr, wavelength_x, wavelength_y, res_x, res_y):
    """
    Apply a Gaussian filter to a 2D array using FFT with mirroring to reduce boundary effects.

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
    
    # Mirror the data to reduce boundary effects
    arr_mirrored = np.pad(arr, ((nx, nx), (ny, ny)), mode='reflect')
    
    # Get new dimensions
    nx_m, ny_m = arr_mirrored.shape
    
    # Create the grid in the frequency domain
    kx = np.fft.fftfreq(nx_m, d=res_x)[:, None]
    ky = np.fft.fftfreq(ny_m, d=res_y)[None, :]
    
    # Calculate the Gaussian filter in the frequency domain using wavelengths
    sigma_x = wavelength_x
    sigma_y = wavelength_y
    # gaussian = np.exp(-0.5 * ((kx**2) * (sigma_x**2) + (ky**2) * (sigma_y**2)))
    gaussian = np.exp(-((kx**2) * (sigma_x**2) + (ky**2) * (sigma_y**2)))

    
    # Perform FFT of the mirrored input array
    arr_fft = np.fft.fft2(arr_mirrored)
    
    # Apply the Gaussian filter in the frequency domain
    arr_fft_filtered = arr_fft * gaussian
    
    # Perform inverse FFT to get the smoothed array
    arr_smoothed = np.fft.ifft2(arr_fft_filtered).real
    
    # Extract the original region
    arr_final = arr_smoothed[nx:nx*2, ny:ny*2]
    
    return arr_final


def butterworth_filter_and_interp(data, z=None, highcut=1/20, fs=1/0.1, order=5):
    """
    Butterworth filter incl. interpolation
    Input
        data (array): first dimension will be reshaped and filtered (z values have to increase from 0 to end)
    
    """

    data_i=data.copy() # aequidistant vertical grid has same number of grid points as terrain following grid
    UNDEF=np.nan 
    for ix in range(0,np.shape(data)[1]):
        data_i[::-1,ix] = np.interp(z[::-1],ds['geom_height'][0,:,0,0].values[::-1],data[::-1,ix])
    data_temp = data_i.T
    
    pert, bgd = butterworth_filter(data_temp, highcut=highcut, fs=fs, order=order)

    return pert, bgd

    # data_i=data.copy() # aequidistant vertical grid has same number of grid points as terrain following grid
    # UNDEF=np.nan 
    # for ix in range(0,np.shape(data)[1]):
    #     data_i[::-1,ix] = np.interp(z[::-1],ds['geom_height'][0,:,0,0].values[::-1],data_temp[::-1,ix])
    # data_lid = data_i.T
    
    # tprime_lid, tbg_lid = butterworthf(data_lid, highcut=1/20000, fs=1/vert_res, order=5)


def butterworth_filter(data, cutoff=1/15, fs=1/0.1, order=5, mode='low'):
    """butterworth filter applied to matrix or each column seperately
        - uses the signal.butter and signal.lfilter functions of the SCIPY library
        - applies a BW filter based on the given order and cutoff frequency
    Input:
        - 2D matrix
        - highcut frequency (1/wavelength) (1/Period)
        - fs (sampling frequency) -> 100m 
        - order of filter = 5
    Output:
        - 2D matrix of perturbations (higher frequencies than cutoff) and background (lower frequencies than cutoff) 
    """
    
    if mode=='low' or mode == 'both':
        b, a = butter_lowpass(cutoff, fs, order=order) 
        # print(b,a)
        # print("filter stable!", np.all(np.abs(np.roots(a))<1))

        # - Filter each column (returns matrix with Nans at bottom and top) - #
        bg = np.full(data.shape, np.nan)
        for col, column in enumerate(data):
            mask = np.isnan(column) # dataarray
            c_masked = column[~mask] # dataarray
            if len(c_masked) >= 10:
                c_mirrored = np.append(np.flip(c_masked, axis=0), c_masked, axis=0) # numpy array
                c_filtered = signal.filtfilt(b,a,c_mirrored, axis=0)
                # c_filtered = signal.lfilter(b,a,c_mirrored, axis=0)
                # c_filtered = signal.filtfilt(b,a,c_masked, axis=0, padtype='odd') # 'even'
                c_filtered = c_filtered[len(c_masked):]
                column_bg = column.copy()
                column_bg[~mask] = c_filtered
                bg[col,:] = column_bg
            else: # column of NANs is just passed through
                bg[col,:] = column
        
        if mode=='low':
            pert = data - bg 
    
    if mode == 'high' or mode == 'both':
        b, a = butter_highpass(cutoff, fs, order=order) 
        # print(b,a)
        # print("filter stable!", np.all(np.abs(np.roots(a))<1))

        # - Filter each column (returns matrix with Nans at bottom and top) - #
        pert = np.full(data.shape, np.nan)
        for col, column in enumerate(data):
            mask = np.isnan(column) # dataarray
            c_masked = column[~mask] # dataarray
            if len(c_masked) >= 10:
                c_mirrored = np.append(np.flip(c_masked, axis=0), c_masked, axis=0) # numpy array
                c_filtered = signal.filtfilt(b,a,c_mirrored, axis=0)
                # c_filtered = signal.lfilter(b,a,c_mirrored, axis=0)
                # c_filtered = signal.filtfilt(b,a,c_masked, axis=0, padtype='odd') # 'even'
                c_filtered = c_filtered[len(c_masked):]
                column_pert = column.copy()
                column_pert[~mask] = c_filtered
                pert[col,:] = column_pert
            else: # column of NANs is just passed through
                pert[col,:] = column

        if mode=='high':
            bg = data - pert

    return pert, bg


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
    

def butter_lowpass(highcut, fs, order=5):
    """
    defines the butterworth filter coefficient based on 
    sample frequency, cut_off frequency and filter order
    """
    nyq = 0.5 * fs # Nyquist frequency
    # highcut = 1/20
    # lowcut = 1/2000000
    # low_crit = lowcut / nyq
   
    high_crit = highcut / nyq # critical frequency ratio
    # Wn = [low_crit, high_crit] # bandpass
    
    b, a = signal.butter(order, high_crit, btype='low', analog=False)
    return b, a


def butter_highpass(highcut, fs, order=5):
    """
    defines the butterworth filter coefficient based on 
    sample frequency, cut_off frequency and filter order
    """
    nyq = 0.5 * fs # Nyquist frequency
    # highcut = 1/20
    # lowcut = 1/2000000
    # low_crit = lowcut / nyq
   
    high_crit = highcut / nyq # critical frequency ratio
    # Wn = [low_crit, high_crit] # bandpass
    
    b, a = signal.butter(order, high_crit, btype='high', analog=False)
    return b, a


# def horizontal_temp_filter_alltime():
#     nx_avg = 50 # 50 -> approx. lambdax=750km assuming 1°=60km, 60->900km
#     # nx_avg2 = 30? 111 km for one lat degree

#     ### - LON - Z - ###
#     lat = -53.75
#     data = ds['t'].sel(latitude=lat)[:,:,:].copy()
#     tmp_equi = data.copy()

#     nx=np.shape(data)[2]

#     data = data.ffill(dim='longitude').bfill(dim='longitude')
#     data = data.pad(longitude=nx_avg, mode="edge")

#     da_fft      = np.fft.fft(data.values,axis=2)
#     da_fft_freq = np.fft.fftfreq(data.values.shape[2])
#     response_func = np.exp(-da_fft_freq**2 * nx_avg**2)
#     da_fft_low = da_fft * np.expand_dims(response_func,axis=0)

#     data_filtered = np.fft.ifft(da_fft_low,axis=2)[:,:,nx_avg:nx+nx_avg]
#     data_filtered = data_filtered.real
#     tprime_lon_z = tmp_equi.copy() - data_filtered # on equidistant grid

#     return tprime_lon_z

    
def horizontal_temp_filter(ds,t,lat,lon,nx_avg=40):
    ### - LON - Z - ###
    data = ds['t'].sel(latitude=lat)[t,:,:].copy()
    tmp_equi = data.copy()

    nx=np.shape(data)[1]

    data = data.ffill(dim='longitude').bfill(dim='longitude')
    data = data.pad(longitude=nx_avg, mode="edge")

    da_fft      = np.fft.fft(data.values,axis=1)
    da_fft_freq = np.fft.fftfreq(data.values.shape[1])
    # response_func = np.exp(-da_fft_freq**2 * nx_avg**2) 
    response_func = np.exp(-da_fft_freq**2 * nx_avg**2 / (4*np.log(2))) # to get ln(2) gain at cutoff like BW!
    da_fft_low = da_fft * np.expand_dims(response_func,axis=0)

    data_filtered = np.fft.ifft(da_fft_low,axis=1)[:,nx_avg:nx+nx_avg]
    data_filtered = data_filtered.real
    tprime_lon_z = tmp_equi.copy() - data_filtered # on equidistant grid

    ### - LAT - Z - ###
    data = ds['t'].sel(longitude=lon)[t,:,:].copy()
    tmp_equi = data.copy()

    nx=np.shape(data)[1]
    data = data.ffill(dim='latitude').bfill(dim='latitude')
    data = data.pad(latitude=nx_avg, mode="edge")

    da_fft      = np.fft.fft(data.values,axis=1)
    da_fft_freq = np.fft.fftfreq(data.values.shape[1])
    response_func = np.exp(-da_fft_freq**2 * nx_avg**2 / (4*np.log(2)))
    da_fft_low = da_fft * np.expand_dims(response_func,axis=0)

    data_filtered = np.fft.ifft(da_fft_low,axis=1)[:,nx_avg:nx+nx_avg]
    data_filtered = data_filtered.real
    tprime_lat_z = tmp_equi - data_filtered # on equidistant grid

    return tprime_lon_z, tprime_lat_z