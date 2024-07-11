import os
import shutil
import sys
# import datetime
import time
import numpy as np
import xarray as xr
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter, AutoMinorLocator
# import statsmodels.api as sm 
# from statsmodels.graphics.gofplots import qqplot_2samples

# import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import imp
# import subroutines
# imp.reload(subroutines)

from xz_single_vars import *
from subroutines import *

import warnings
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", UserWarning)

# import matplotlib
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = ['Tahoma']

CENTER_STR_X = 0.53
CENTER_STR_Y = 0.5

CLEV_100 = np.array([-100,-50,-20,-10,-7,-5,-3,-1,1,3,5,7,10,20,50,100])
CLEV_100_LABELS = np.array([-50,-10,-5,-1,1,5,10,50])

CLEV_30 = [-30,-10,-7,-5,-3,-2,-1,-0.5,0.5,1,2,3,5,7,10,30]
CLEV_30_LABELS = [-10,-5,-2,-0.5,0.5,2,5,10]

# CLEV_20 = [-20.,-7.,-5.,-3.,-2.,-1.,-0.7,-0.5,-0.2,-0.1,0.1,0.2,0.5,0.7,1.,2.,3.,5.,7.,20.]

CLEV_10 = np.array([-10.,-5.,-2.,-1.,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,1.,2.,5.,10.])
CLEV_10_LABELS = np.array([-5.,-1.,-0.5,-0.1,0.1,0.5,1.,5.])

CLEV_5 = [-5.,-2.,-1.,-0.5,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.5,1.,2.,5.]
CLEV_5_LABELS = [-2.,-0.5,-0.2,-0.05,0.05,0.2,0.5,2.]

CLEV_2 = [-2.,-1.,-0.5,-0.3,-0.2,-0.1,-0.07,-0.05,0.05,0.07,0.1,0.2,0.3,0.5,1.,2.]
CLEV_2_LABELS = [-1.,-0.3,-0.1,-0.05,0.05,0.1,0.3,1.]

CLEV_05 = [-0.5,-0.3,-0.1,-0.05,-0.03,-0.01,-0.005,-0.002,0.002,0.005,0.01,0.03,0.05,0.1,0.3,0.5]
CLEV_05_LABELS = [-0.3,-0.05,-0.01,-0.002,0.002,0.01,0.05,0.3]

CLEV_01 = CLEV_100/1000
CLEV_01_LABELS = CLEV_100_LABELS/1000

FIGSIZE = (10,7)
FIGSIZE_2 = (10,8)
FIGSIZE_QQ_PLOT = (5,5)

def vis_eulag(SETTINGS=None):
    """
    Input:
        SETTINGS
    """
    # global FIGSIZE
    # FIGSIZE = eval(SETTINGS['FIGURE_SIZE'])
    
    # Load and combine NETCDF data
    fileLocation = SETTINGS['FILE_LOCATION']
    env_fileName = "env.nc"
    tapef_fileName = "tapef.nc"
    tapes_fileName = "tapes.nc"
    grid_fileName = "grd.nc"

    env_path = os.path.join(fileLocation, env_fileName)
    tapef_path = os.path.join(fileLocation, tapef_fileName)
    tapes_path = os.path.join(fileLocation, tapes_fileName)
    grid_path = os.path.join(fileLocation, grid_fileName)
    
    if int(SETTINGS['USE_TAPES']):
        print('Using tapes.nc...')
        ds = xr.open_dataset(tapes_path)
        
        # Test for pprime (realpprim function of Zbig)
        # dsf = xr.open_dataset(tapef_path)
        # ds['p'][-1,:,:] = dsf['p'][-1,:,:]
        
        # only use last timestep of tapes.nc
        # ds = ds.isel(t=slice(0,23))
        # ds = ds.isel(t=-1)
        # ds = ds.expand_dims('t', axis=0)
    else:
        ds = xr.open_dataset(tapef_path) 
    ds_grid = xr.open_dataset(grid_path)
    ds_env = xr.open_dataset(env_path)
    
    ds_lid_list = []
    list_of_files = sorted(os.listdir(fileLocation))
    for filename in list_of_files:
        if filename.startswith('lid_'):
            ds_lid = xr.open_dataset(os.path.join(fileLocation, filename))
            ds_lid['location'] = filename[:-3]
            ds_lid_list.append(ds_lid)
    
    if len(ds.w.shape) == 3: # 2D case
        # ds_env['time'] = ds['time']
        ds = ds.drop('time')
        ds['ELEVATION'] = ds['ELEVATION'].expand_dims({'t':ds.t})
        # ds = ds.drop('ELEVATION')
        ds = ds.expand_dims('y',axis=2)
    
    if len(ds_env.ue.shape) == 3: # not time depedant case
        ds_env = ds_env.expand_dims({'t':ds.t})
        
    ds = ds.assign_coords({'xcr':ds_grid['xcr']/1000, 'zcr':ds_env['zcr']/1000, 'ycr':ds_grid['ycr']/1000,
                          'zs':ds_grid['zs']/1000, 'zh':ds_grid['zh']/1000, 'gi':ds_grid['gi']})
    ds_env = ds_env.drop('zcr')
    ds = ds.merge(ds_env)

    # ---- Sim parameters -------------- # 
    if int(SETTINGS['USE_DS_GRID_ATTRIBUTES']):
        print('Using grd.nc attributes...')
        ds.attrs = ds_grid.attrs
        ds.attrs['bv'] = ds.attrs['bv'].round(3)
    else:
        ds.attrs['dt'] = 10 # int(SETTINGS['dt'])
        ds.attrs['nt'] = 8640 # int(SETTINGS['nt'])
        # ds.attrs['nstore'] = int(SETTINGS['nstore']) # n steps until tapef storage  
        # ds.attrs['nslice'] = int(SETTINGS['nslice']) # n steps until LIDAR storage  
        # ds.attrs['dx'] = (ds_grid['xcr'][0,1]-ds_grid['xcr'][0,0]) # m
        # ds.attrs['dy'] = int(SETTINGS['dy']) # (ds_grid['ycr'][0,1]-ds_grid['ycr'][0,0]) # m
        # ds.attrs['dz'] = int(SETTINGS['dz'])
        ds.attrs['amp'] = 100 # int(SETTINGS['amp']) # height of mountain
        ds.attrs['xml'] = 1000 # int(SETTINGS['xml']) # width of mountain
        ds.attrs['bv'] = 0.01
        ds.attrs['rg']=287.04
        ds.attrs['g']=9.81
        ds.attrs['dx00']=(ds.xcr[0,1]-ds.xcr[0,0]).values*1000
        ds.attrs['dz00']=(ds.zcr[0,1,0,0]-ds.zcr[0,0,0,0]).values*1000

    ds.attrs['case'] = SETTINGS['CASE']
    ds.attrs['u00'] = ds['ue'][0,0,0,0].values
    ds.attrs['v00'] = ds['ve'][0,0,0,0].values
    ds.attrs['nx'] = np.shape(ds['w'])[3]
    ds.attrs['ny'] = np.shape(ds['w'])[2]
    ds.attrs['nz'] = np.shape(ds['w'])[1]

    ds = dzdx_topo(ds)
    # ---- Sim parameters -------------- # 

    # ---- Constants ------------------- #
    print('Defining constants...')
    pref00=101325.
    # pref00=23700.
    ds.attrs['cp']=3.5*ds.rg # Earth
    ds.attrs['grav']=ds.g
    ds.attrs['cap']=ds.rg/ds.cp
    ds.attrs['capi'] =1/ds.cap
    ds.attrs['capp']=1-ds.cap # Cv/Cp
    ds.attrs['cappi']=1/ds.capp   # Cp/Cv
    ds.attrs['compri']=ds.rg/pref00**ds.cap

    ds.attrs['cp'] = np.round(ds.attrs['cp'],3) # .round(3)
    ds.attrs['cap'] = np.round(ds.attrs['cap'])
    ds.attrs['capp'] = np.round(ds.attrs['capp'])
    ds.attrs['compri'] = np.round(ds.attrs['compri'])
    # ---- Constants ------------------- # 

    # ---- IDL code of AD -------------- # 
    # Forces (MF_x, EF_z, EF_x)
    # h1: ttt (theta)
    # h2: tloc-tte with tloc = thloc*(ploc/pref00)^cap
    # h3: wprime = w
    # h4: tprime+the
    # h5: sqrt(uprime^2+vprime^2)
    # h6: uprime
    # h7: ploc or h12-h12(100), h12=p
    # h9: MFx
    # h11: EFx
    # h12: pprime filtered
    # ---- IDL code of AD -------------- #
    
    start_time = time.time()
    
    # ---- PRESSURE -------------------- #
    print('Defining pref00...', end="", flush=True)
    ds['pref00'] = ds['pr0'].max() # Pa pref00=101325.  
    print('Done')
    # p is pressure perturbation output from EULAG
    # h12 is 2D filtered
    # h7 additionally referenced to x-location before mtn
    # ppe=(rhe*the*compri)^cappi
    ds['pprime'] = ds['p']
    # ds['pprime']=filt_2dx(ds['p'],1) # h12
    # ds['pprime']=filt_2dx(ds['p'],0) # h12
    
    # print(ds['p'][-1,:,0,0])
    # ds['p'] = ds['p'] - ds['p'][:,:,:,0]
    # ds['ploc'] = ds['p'] + ds['pr0']
    # ds['h7'] = ds['pprime'] - ds['pprime'][:,:,:,100] # x=100 before mtn?
    # ---- PRESSURE --------------------- #

    # ---- TEMPERATURE ------------------ #                           
    # ds['thloc'] = ds['the'] + ds['th'] # Theta
    # ds['tloc'] = ds['thloc']*(ds['ploc']/ds['pref00'])**ds.cap
    # ds['tte'] = ds['the']*(ds['pr0']/ds['pref00'])**ds.cap # T_env
    # ds['tloc'] = ds['thloc']*(ds['pr0']/ds['pref00'])**ds.cap # T
    # ds['tprime'] = ds['tloc']-ds['tte'] # T_prime
    # ---- TEMPERATURE ------------------ #  

    # ---- FLUXES ----------------------- #
    # if (int(SETTINGS['PLOT_FLUXES_XZ']) or int(SETTINGS['PLOT_FLUXES_HAVG']) or int(SETTINGS['PLOT_EP_THEOREM'])):
    #     print('Calculating fluxes...')
    #     ds['mfx'] = ds['rh0'] * ds['w'] * ds['uprime'] # h9=h3*h6
    #     ds['mfy'] = ds['rh0'] * ds['w'] * ds['vprime'] # h9=h3*h6
    #     # ds['mfy'] = ds['rh0'].expand_dims({'t':ds.t}) * ds['w'] * ds['vprime'] # h9=h3*h6
    #     ds['MF_U'] = ds['mfx'] * ds['u'] + ds['mfy'] * ds['v'] # = -EFz
        
    #     ds['efx'] = ds['uprime'] * ds['pprime'] # h14, EFx1
    #     ds['efz'] = ds['w'] * ds['pprime'] # h13, EFz1
    #     ds['ep'] = 1/2*(ds.grav/ds.bv)**2 * (ds['tprime']/ds['tte'])**2 # * ds['rh0'].expand_dims({'t':ds.t}) potential energy density
    #     # ds['mfx_u'] = ds['mfx'] * ds['ue'].expand_dims({'t':ds.t})
    #     # ds['efz2'] = # h8
    #     # ds['efzline'] = # h10
    #     ds['MF_U_noFilter'] = ds['MF_U']
    #     ds['efz_noFilter'] = ds['efz']
    # ---- FLUXES ----------------------- #

    # -- Timesteps and xz-planes for visulization -- #
    tvec = eval(SETTINGS['T'])
    # xvec = eval(SETTINGS['X'])
    # yvec = eval(SETTINGS['Y'])
    xvec = [int(ds.nx/2)]
    yvec = [int(ds.ny/2)]
    zvec = eval(SETTINGS['Z'])
    
    # -- Filter -- # 
    # ds.attrs['nx_avg'] = int(ds.xml*4/ds.dy00)
    # ds.attrs['nz_avg'] = int(2*np.pi*ds['ue'][0,0,0,0]/ds.bv/ds.dz00)+1
    ds.attrs['nx_avg'] = int(600*1000/ds.dx00) # 600km
    ds.attrs['nz_avg'] = int(9000/ds.dz00)+1 # 12km

    # if int(SETTINGS['USE_LATEX_FORMAT']):
    plt.style.use('latex_default.mplstyle')
            
    plotdir = SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER']
    if  not os.path.isdir(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER']):
        os.makedirs(plotdir)
        print('Plot folder created...')
    else:
        print('Plot folder already exists and will be overwritten...')
        for f in os.listdir(plotdir):
            if os.path.isdir(os.path.join(plotdir, f)):
                shutil.rmtree(os.path.join(plotdir, f))
            else:
                os.remove(os.path.join(plotdir, f))
    
    # -- WRITE ATTRIBUTES TO TXT FILE -- #
    list_of_strings = [ f'{key} : {ds.attrs[key]}' for key in ds.attrs ]

    # write string one by one adding newline
    with open(plotdir + '/settings_and_params.txt','w') as f:
        [ f.write(f'{st}\n') for st in list_of_strings ]
    
    # -- 3x3 plots -- #
    print('Visualize yz plots in t-x frame for single variable...', end="", flush=True)
    plot_txofyz(ds, SETTINGS)
    print('Done')
    
    print('Visualize xy plots in t-z frame for single variable...', end="", flush=True)
    plot_tzofxy(ds, SETTINGS)
    print('Done')


    print('Visualize x-z planes...', end="", flush=True)
    for y in yvec:
        for t in tvec:
            if y < len(ds.w[0][0]):
                if t < len(ds.w):   
                    if int(SETTINGS['PLOT_INDIVIDUAL_VARS_XZ']):
                        plot_uprime_xz(ds, SETTINGS, t, y=y)
                        plot_wprime_xz(ds, SETTINGS, t, y=y)
                        plot_thprime_xz(ds, SETTINGS, t, y=y)
                        plot_tprime_xz(ds, SETTINGS, t, y=y)
                        plot_pprime_xz(ds, SETTINGS, t, y=y)
                        plot_pprime2_xz(ds, SETTINGS, t, y=y)
                        
                        plot_mfx_xz(ds, SETTINGS, t, y=y)
                        plot_efx1_xz(ds, SETTINGS, t, y=y)
                        plot_efz1_xz(ds, SETTINGS, t, y=y)
                    
                    if int(SETTINGS['PLOT_PRIMES_XZ']):
                        # plot_primes_xz(ds, SETTINGS, t, y=y, theta=0)
                        plot_primes_xz(ds, SETTINGS, t, y=y, theta=1)
                        
                    if int(SETTINGS['PLOT_FLUXES_XZ']):
                        plot_fluxes_xz(ds, SETTINGS, t, y=y)

                else:
                    print('tout = {} is not available. Eventually check CFL criterium.'.format(t))
            else:
                print('yout = {} is not available. Maybe no 3D data.'.format(y))
    print('Done')


    if int(SETTINGS['PLOT_PRIMES_YZ']):
        print('Visualize y-z planes...', end="", flush=True)
        for x in xvec:
            for t in tvec:
                if x < len(ds.w[0,0,0]):
                    if t < len(ds.w):                     
                        plot_primes_yz(ds, SETTINGS, t, x=x, theta=1)
                    else:
                        print('tout = {} is not available. Eventually check CFL criterium.'.format(t))
                else:
                    print('xout = {} is not available. Check x coordinates.'.format(x))
        print('Done')

    
    if int(SETTINGS['PLOT_PRIMES_XY']):
        print('Visualize x-y planes...', end="", flush=True)
        for z in zvec:
            for t in tvec:
                if z < len(ds.w[0]):
                    if t < len(ds.w):                     
                        plot_primes_xy(ds, SETTINGS, t, z=z, theta=1)
                    else:
                        print('tout = {} is not available. Eventually check CFL criterium.'.format(t))
                else:
                    print('zout = {} is not available. Check height of upper bound.'.format(z))
        print('Done')


    if int(SETTINGS['PLOT_SURF']):
        print('Visualize surface vars...', end="", flush=True)
        "Surface and drag plot"
        for y in yvec:
            if y < len(ds.w[0][0]):
                plot_surf(ds, SETTINGS, y=y)
            else:
                print('yout = {} is not available. Maybe no 3D data.'.format(y))
        print('Done')

    if int(SETTINGS['PLOT_FLUXES_HAVG']):
        print('Visualize horizontal avg of fluxes...', end="", flush=True)
        "Horizontal averages of fluxes"
        for y in yvec:
            if y < len(ds.w[0][0]):
                plot_fluxes_horizontalAvg(ds, SETTINGS, y=y)
            else:
                print('yout = {} is not available. Maybe no 3D data.'.format(y))
        print('Done')

    if int(SETTINGS['PLOT_PROFILES']):
        print('Visualize vertical profiles...', end="", flush=True)
        "Vertical profiles of u, N, du/dy, Ri, ..."
        for y in yvec:
            if y < len(ds.w[0][0]):
                plot_vertical_profiles(ds, SETTINGS, y=0)
            else:
                print('yout = {} is not available. Maybe no 3D data.'.format(y))
        print('Done')

    if int(SETTINGS['PLOT_LIDARS']):
        print('Visualize Lidar profiles...', end="", flush=True)
        "Lidar profiles (t-z)"
        for ds_lid in ds_lid_list:
            ds_lid = process_ds_lid(ds_lid, ds)
            plot_lidar_location_primes(ds_lid, ds, SETTINGS, theta=1)
            # plot_lidar_location_fluxes(ds_lid, ds, SETTINGS)
        print('Done')
    
    if int(SETTINGS['PLOT_EP_THEOREM']):
        plot_eliassenPalm_theorem(ds, SETTINGS)
    
    print("--- %s seconds ---" % (time.time() - start_time))


# ----- TZ plots of 2D SLICES ---- #
def plot_tzofxy(ds, SETTINGS):
    "Plots grid of 12 (x horizontal, 4 in vertical) xy plots for varying z and t"
    FIGSIZE_3 = (14,12.5)
    tvec = [2,4,6]
    zvec = [133,100,33] # dz=300m 67=20km, 33=10km
    zvec_l = [40,30,10] # km
    varvec = eval(SETTINGS['VARS'])    
    
    cmap = plt.get_cmap('RdBu_r')
    nx_avg = ds.nx_avg

    for var in varvec:
        fig, axes = plt.subplots(len(zvec),len(tvec), sharex=True, sharey=True, figsize=FIGSIZE_3)
        if var=='TH':
            print('TH, ', end="", flush=True)
            clev = eval(SETTINGS['CLEV_TH'])
            clev_l = eval(SETTINGS['CLEV_TH'] + '_LABELS')
            norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
            i=0
            for z in zvec:
                j=0
                for t in tvec:
                    pcMesh = axes[i,j].pcolormesh(ds.xcr, ds.ycr, ds.th[t,z,:,:],
                                     cmap=cmap, norm=norm, shading='nearest')
                    # cbar = fig.colorbar(pcMesh, ax=, orientation='horizontal', ticks=clev_l, pad=0.07)
                    # cbar.set_label("$\Theta'$ / K")
                    
                    axes[i,j] = format_xy_plot(ds, SETTINGS, axes[i,j], t, z=z)
                    axes[i,j].axis('scaled')

                    if i==0:
                        axes[i,j].set_title('T: ' + str(t*12), weight='bold',pad=10)
                    if i==(len(zvec)-1):
                        axes[i,j].set_xlabel('streamwise x / km')
                        # if j==1:
                            # fig.add_axes(cax)
                            # cbar = fig.colorbar(pcMesh, cax = cax, orientation = 'horizontal',ticks=clev_l)
                            
                    if j==0:
                        axes[i,j].set_ylabel('spanwise / km')
                        axes[i,j].text(0.05,0.05,'Z: ' + str(zvec_l[i]) + 'km', transform=axes[i,j].transAxes, weight='bold')

                    j+=1
                i+=1
            fig_name = 'tzxy_theta.png'
            cbar = fig.colorbar(pcMesh, ax=axes[len(zvec)-1,:], location='bottom', ticks=clev_l, pad=-0.9)
            cbar.set_label("$\Theta'$ / K")

        elif var=='MFX':
            print('MFx, ', end="", flush=True)
            clev = eval(SETTINGS['CLEV_MFX'])
            clev_l = eval(SETTINGS['CLEV_MFX'] + '_LABELS')
            norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
            i=0
            for z in zvec:
                j=0
                for t in tvec:
                    # --------- Calculate and filter fluxes -------------------- # 
                    mfx = ds['w'][t,z,:,:] * (ds['u'][t,z,:,:]-ds['ue'][t,z,:,:]) * ds['rh0'][t,z,:,:]
                    mfx=fft_gaussian_xy(mfx,nx_avg,ny_avg=None)
                    # --------- Calculate and filter fluxes -------------------- #
                    pcMesh = axes[i,j].pcolormesh(ds.xcr, ds.ycr, 1000*mfx,
                                     cmap=cmap, norm=norm, shading='nearest')
                    
                    axes[i,j] = format_xy_plot(ds, SETTINGS, axes[i,j], t, z=z)
                    axes[i,j].axis('scaled')

                    if i==0:
                        axes[i,j].set_title('T: ' + str(t*12) + 'h', weight='bold',pad=10)
                    if i==(len(zvec)-1):
                        axes[i,j].set_xlabel('streamwise x / km')
                    if j==0:
                        axes[i,j].set_ylabel('spanwise / km')
                        axes[i,j].text(0.05,0.05,'Z: ' + str(zvec_l[i]) + 'km', transform=axes[i,j].transAxes, weight='bold')
                    j+=1
                i+=1
            fig_name = 'tzxy_mfx.png'
            cbar = fig.colorbar(pcMesh, ax=axes[len(zvec)-1,:], location='bottom', ticks=clev_l, pad=-0.9)
            cbar.set_label('MF$_x$ / mPa')
            # cbar.set_label("$u'w'$ / m$^2$ s$^{-2}$")
        
        elif var=='MFY':
            print('MFy, ', end="", flush=True)
            clev = eval(SETTINGS['CLEV_MFX'])
            clev_l = eval(SETTINGS['CLEV_MFX'] + '_LABELS')
            norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
            i=0
            for z in zvec:
                j=0
                for t in tvec:
                    # --------- Calculate fluxes -------------------- # 
                    mfy = ds['w'][t,z,:,:] * (ds['v'][t,z,:,:]-ds['ve'][t,z,:,:]) * ds['rh0'][t,z,:,:] 
                    mfy=fft_gaussian_xy(mfy,nx_avg,ny_avg=None)
                    # --------- Calculate and filter fluxes -------------------- #
                    pcMesh = axes[i,j].pcolormesh(ds.xcr, ds.ycr, 1000*mfy,
                                     cmap=cmap, norm=norm, shading='nearest')
                    
                    axes[i,j] = format_xy_plot(ds, SETTINGS, axes[i,j], t, z=z)
                    axes[i,j].axis('scaled')

                    if i==0:
                        axes[i,j].set_title('T: ' + str(t*12) + 'h', weight='bold',pad=10)
                    if i==(len(zvec)-1):
                        axes[i,j].set_xlabel('streamwise x / km')
                    if j==0:
                        axes[i,j].set_ylabel('spanwise / km')
                        axes[i,j].text(0.05,0.05,'Z: ' + str(zvec_l[i]) + 'km', transform=axes[i,j].transAxes, weight='bold')
                    j+=1
                i+=1
            fig_name = 'tzxy_mfy.png'
            cbar = fig.colorbar(pcMesh, ax=axes[len(zvec)-1,:], location='bottom', ticks=clev_l, pad=-0.9)
            cbar.set_label('MF$_y$ / mPa')

        elif var=='EP':
            print('Ep, ', end="", flush=True)
            clev = eval(SETTINGS['CLEV_EP'])
            clev_l = eval(SETTINGS['CLEV_EP'] + '_LABELS')
            norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
            i=0
            for z in zvec:
                j=0
                for t in tvec:
                    # --------- Calculate fluxes -------------------- #                           
                    thloc = ds['the'][t,z,:,:] + ds['th'][t,z,:,:] # Theta
                    # ds['tloc'] = ds['thloc']*(ds['ploc']/ds['pref00'])**ds.cap
                    tte = ds['the'][t,z,:,:]*(ds['pr0'][t,z,:,:]/ds['pref00'])**ds.cap # T_env
                    tloc = thloc*(ds['pr0'][t,z,:,:]/ds['pref00'])**ds.cap # T
                    tprime = tloc-tte

                    ep = 1/2*(ds.grav/ds.bv)**2 * (tprime/tte)**2 # * ds['rh0'].expand_dims({'t':ds.t}) potential energy density
                    ep=fft_gaussian_xy(ep,nx_avg,ny_avg=None)
                    # --------- Calculate and filter fluxes -------------------- #
                    pcMesh = axes[i,j].pcolormesh(ds.xcr, ds.ycr, ep,
                                     cmap=cmap, norm=norm, shading='nearest')
                    
                    axes[i,j] = format_xy_plot(ds, SETTINGS, axes[i,j], t, z=z)
                    axes[i,j].axis('scaled')

                    if i==0:
                        axes[i,j].set_title('T: ' + str(t*12) + 'h', weight='bold',pad=10)
                    if i==(len(zvec)-1):
                        axes[i,j].set_xlabel('streamwise x / km')
                    if j==0:
                        axes[i,j].set_ylabel('spanwise / km')
                        axes[i,j].text(0.05,0.05,'Z: ' + str(zvec_l[i]) + 'km', transform=axes[i,j].transAxes, weight='bold')
                    j+=1
                i+=1
            fig_name = 'tzxy_ep.png'
            cbar = fig.colorbar(pcMesh, ax=axes[len(zvec)-1,:], location='bottom', ticks=clev_l, pad=-0.9)
            cbar.set_label("E$_P$ / J kg$^{-1}$")

        xrange = eval(SETTINGS['XRANGE'])
        if xrange != 'NONE':    
            axes[0,0].set_xlim(xrange)
        
        yrange = eval(SETTINGS['YRANGE'])
        if yrange != 'NONE':    
            axes[0,0].set_ylim(yrange)

        # --- Add text --- #
        # center_str = 'T: ' + str(int(ds.dt00*t*ds.nplot/3600)) +'h, z: ' + str(int(ds.dz00*z/1000)) + 'km' 
        # fig.text(CENTER_STR_X,CENTER_STR_Y,center_str,horizontalalignment='center',
        #         verticalalignment='center', ma='center', fontsize=14)

        # --- Save fig --- #
        fig.tight_layout()
        fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                    format='png', dpi=150) # orientation='portrait'


# ----- XT plots of 2D SLICES ---- #
def plot_txofyz(ds, SETTINGS):
    "Plots grid of 9 yz plots for varying x and t"
    FIGSIZE_3 = (14,12.5)
    tvec = [2,4,6] # 23h,48h,72h
    x1 = int((-2400+50*24)*1000/ds.dx00+ds.nx/2)
    x2 = int((-2400+50*48)*1000/ds.dx00+ds.nx/2)
    x3 = int((-2400+50*72)*1000/ds.dx00+ds.nx/2)
    xvec = [x1,x2,x3] # position of TD at corresponding points in time
    # xvec = [int(7/16*ds.nx),int(ds.nx/2),int(9/16*ds.nx)] # dz=300m 67=20km, 33=10km
    # xvec_l = [40,30,10] # km
    varvec = eval(SETTINGS['VARS'])    
    
    cmap = plt.get_cmap('RdBu_r')
    nx_avg = ds.nx_avg
    nz_avg = ds.nz_avg
    

    for var in varvec:
        fig, axes = plt.subplots(len(tvec),len(xvec), sharex=True, sharey=True, figsize=FIGSIZE_3)
        if var=='TH':
            print('TH, ', end="", flush=True)
            clev = eval(SETTINGS['CLEV_TH'])
            clev_l = eval(SETTINGS['CLEV_TH'] + '_LABELS')
            norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
            j=0
            for x in xvec:
                i=0
                for t in tvec:
                    pcMesh = axes[i,j].pcolormesh(ds.ycr[:,x], ds.zcr[t,:,:,x], ds.th[t,:,:,x],
                                     cmap=cmap, norm=norm, shading='nearest')
                    # cbar = fig.colorbar(pcMesh, ax=, orientation='horizontal', ticks=clev_l, pad=0.07)
                    # cbar.set_label("$\Theta'$ / K")
                    
                    axes[i,j] = format_yz_plot(ds, SETTINGS, axes[i,j], t, x=x)
                    # axes[i,j].axis('scaled')

                    if i==0:
                        axes[i,j].set_title('x: ' + str(xvec[j]*ds.dx00/1000) + 'km', weight='bold',pad=10)
                    if i==(len(xvec)-1):
                        axes[i,j].set_xlabel('spanwise y / km')
                    if j==0:
                        axes[i,j].set_ylabel('altitude / km')
                        axes[i,j].text(0.05,0.15,'T: ' + str(t*12) + 'h', transform=axes[i,j].transAxes, weight='bold')
                    i+=1
                j+=1
            fig_name = 'txyz_theta.png'
            cbar = fig.colorbar(pcMesh, ax=axes[len(xvec)-1,:], location='bottom', ticks=clev_l, pad=-0.9)
            cbar.set_label("$\Theta'$ / K")

        elif var=='MFX':
            print('MFx, ', end="", flush=True)
            clev = eval(SETTINGS['CLEV_MFX'])
            clev_l = eval(SETTINGS['CLEV_MFX'] + '_LABELS')
            norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
            j=0
            for x in xvec:
                i=0
                for t in tvec:
                    # --------- Calculate and filter fluxes -------------------- # 
                    mfx = ds['w'][t,:,:,x] * (ds['u'][t,:,:,x]-ds['ue'][t,:,:,x]) * ds['rh0'][t,:,:,x]
                    mfx=fft_gaussian_filter(mfx,nz_avg,nx_avg=nx_avg,usey=1)
                    # --------- Calculate and filter fluxes -------------------- #
                    pcMesh = axes[i,j].pcolormesh(ds.ycr[:,x], ds.zcr[t,:,:,x], 1000*mfx,
                                     cmap=cmap, norm=norm, shading='nearest')
                    
                    axes[i,j] = format_yz_plot(ds, SETTINGS, axes[i,j], t, x=x)
                    # axes[i,j].axis('scaled')

                    if i==0:
                        axes[i,j].set_title('x: ' + str(xvec[j]*ds.dx00/1000) + 'km', weight='bold',pad=10)
                    if i==(len(xvec)-1):
                        axes[i,j].set_xlabel('spanwise y / km')
                    if j==0:
                        axes[i,j].set_ylabel('altitude / km')
                        axes[i,j].text(0.05,0.15,'T: ' + str(t*12) + 'h', transform=axes[i,j].transAxes, weight='bold')
                    i+=1
                j+=1
            fig_name = 'txyz_mfx.png'
            cbar = fig.colorbar(pcMesh, ax=axes[len(xvec)-1,:], location='bottom', ticks=clev_l, pad=-0.9)
            cbar.set_label('MF$_x$ / mPa')
            # cbar.set_label("$u'w'$ / m$^2$ s$^{-2}$")
        
        elif var=='MFY':
            print('MFy, ', end="", flush=True)
            clev = eval(SETTINGS['CLEV_MFX'])
            clev_l = eval(SETTINGS['CLEV_MFX'] + '_LABELS')
            norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
            j=0
            for x in xvec:
                i=0
                for t in tvec:
                    # --------- Calculate fluxes -------------------- # 
                    mfy = ds['w'][t,:,:,x] * (ds['v'][t,:,:,x]-ds['ve'][t,:,:,x]) * ds['rh0'][t,:,:,x] 
                    mfy=fft_gaussian_filter(mfy,nz_avg,nx_avg=nx_avg,usey=1)
                    # --------- Calculate and filter fluxes -------------------- #
                    pcMesh = axes[i,j].pcolormesh(ds.ycr[:,x], ds.zcr[t,:,:,x], 1000*mfy,
                                     cmap=cmap, norm=norm, shading='nearest')
                    
                    axes[i,j] = format_yz_plot(ds, SETTINGS, axes[i,j], t, x=x)
                    # axes[i,j].axis('scaled')

                    if i==0:
                        axes[i,j].set_title('x: ' + str(xvec[j]*ds.dx00/1000) + 'km', weight='bold',pad=10)
                    if i==(len(xvec)-1):
                        axes[i,j].set_xlabel('spanwise y / km')
                    if j==0:
                        axes[i,j].set_ylabel('altitude / km')
                        axes[i,j].text(0.05,0.15,'T: ' + str(t*12) + 'h', transform=axes[i,j].transAxes, weight='bold')
                    i+=1
                j+=1
            fig_name = 'txyz_mfy.png'
            cbar = fig.colorbar(pcMesh, ax=axes[len(xvec)-1,:], location='bottom', ticks=clev_l, pad=-0.9)
            cbar.set_label('MF$_y$ / mPa')

        elif var=='EP':
            print('Ep, ', end="", flush=True)
            clev = eval(SETTINGS['CLEV_EP'])
            clev_l = eval(SETTINGS['CLEV_EP'] + '_LABELS')
            norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
            j=0
            for x in xvec:
                i=0
                for t in tvec:
                    # --------- Calculate fluxes -------------------- #                           
                    thloc = ds['the'][t,:,:,x] + ds['th'][t,:,:,x] # Theta
                    # ds['tloc'] = ds['thloc']*(ds['ploc']/ds['pref00'])**ds.cap
                    tte = ds['the'][t,:,:,x]*(ds['pr0'][t,:,:,x]/ds['pref00'])**ds.cap # T_env
                    tloc = thloc*(ds['pr0'][t,:,:,x]/ds['pref00'])**ds.cap # T
                    tprime = tloc-tte

                    ep = 1/2*(ds.grav/ds.bv)**2 * (tprime/tte)**2 # * ds['rh0'].expand_dims({'t':ds.t}) potential energy density
                    ep=fft_gaussian_filter(ep,nz_avg,nx_avg=nx_avg,usey=1)
                    # --------- Calculate and filter fluxes -------------------- #
                    pcMesh = axes[i,j].pcolormesh(ds.ycr[:,x], ds.zcr[t,:,:,x], ep,
                                     cmap=cmap, norm=norm, shading='nearest')
                    
                    axes[i,j] = format_yz_plot(ds, SETTINGS, axes[i,j], t, x=x)
                    # axes[i,j].axis('scaled')

                    if i==0:
                        axes[i,j].set_title('x: ' + str(xvec[j]*ds.dx00/1000) + 'km', weight='bold',pad=10)
                    if i==(len(xvec)-1):
                        axes[i,j].set_xlabel('spanwise y / km')
                    if j==0:
                        axes[i,j].set_ylabel('altitude / km')
                        axes[i,j].text(0.05,0.15,'T: ' + str(t*12) + 'h', transform=axes[i,j].transAxes, weight='bold')
                    i+=1
                j+=1
            fig_name = 'txyz_ep.png'
            cbar = fig.colorbar(pcMesh, ax=axes[len(xvec)-1,:], location='bottom', ticks=clev_l, pad=-0.9)
            cbar.set_label("E$_P$ / J kg$^{-1}$")

        xrange = eval(SETTINGS['XRANGE'])
        if xrange != 'NONE':    
            axes[0,0].set_xlim(xrange)
        
        yrange = eval(SETTINGS['YRANGE'])
        if yrange != 'NONE':    
            axes[0,0].set_ylim(yrange)

        # --- Add text --- #
        # center_str = 'T: ' + str(int(ds.dt00*t*ds.nplot/3600)) +'h, z: ' + str(int(ds.dz00*z/1000)) + 'km' 
        # fig.text(CENTER_STR_X,CENTER_STR_Y,center_str,horizontalalignment='center',
        #         verticalalignment='center', ma='center', fontsize=14)

        # --- Save fig --- #
        fig.tight_layout()
        fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                    format='png', dpi=150) # orientation='portrait'


# ----- 2D SLICE FUNCTIONS ---- #    
def plot_primes_xz(ds, SETTINGS, t, y=0, theta=0):
    "Plots u, w, t, theta, and p prime in xz plane"
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=FIGSIZE_2)
    
    cmap = plt.get_cmap('RdBu_r')
    # if t==0:
    # used for shifting statlb simulation for easier analysis
    #     ds.xcr[y]=ds.xcr[y].values+400.0
    
    ## U ##
    clev = eval(SETTINGS['CLEV_U'])
    clev_l = eval(SETTINGS['CLEV_U'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)    
    pcMesh0 = ax0.pcolormesh(ds.xcr[y], ds.zcr[t,:,y,:], ds.u[t,:,y,:]-ds.ue[t,:,y,:],
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh0, ax=ax0, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.set_label("$u'=u-u_e$ / m s$^{-1}$")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_U'] == 'CLEV_01' or SETTINGS['CLEV_U'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    
    ## W or V ##
    clev = eval(SETTINGS['CLEV_W'])
    clev_l = eval(SETTINGS['CLEV_W'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    # pcMesh1 = ax1.pcolormesh(ds.xcr[y], ds.zcr[t,:,y,:], ds.v[t,:,y,:],
    pcMesh1 = ax1.pcolormesh(ds.xcr[y], ds.zcr[t,:,y,:], ds.w[t,:,y,:],
                            cmap=cmap, norm=norm, shading='nearest')
    # cbar = fig.colorbar(pcMesh1, ax=ax1, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.set_label("$w'$ (vertical velocity) / m s$^{-1}$")
    cbar.set_label("$v'$ (meridional velocity) / m s$^{-1}$")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_W'] == 'CLEV_01' or SETTINGS['CLEV_W'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    # cbar1.ax.set_xticklabels(cbar1.ax.get_xticklabels(), rotation=-30) # for rotation of labels
    
    ## THETA ##
    clev = eval(SETTINGS['CLEV_TH'])
    clev_l = eval(SETTINGS['CLEV_TH'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    if theta:     
        pcMesh2 = ax2.pcolormesh(ds.xcr[y], ds.zcr[t,:,y,:], ds.th[t,:,y,:],
                            cmap=cmap, norm=norm, shading='nearest')
        cbar = fig.colorbar(pcMesh2, ax=ax2, orientation='horizontal', ticks=clev_l, pad=0.07)
        cbar.set_label("$\Theta'$ / K")
        
    else:
        ## TEMPERATURE CALCULATION##                             
        thloc = ds['the'][t,:,y,:] + ds['th'][t,:,y,:] # Theta
        # ds['tloc'] = ds['thloc']*(ds['ploc']/ds['pref00'])**ds.cap
        tte = ds['the'][t,:,y,:]*(ds['pr0'][t,:,y,:]/ds['pref00'])**ds.cap # T_env
        tloc = thloc*(ds['pr0'][t,:,y,:]/ds['pref00'])**ds.cap # T
        tprime = tloc-tte
        ## TEMPERATURE CALCULATION##
        pcMesh2 = ax2.pcolormesh(ds.xcr[y], ds.zcr[t,:,y,:], tprime,
                            cmap=cmap, norm=norm, shading='nearest')
        cbar = fig.colorbar(pcMesh2, ax=ax2, orientation='horizontal', ticks=clev_l, pad=0.07)
        cbar.set_label("$T'$ / K")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_TH'] == 'CLEV_01' or SETTINGS['CLEV_TH'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
      # PRESSURE ##  
    clev = eval(SETTINGS['CLEV_P'])
    clev_l = eval(SETTINGS['CLEV_P'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    
    # ppp = ds.p[t,:,y,:] - ds.p[t,:,y,:].mean(axis=1)
    # ppe = (ds['rhe'][t,:,y,:]*ds['the'][t,:,y,:]*ds.compri)**ds.cappi
    
    # the = ds['the'][t,:,y,:] 
    # ppe = ds['pref00'] * (288/the)**ds.capi
    # thloc = ds['the'][t,:,y,:] + ds['th'][t,:,y,:]
    # ppp = ds['pref00'] * (288/thloc)**ds.capi - ppe
    
    # Reverse realpprim()
    # ppp = ds['p'][t,:,y,:] * ds.dt00 / (2*ds['rh0'][t,:,y,:])
    # ppp = ppp - ppp.mean(axis=1)
    # ppp = ppp - ds['ppe'][t,:,y,:]
    
    ppp = ds['p'][t,:,y,:]
    # pcMesh3 = ax3.pcolormesh(ds.xcr[y], ds.zcr[t,:,y,:], ppp,cmap=cmap)
    pcMesh3 = ax3.pcolormesh(ds.xcr[y], ds.zcr[t,:,y,:], ppp,
                                cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh3, ax=ax3, orientation='horizontal', ticks=clev_l, pad=0.07)
    # cbar = fig.colorbar(pcMesh3, ax=ax3, orientation='horizontal', pad=0.07)
    cbar.set_label("$p'$ / Pa")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_P'] == 'CLEV_01' or SETTINGS['CLEV_P'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
    ## ISENTROPES ##
    ax0 = format_xz_plot(ds, SETTINGS, ax0, t, y=y)
    ax1 = format_xz_plot(ds, SETTINGS, ax1, t, y=y)
    ax2 = format_xz_plot(ds, SETTINGS, ax2, t, y=y)
    ax3 = format_xz_plot(ds, SETTINGS, ax3, t, y=y)
    
    xrange = eval(SETTINGS['XRANGE'])
    if xrange != 'NONE':    
        ax0.set_xlim(xrange)
    
    yrange = eval(SETTINGS['ZRANGE'])
    if yrange != 'NONE':    
        ax0.set_ylim(yrange)

    ax0.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax1.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax0.xaxis.set_label_position('top') 
    ax1.xaxis.set_label_position('top') 
    ax0.tick_params(labelbottom=False,labeltop=True)
    ax1.tick_params(labelbottom=False,labeltop=True)
    ax2.tick_params(labelbottom=False,labeltop=False)
    ax3.tick_params(labelbottom=False,labeltop=False)
    ax0.set_ylabel('altitude / km')
    ax2.set_ylabel('altitude / km')
    
    center_str = 'T: ' + str(int(ds.dt00*t*ds.nplot/3600)) +'h, y: ' + str(int(y-ds.ny/2)*ds.dy00/1000) + 'km' 
    # center_str = 'T: ' + str(int(ds.dt00*t*ds.nplot/60)) +'min, y: ' + str(int(y-ds.ny/2)*ds.dy00/1000) + 'km' 
    fig.text(CENTER_STR_X,CENTER_STR_Y,center_str,horizontalalignment='center',
            verticalalignment='center', ma='center', fontsize=14)

    # Save fig
    fig.tight_layout()
    if theta:
        if t<10:
            fig_name = 'primes_xzth_y' + str(y) + '_t0' +str(t) + '.png'
        else:
            fig_name = 'primes_xzth_y' + str(y) + '_t' +str(t) + '.png'
    else:
        fig_name = 'primes_xztemp_y' + str(y) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'
    

def plot_primes_yz(ds, SETTINGS, t, x=0, theta=0):
    "Plots u, w, t, theta, and p prime in xz plane"
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=FIGSIZE_2)
    
    cmap = plt.get_cmap('RdBu_r')
    
    ## U ##
    clev = eval(SETTINGS['CLEV_U'])
    clev_l = eval(SETTINGS['CLEV_U'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)    
    pcMesh0 = ax0.pcolormesh(ds.ycr[:,x], ds.zcr[t,:,:,x], ds.u[t,:,:,x]-ds.ue[t,:,:,x],
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh0, ax=ax0, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.set_label("$u'=u-u_e$ / m s$^{-1}$")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_U'] == 'CLEV_01' or SETTINGS['CLEV_U'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    
    ## W or V ##
    clev = eval(SETTINGS['CLEV_W'])
    clev_l = eval(SETTINGS['CLEV_W'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    pcMesh1 = ax1.pcolormesh(ds.ycr[:,x], ds.zcr[t,:,:,x], ds.w[t,:,:,x],
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh1, ax=ax1, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.set_label("$w'$ (vertical velocity) / m s$^{-1}$")
    # cbar.set_label("$v'$ (meridional velocity) / m s$^{-1}$")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_W'] == 'CLEV_01' or SETTINGS['CLEV_W'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    # cbar1.ax.set_xticklabels(cbar1.ax.get_xticklabels(), rotation=-30) # for rotation of labels
    
    ## THETA ##
    clev = eval(SETTINGS['CLEV_TH'])
    clev_l = eval(SETTINGS['CLEV_TH'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    if theta:     
        pcMesh2 = ax2.pcolormesh(ds.ycr[:,x], ds.zcr[t,:,:,x], ds.th[t,:,:,x],
                            cmap=cmap, norm=norm, shading='nearest')
        cbar = fig.colorbar(pcMesh2, ax=ax2, orientation='horizontal', ticks=clev_l, pad=0.07)
        cbar.set_label("$\Theta'$ / K")
        
    else:
        ## TEMPERATURE CALCULATION##                             
        thloc = ds['the'][t,:,:,x] + ds['th'][t,:,:,x] # Theta
        # ds['tloc'] = ds['thloc']*(ds['ploc']/ds['pref00'])**ds.cap
        tte = ds['the'][t,:,:,x]*(ds['pr0'][t,:,:,x]/ds['pref00'])**ds.cap # T_env
        tloc = thloc*(ds['pr0'][t,:,:,x]/ds['pref00'])**ds.cap # T
        tprime = tloc-tte
        ## TEMPERATURE CALCULATION##
        pcMesh2 = ax2.pcolormesh(ds.ycr[:,x], ds.zcr[t,:,:,x], tprime,
                            cmap=cmap, norm=norm, shading='nearest')
        cbar = fig.colorbar(pcMesh2, ax=ax2, orientation='horizontal', ticks=clev_l, pad=0.07)
        cbar.set_label("$T'$ / K")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_TH'] == 'CLEV_01' or SETTINGS['CLEV_TH'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
      # PRESSURE ##  
    clev = eval(SETTINGS['CLEV_P'])
    clev_l = eval(SETTINGS['CLEV_P'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    # pcMesh3 = ax3.pcolormesh(ds.xcr[y], ds.zcr[t,:,y,:], ds.pprime[t,:,y,:],
    #                         cmap=cmap, norm=norm, shading='nearest')
    pcMesh3 = ax3.pcolormesh(ds.ycr[:,x], ds.zcr[t,:,:,x], ds.p[t,:,:,x],
                            cmap=cmap, norm=norm, shading='nearest')
    # pcMesh3 = ax3.pcolormesh(ds.xcr[y], ds.zcr[t,:,y,:], ds.vprime[t,:,y,:],
    #                         cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh3, ax=ax3, orientation='horizontal', ticks=clev_l, pad=0.07)
    # cbar = fig.colorbar(pcMesh3, ax=ax3, orientation='horizontal', pad=0.07)
    cbar.set_label("$p'_z$' / Pa")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_P'] == 'CLEV_01' or SETTINGS['CLEV_P'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
    ## ISENTROPES ##
    ax0 = format_yz_plot(ds, SETTINGS, ax0, t, x=x)
    ax1 = format_yz_plot(ds, SETTINGS, ax1, t, x=x)
    ax2 = format_yz_plot(ds, SETTINGS, ax2, t, x=x)
    ax3 = format_yz_plot(ds, SETTINGS, ax3, t, x=x)
    
    xrange = eval(SETTINGS['YRANGE'])
    if xrange != 'NONE':    
        ax0.set_xlim(xrange)
    
    yrange = eval(SETTINGS['ZRANGE'])
    if yrange != 'NONE':    
        ax0.set_ylim(yrange)

    ax0.set_xlabel('spanwise y / km') # change to longitudes, latitude
    ax1.set_xlabel('spanwise y / km') # change to longitudes, latitude
    ax0.xaxis.set_label_position('top') 
    ax1.xaxis.set_label_position('top') 
    ax0.tick_params(labelbottom=False,labeltop=True)
    ax1.tick_params(labelbottom=False,labeltop=True)
    ax2.tick_params(labelbottom=False,labeltop=False)
    ax3.tick_params(labelbottom=False,labeltop=False)
    ax0.set_ylabel('altitude / km')
    ax2.set_ylabel('altitude / km')
    
    center_str = 'T: ' + str(int(ds.dt00*t*ds.nplot/3600)) +'h, x: ' + str(int(ds.dx00*x/1000)) + 'km' 
    fig.text(CENTER_STR_X,CENTER_STR_Y,center_str,horizontalalignment='center',
            verticalalignment='center', ma='center', fontsize=14)

    # Save fig
    fig.tight_layout()
    if theta:
        fig_name = 'primes_yzth_x' + str(x) + '_t' +str(t) + '.png'
    else:
        fig_name = 'primes_yztemp_x' + str(x) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'


def plot_primes_xy(ds, SETTINGS, t, z=0, theta=0):
    "Plots u, w, t, theta, and p prime in xz plane"
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=FIGSIZE_2)
    
    cmap = plt.get_cmap('RdBu_r')
    
    ## U ##
    clev = eval(SETTINGS['CLEV_U'])
    clev_l = eval(SETTINGS['CLEV_U'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)    
    pcMesh0 = ax0.pcolormesh(ds.xcr, ds.ycr, ds.u[t,z,:,:]-ds.ue[t,z,:,:],
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh0, ax=ax0, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.set_label("$u'=u-u_e$ / m s$^{-1}$")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_U'] == 'CLEV_01' or SETTINGS['CLEV_U'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    
    ## W or V ##
    clev = eval(SETTINGS['CLEV_W'])
    clev_l = eval(SETTINGS['CLEV_W'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    pcMesh1 = ax1.pcolormesh(ds.xcr, ds.ycr, ds.w[t,z,:,:],
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh1, ax=ax1, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.set_label("$w'$ (vertical velocity) / m s$^{-1}$")
    # cbar.set_label("$v'$ (meridional velocity) / m s$^{-1}$")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_W'] == 'CLEV_01' or SETTINGS['CLEV_W'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    # cbar1.ax.set_xticklabels(cbar1.ax.get_xticklabels(), rotation=-30) # for rotation of labels
    
    ## THETA ##
    clev = eval(SETTINGS['CLEV_TH'])
    clev_l = eval(SETTINGS['CLEV_TH'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    if theta:     
        pcMesh2 = ax2.pcolormesh(ds.xcr, ds.ycr, ds.th[t,z,:,:],
                            cmap=cmap, norm=norm, shading='nearest')
        cbar = fig.colorbar(pcMesh2, ax=ax2, orientation='horizontal', ticks=clev_l, pad=0.07)
        cbar.set_label("$\Theta'$ / K")
        
    else:
        ## TEMPERATURE CALCULATION##                             
        thloc = ds['the'][t,z,:,:] + ds['th'][t,z,:,:] # Theta
        # ds['tloc'] = ds['thloc']*(ds['ploc']/ds['pref00'])**ds.cap
        tte = ds['the'][t,z,:,:]*(ds['pr0'][t,z,:,:]/ds['pref00'])**ds.cap # T_env
        tloc = thloc*(ds['pr0'][t,z,:,:]/ds['pref00'])**ds.cap # T
        tprime = tloc-tte
        ## TEMPERATURE CALCULATION##
        pcMesh2 = ax2.pcolormesh(ds.xcr, ds.ycr, tprime,
                            cmap=cmap, norm=norm, shading='nearest')
        cbar = fig.colorbar(pcMesh2, ax=ax2, orientation='horizontal', ticks=clev_l, pad=0.07)
        cbar.set_label("$T'$ / K")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_TH'] == 'CLEV_01' or SETTINGS['CLEV_TH'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
      # PRESSURE ##  
    clev = eval(SETTINGS['CLEV_P'])
    clev_l = eval(SETTINGS['CLEV_P'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    # pcMesh3 = ax3.pcolormesh(ds.xcr[y], ds.zcr[t,:,y,:], ds.pprime[t,:,y,:],
    #                         cmap=cmap, norm=norm, shading='nearest')
    pcMesh3 = ax3.pcolormesh(ds.xcr, ds.ycr, ds.p[t,z,:,:],
                            cmap=cmap, norm=norm, shading='nearest')
    # pcMesh3 = ax3.pcolormesh(ds.xcr[y], ds.zcr[t,:,y,:], ds.vprime[t,:,y,:],
    #                         cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh3, ax=ax3, orientation='horizontal', ticks=clev_l, pad=0.07)
    # cbar = fig.colorbar(pcMesh3, ax=ax3, orientation='horizontal', pad=0.07)
    cbar.set_label("$p'_z$' / Pa")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_P'] == 'CLEV_01' or SETTINGS['CLEV_P'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
    ## ISENTROPES ##
    ax0 = format_xy_plot(ds, SETTINGS, ax0, t, z=z)
    ax1 = format_xy_plot(ds, SETTINGS, ax1, t, z=z)
    ax2 = format_xy_plot(ds, SETTINGS, ax2, t, z=z)
    ax3 = format_xy_plot(ds, SETTINGS, ax3, t, z=z)
    
    xrange = eval(SETTINGS['XRANGE'])
    if xrange != 'NONE':    
        ax0.set_xlim(xrange)
    
    yrange = eval(SETTINGS['YRANGE'])
    if yrange != 'NONE':    
        ax0.set_ylim(yrange)

    ax0.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax1.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax0.xaxis.set_label_position('top') 
    ax1.xaxis.set_label_position('top') 
    ax0.tick_params(labelbottom=False,labeltop=True)
    ax1.tick_params(labelbottom=False,labeltop=True)
    ax2.tick_params(labelbottom=False,labeltop=False)
    ax3.tick_params(labelbottom=False,labeltop=False)
    ax0.set_ylabel('spanwise / km')
    ax2.set_ylabel('spanwise / km')
    
    center_str = 'T: ' + str(int(ds.dt00*t*ds.nplot/3600)) +'h, z: ' + str(int(ds.dz00*z/1000)) + 'km' 
    fig.text(CENTER_STR_X,CENTER_STR_Y,center_str,horizontalalignment='center',
            verticalalignment='center', ma='center', fontsize=14)

    # Save fig
    fig.tight_layout()
    if theta:
        fig_name = 'primes_xyth_z' + str(z) + '_t' +str(t) + '.png'
    else:
        fig_name = 'primes_xytemp_z' + str(z) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'


# --- FLUXES --- #
def plot_fluxes_xz(ds, SETTINGS, t, y=0):
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=FIGSIZE_2)
    
    # --------- Calculate fluxes -------------------- # 
    mfx = ds['rh0'][t,:,y,:] * ds['w'][t,:,y,:] * (ds['u'][t,:,y,:]-ds['ue'][t,:,y,:]) # h9=h3*h6
    mfy = ds['rh0'][t,:,y,:] * ds['w'][t,:,y,:] * (ds['v'][t,:,y,:]-ds['ve'][t,:,y,:])
    MF_U = mfx * ds['u'][t,:,y,:] + mfy * ds['v'][t,:,y,:] # = -EFz
    
    efx = (ds['u'][t,:,y,:]-ds['ue'][t,:,y,:]) * ds['pprime'][t,:,y,:] # h14, EFx1
    efz = ds['w'][t,:,y,:] * ds['pprime'][t,:,y,:] # h13, EFz1

    ## TEMPERATURE CALCULATION##                             
    thloc = ds['the'][t,:,y,:] + ds['th'][t,:,y,:] # Theta
    # ds['tloc'] = ds['thloc']*(ds['ploc']/ds['pref00'])**ds.cap
    tte = ds['the'][t,:,y,:]*(ds['pr0'][t,:,y,:]/ds['pref00'])**ds.cap # T_env
    tloc = thloc*(ds['pr0'][t,:,y,:]/ds['pref00'])**ds.cap # T
    tprime = tloc-tte

    ep = 1/2*(ds.grav/ds.bv)**2 * (tprime/tte)**2 # * ds['rh0'].expand_dims({'t':ds.t}) potential energy density
    # --------- Calculate fluxes -------------------- # 

    ## INTERPOLATE TO REGULAR VERTICAL GRID ##
    # elev1=ds['zcr'][:,y,:] # zcr coordinate
    # elev2=ds['ELEVATION'][0,:,y,:] # ELEVATION var
    # z = np.linspace(0,50,26)
    # z = np.linspace(0,50,501) # ds['dx']=100
    # Dont start from 0 eventually
    z = np.linspace(0,ds['zcr'][t,-1,y,0].values,int(ds.nz)) # ds['dx']=200

    mfx = interp_elev_to_z(mfx,ds.zcr[t,:,y,:],z)
    MF_U = interp_elev_to_z(MF_U,ds.zcr[t,:,y,:],z)
    efx = interp_elev_to_z(efx,ds.zcr[t,:,y,:],z)
    efz = interp_elev_to_z(efz,ds.zcr[t,:,y,:],z)
    ep = interp_elev_to_z(ep,ds.zcr[t,:,y,:],z)

    ## AVERAGE VERTICAL AND HORIZONTAL
    # nx_avg = int(2*np.pi*ds['ue'][0,0,0,0]/ds.bv/ds.dx00)+1
    # nz_avg = int(2*np.pi*ds['ue'][0,0,0,0]/ds.bv/ds.dz00)+1
    # nz_avg = int(2*np.pi*(ds['ue'][0,0,0,0]-50/3.6)/ds.bv/ds.dz00)+1
    
    nx_avg = ds.nx_avg
    nz_avg = ds.nz_Avg
    
    mfx=filter_2D(mfx,nx_avg,nz_avg,mode=1)
    # mfx=filter_1Dz(mfx,nz_avg,mode=1)
    MF_U=filter_2D(MF_U,nx_avg,nz_avg,mode=1)
    efx=filter_2D(efx,nx_avg,nz_avg,mode=1)
    efz=filter_2D(efz,nx_avg,nz_avg,mode=1)
    ep=filter_2D(ep,nx_avg,nz_avg,mode=1)
    # ep=average_horizontal(ep,nx_avg)
    # ep=average_vertical(ep,nz_avg)

    cmap = plt.get_cmap('RdBu_r')
    
    ## MFx ##
    clev = eval(SETTINGS['CLEV_MFX'])
    clev_l = eval(SETTINGS['CLEV_MFX'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    pcMesh0 = ax0.pcolormesh(ds.xcr[y], z, 1000*mfx,
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh0, ax=ax0, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('MF$_x$ / mPa')
    if SETTINGS['CLEV_MFX'] == 'CLEV_01' or SETTINGS['CLEV_MFX'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
    ## EFx ##
    clev = eval(SETTINGS['CLEV_EFX'])
    clev_l = eval(SETTINGS['CLEV_EFX'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    pcMesh1 = ax1.pcolormesh(ds.xcr[y], z, efx,
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh1, ax=ax1, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('EF$_x$ / W m$^{-2}$')
    if SETTINGS['CLEV_EFX'] == 'CLEV_01' or SETTINGS['CLEV_EFX'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    
    ## Ep ##
    clev = eval(SETTINGS['CLEV_EP'])
    clev_l = eval(SETTINGS['CLEV_EP'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    pcMesh2 = ax2.pcolormesh(ds.xcr[y], z, ep,
                            cmap=cmap, norm=norm, shading='nearest')
    # pcMesh2 = ax2.pcolormesh(ds.xcr[y], ds.zcr[:,y,:], -ds.mfx_u[t,:,y,:],
    #                         cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh2, ax=ax2, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("E$_P$ / J kg$^{-1}$")
    if SETTINGS['CLEV_EP'] == 'CLEV_01' or SETTINGS['CLEV_EP'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
    ## EFz (-MF*U)##
    clev = eval(SETTINGS['CLEV_EFZ'])
    clev_l = eval(SETTINGS['CLEV_EFZ'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)
    pcMesh3 = ax3.pcolormesh(ds.xcr[y], z, efz,
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh3, ax=ax3, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('EF$_z$ / W m$^{-2}$')
    if SETTINGS['CLEV_EFZ'] == 'CLEV_01' or SETTINGS['CLEV_EFZ'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
    ## ISENTROPES ##
    ax0 = format_xz_plot(ds, SETTINGS, ax0, t, y=y)
    ax1 = format_xz_plot(ds, SETTINGS, ax1, t, y=y)
    ax2 = format_xz_plot(ds, SETTINGS, ax2, t, y=y)
    ax3 = format_xz_plot(ds, SETTINGS, ax3, t, y=y)
    

    xrange = eval(SETTINGS['XRANGE'])
    if xrange != 'NONE':    
        ax0.set_xlim(xrange)
    
    yrange = eval(SETTINGS['ZRANGE'])
    if yrange != 'NONE':    
        ax0.set_ylim(yrange)

    ax0.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax1.set_xlabel('streamwise x / km') # change to longitudes, latitude
    ax0.xaxis.set_label_position('top') 
    ax1.xaxis.set_label_position('top') 
    ax0.tick_params(labelbottom=False,labeltop=True)
    ax1.tick_params(labelbottom=False,labeltop=True)
    ax2.tick_params(labelbottom=False,labeltop=False)
    ax3.tick_params(labelbottom=False,labeltop=False)
    ax0.set_ylabel('altitude / km')
    ax2.set_ylabel('altitude / km')
    
    center_str = 'T: ' + str(int(ds.dt00*t*ds.nplot/3600)) +'h, y: ' + str(int(ds.dy00*y/1000)) + 'km' 
    fig.text(CENTER_STR_X,CENTER_STR_Y,center_str,horizontalalignment='center',
            verticalalignment='center', ma='center', fontsize=14)

    # Save fig
    fig.tight_layout()
    fig_name = 'fluxes_y' + str(y) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'
    

def plot_fluxes_horizontalAvg(ds, SETTINGS, y=0):
    "horizontal averages of fluxes without damping areas (sponge layers)"
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharey=True, figsize=FIGSIZE)

    z = np.linspace(0,ds['zcr'][-1,-1,y,0].values,int(ds.nz)) # ds['dx']=200
    
    for t in range(0,np.shape(ds['th'])[0],2):
        # --------- Calculate fluxes -------------------- # 
        mfx = ds['rh0'][t,:,y,:] * ds['w'][t,:,y,:] * (ds['u'][t,:,y,:]-ds['ue'][t,:,y,:]) # h9=h3*h6
        mfy = ds['rh0'][t,:,y,:] * ds['w'][t,:,y,:] * (ds['v'][t,:,y,:]-ds['ve'][t,:,y,:])
        MF_U = mfx * ds['u'][t,:,y,:] + mfy * ds['v'][t,:,y,:] # = -EFz
        
        efx = (ds['u'][t,:,y,:]-ds['ue'][t,:,y,:]) * ds['pprime'][t,:,y,:] # h14, EFx1
        efz = ds['w'][t,:,y,:] * ds['pprime'][t,:,y,:] # h13, EFz1

        ## TEMPERATURE CALCULATION##                             
        thloc = ds['the'][t,:,y,:] + ds['th'][t,:,y,:] # Theta
        # ds['tloc'] = ds['thloc']*(ds['ploc']/ds['pref00'])**ds.cap
        tte = ds['the'][t,:,y,:]*(ds['pr0'][t,:,y,:]/ds['pref00'])**ds.cap # T_env
        tloc = thloc*(ds['pr0'][t,:,y,:]/ds['pref00'])**ds.cap # T
        tprime = tloc-tte

        ep = 1/2*(ds.grav/ds.bv)**2 * (tprime/tte)**2 # * ds['rh0'].expand_dims({'t':ds.t}) potential energy density
        # --------- Calculate fluxes -------------------- #

        ## INTERPOLATE TO REGULAR VERTICAL GRID ##
        mfx = interp_elev_to_z(mfx,ds.zcr[t,:,y,:],z)
        MF_U = interp_elev_to_z(MF_U,ds.zcr[t,:,y,:],z)
        efx = interp_elev_to_z(efx,ds.zcr[t,:,y,:],z)
        efz = interp_elev_to_z(efz,ds.zcr[t,:,y,:],z)
        ep = interp_elev_to_z(ep,ds.zcr[t,:,y,:],z)

        ## AVERAGE/FILTER VERTICAL AND HORIZONTAL OVER ONE WAVELENGTH
        nx_avg = ds.nx_avg
        nz_avg = ds.nz_avg

        mfx = filter_1Dz(mfx,nz_avg,mode=SETTINGS['FILTER_MODE'])
        MF_U = filter_1Dz(MF_U,nz_avg,mode=SETTINGS['FILTER_MODE'])
        efx = filter_1Dz(efx,nz_avg,mode=SETTINGS['FILTER_MODE'])
        efz = filter_1Dz(efz,nz_avg,mode=SETTINGS['FILTER_MODE'])
        ep = filter_1Dz(ep,nz_avg,mode=SETTINGS['FILTER_MODE'])

        ## HORIZONTAL AVERAGE ##
        # Do not include sponge layers for horizontal average!
        nx=np.shape(mfx)[1]
        if ds.irelx:
            n_sponge=int(ds.dxabL/ds.dx00)
        else:
            n_sponge=0
        mfx_m = mfx[:,n_sponge:nx-n_sponge].mean(axis=1)
        mfx_m = mfx.mean(axis=1)
        MF_U_m = MF_U.mean(axis=1)
        efx_m = efx.mean(axis=1)
        efz_m = efz.mean(axis=1)
        ep_m = ep.mean(axis=1)

        if (t==np.shape(ds['th'])[0]-1):
            linePlot0 = ax0.plot(1000*mfx_m[:],z, color='black', lw=2)
            linePlot1 = ax1.plot(efx_m[:],z, color='indianred', lw=2)
            linePlot2 = ax2.plot(ep_m[:],z, color='indianred', lw=2)
            linePlot3 = ax3.plot(efz_m[:],z, color='indianred', lw=2)
            linePlot3 = ax3.plot(MF_U_m[:],z, color='blue', lw=2)
            
        else:
            linePlot0 = ax0.plot(1000*mfx_m[:],z, color='black')
            linePlot1 = ax1.plot(efx_m[:],z, color='indianred')
            linePlot2 = ax2.plot(ep_m[:],z, color='indianred')
            linePlot3 = ax3.plot(efz_m[:],z, color='indianred')
            linePlot3 = ax3.plot(MF_U_m[:],z, color='blue')
            
    # Labels and Limits
    ax0.set_ylabel('altitude / km')
    ax2.set_ylabel('altitude / km')
    ax0.set_xlabel('MF$_x$ / mPa')
    ax1.set_xlabel('EF$_x$ / W m$^{-2}$')
    ax2.set_xlabel('E$_P$ / J kg$^{-1}$')
    ax3.set_xlabel('EF$_z$ = $\\bf{-MF}$ $\cdot$ $\\bf{U}$ / W m$^{-2}$')
    
    ## X LIMITS ##
    if SETTINGS['CASE']=='inertia-gravity':
        ax0.set_xlim([-1,1])
        ax1.set_xlim([-0.1,0.1])
        ax2.set_xlim([0,0.5])
        ax3.set_xlim([-0.01,0.01])
#    else:
#        ax0.set_xlim([-5,1])
#        ax1.set_xlim([-0.3,0.3])
#        ax2.set_xlim([0,2])
#        ax3.set_xlim([-0.05,0.05])
        
    
    # ax0.xaxis.set_major_locator(MultipleLocator(2))
    ax0.yaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_major_locator(MultipleLocator(10))
    ax0.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    
    # Grey origin lines
    ax0.axhline(y=0, color='grey')
    ax0.axvline(x=0, color='grey')
    ax1.axhline(y=0, color='grey')
    ax1.axvline(x=0, color='grey')
    ax2.axhline(y=0, color='grey')
    ax2.axvline(x=0, color='grey')
    ax3.axhline(y=0, color='grey')
    ax3.axvline(x=0, color='grey')
    
    # Save fig
    fig.tight_layout()
    fig_name = 'fluxes_horizontalAvg_y' + str(y) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'
    

def plot_vertical_profiles(ds, SETTINGS, y=0):
    "vertical profiles of u, l^2, N, Ri"
    fig, (ax0,ax1,ax2,ax3) = plt.subplots(1, 4, sharey=True, figsize=FIGSIZE)

    x = int(ds.nx/2)
    # z = np.linspace(0,ds['zcr'][-1,-1,y,0].values,int(ds.nz)) # ds['dx']=200

    ## ENV PROFILE (u_e) ##
    t=0
    ax0.plot(ds['ue'][t,:,y,x], ds.zcr[t,:,y,x], lw=2, color='goldenrod', label='$u_e$')
    
    dthdz = dthdz_prof(ds,t=t,y=y,x=x)
    dudz, dudz2 = dudz_prof(ds,t=t,y=y,x=x)
    thloc = ds['the'][t,:,y,x] + ds['th'][t,:,y,x] # Theta
    N_prof = np.sqrt(ds.g/thloc*dthdz)
    Ri_prof = N_prof**2 / dudz**2 # eventually gradient of u and v
    l2_prof = N_prof**2 / ds['ue'][t,:,y,x]**2 - 1/ds['ue'][t,:,y,x] * dudz2
    l2_prof_simp = N_prof**2 / ds['ue'][t,:,y,x]**2
    ax1.plot(l2_prof, ds.zcr[t,:,y,x], lw=2, color='goldenrod', label='$l^2$ t0')
    ax1.plot(l2_prof_simp, ds.zcr[t,:,y,x], lw=2, color='indianred', label='$l^2$ (no curv) t0')
    ax2.plot(N_prof, ds.zcr[t,:,y,x], lw=2, color='goldenrod', label='$N$ t0')
    ax3.plot(Ri_prof, ds.zcr[t,:,y,x], lw=2, color='goldenrod', label='$Ri$ t0')
    

    ## HORIZONTAL AVERAGE ##
    # Do not include sponge layers for horizontal average if existing
    for t in range(0,np.shape(ds['u'])[0],2):
        n_sponge=0
        u_avg = ds['u'][t,:,y,n_sponge:ds.nx-n_sponge].mean(axis=1) # maybe axis 3

        # N, Ri, l
        dthdz = dthdz_prof(ds,t=t,y=y,x=x)
        dudz, dudz2 = dudz_prof(ds,t=t,y=y,x=x)
        dvdz = dvdz_prof(ds,t=t,y=y,x=x)
        thloc = ds['the'][t,:,y,x] + ds['th'][t,:,y,x] # Theta
        N_prof = np.sqrt(ds.g/thloc*dthdz)
        Ri_prof = N_prof**2 / (dudz**2 + dvdz**2)
        l2_prof = N_prof**2 / ds['u'][t,:,y,x]**2 - 1/ds['u'][t,:,y,x] * dudz2

        if (t==np.shape(ds['u'])[0]-1):
            ax0.plot(u_avg, ds.zcr[t,:,y,x], lw=2, color='black', alpha=0.6, label='$\hat{u}$')
            ax1.plot(l2_prof, ds.zcr[t,:,y,x], lw=2, color='black', alpha=0.6, label='$l^2$ t6')
            ax2.plot(N_prof, ds.zcr[t,:,y,x], lw=2, color='black', alpha=0.6, label='$N$ t6') 
            ax3.plot(Ri_prof, ds.zcr[t,:,y,x], lw=2, color='black', alpha=0.6, label='$Ri$ t6')         
        else:
            ax0.plot(u_avg, ds.zcr[t,:,y,x], color='black',alpha=0.6)
            ax1.plot(l2_prof, ds.zcr[t,:,y,x], color='black',alpha=0.6)
            ax2.plot(N_prof, ds.zcr[t,:,y,x], color='black',alpha=0.6) 
            ax3.plot(Ri_prof, ds.zcr[t,:,y,x], color='black',alpha=0.6) 
           
    
    # Labels and Limits
    ax0.set_ylabel('altitude / km')
    ax0.set_xlabel('$\hat{u}$ / m s$^{-1}$')
    ax1.set_xlabel('$l^2$ / m$^{-2}$')
    ax2.set_xlabel('$N$ / s$^{-1}$')
    ax3.set_xlabel('$Ri$ / -')
    
    
    ## LIMITS AND LEGEND ##
    # ax0.set_xlim([-5,1])
    if SETTINGS['STRATOS']:
        ax2.set_xlim([0.012,0.028])
    else:
        ax2.set_xlim([0.02,0.018])
    ax3.set_xlim([-10,120])
    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax3.legend()

    # ax0.xaxis.set_major_locator(MultipleLocator(2))
    ax0.yaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_major_locator(MultipleLocator(10))
    ax0.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    
    # Grey origin lines
    # ax0.axhline(y=0, color='grey')
    # ax0.axvline(x=0, color='grey')
    # ax1.axhline(y=0, color='grey')
    # ax1.axvline(x=0, color='grey')
    # ax2.axhline(y=0, color='grey')
    ax3.axvline(x=0.25, color='grey')
    ax3.axvline(x=2, color='grey')
    # ax3.axhline(y=0, color='grey')
    # ax3.axvline(x=0, color='grey')
    
    # Save fig
    fig.tight_layout()
    fig_name = 'vertical_profiles_y' + str(y) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'


def plot_eliassenPalm_theorem(ds, SETTINGS):
    "Visualizes Eliassen–Palm theorem for last time step and y"
    fig, ax0 = plt.subplots(figsize=FIGSIZE_QQ_PLOT)
    y=0

    # array.stack(z=("x", "y"))
    for t in range(0,np.shape(ds['efz'])[0]):
        # --------- Calculate fluxes -------------------- # 
        mfx = ds['rh0'][t,:,y,:] * ds['w'][t,:,y,:] * (ds['u'][t,:,y,:]-ds['ue'][t,:,y,:]) # h9=h3*h6
        mfy = ds['rh0'][t:,y,:,] * ds['w'][t,:,y,:] * (ds['v'][t,:,y,:]-ds['ve'][t,:,y,:])
        MF_U = mfx * ds['u'][t:,y,:,] + mfy * ds['v'][t:,y,:,] # = -EFz
        
        efz = ds['w'][t,:,y,:] * ds['pprime'][t,:,y,:] # h13, EFz1
        # --------- Calculate fluxes -------------------- #
        
        linePlot0 = ax0.plot(efz, -MF_U, color='black', linestyle='dotted')
        # linePlot0 = ax0.plot(ds['efz_noFilter'][t,:,y,:], -ds['mfx_u_noFilter'][t,:,y,:], color='black', linestyle='dotted')
    # sm.qqplot(data_points, axes=ax0, line ='45')
    # qqplot_2samples(ds['mfx_u'][12,:,:], ds['efz1'][12,:,:,:], xlabel=None, ylabel=None, line=None, ax=ax0)    
    
    ax0.axhline(y=0, color='grey')
    ax0.axvline(x=0, color='grey')
    
    line_45 = np.linspace(-10,10,50)
    ax0.plot(line_45,line_45, color='grey', linestyle='--')
    
    ax0.set_xlim([-4,4])
    ax0.set_ylim([-4,4])
    ax0.set_xlabel("EF$_z$ / W m$^{-2}$")
    ax0.set_ylabel('$\\bf{-MF}$ $\cdot$ $\\bf{U}$ / W m$^{-2}$')
    ax0.xaxis.set_minor_locator(AutoMinorLocator())
    ax0.xaxis.set_major_locator(MultipleLocator(1))
    ax0.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Save fig
    fig.tight_layout()
    fig_name = 'eliassenPalm_theorem_qq_plot' + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'


# ---- LIDAR STUFF ---------------------- #
def plot_lidar_location_primes(ds_lid, ds, SETTINGS, theta=0):
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=FIGSIZE_2)
    
    cmap = plt.get_cmap('RdBu_r')

    ## U ##
    clev = eval(SETTINGS['CLEV_U'])
    clev_l = eval(SETTINGS['CLEV_U'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True)  
    pcMesh0 = ax0.pcolormesh(ds_lid.time, ds_lid.zcr, ds_lid.uprime,
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh0, ax=ax0, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.set_label("$u'=u-u_e$ / m s$^{-1}$")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_U'] == 'CLEV_01' or SETTINGS['CLEV_U'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    
    ## W ##
    clev = eval(SETTINGS['CLEV_W'])
    clev_l = eval(SETTINGS['CLEV_W'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    pcMesh1 = ax1.pcolormesh(ds_lid.time, ds_lid.zcr, ds_lid.w,
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh1, ax=ax1, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.set_label("$w'$ (vertical velocity) / m s$^{-1}$")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_W'] == 'CLEV_01' or SETTINGS['CLEV_W'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
    ## T & THETA ##
    clev = eval(SETTINGS['CLEV_TH'])
    clev_l = eval(SETTINGS['CLEV_TH'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    if theta: 
        pcMesh2 = ax2.pcolormesh(ds_lid.time, ds_lid.zcr, ds_lid.th,
                                 cmap=cmap, norm=norm, shading='nearest')
        cbar = fig.colorbar(pcMesh2, ax=ax2, orientation='horizontal', ticks=clev_l, pad=0.07)
        cbar.set_label("$\Theta'$ / K")
    else:
        pcMesh2 = ax2.pcolormesh(ds_lid.time, ds_lid.zcr, ds_lid.tprime,
                                 cmap=cmap, norm=norm, shading='nearest')
        cbar = fig.colorbar(pcMesh2, ax=ax2, orientation='horizontal', ticks=clev_l, pad=0.07)
        cbar.set_label("$T'$ / K")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_TH'] == 'CLEV_01' or SETTINGS['CLEV_TH'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
           
    ## PRESSURE ##
    clev = eval(SETTINGS['CLEV_P'])
    clev_l = eval(SETTINGS['CLEV_P'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    pcMesh3 = ax3.pcolormesh(ds_lid.time, ds_lid.zcr, ds_lid.p,
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh3, ax=ax3, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.set_label("$p'_z$ / Pa")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_P'] == 'CLEV_01' or SETTINGS['CLEV_P'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
    ## ISENTROPES ##
    ax0 = format_tz_plot(ds_lid, ds, SETTINGS, ax0)
    ax1 = format_tz_plot(ds_lid, ds, SETTINGS, ax1)
    ax2 = format_tz_plot(ds_lid, ds, SETTINGS, ax2)
    ax3 = format_tz_plot(ds_lid, ds, SETTINGS, ax3)
    
    # ax0.set_xlim([0,48])

    ## LABELS ##
    ax0.set_xlabel('time / h') # change to longitudes, latitude
    ax1.set_xlabel('time / h') # change to longitudes, latitude
    ax0.xaxis.set_label_position('top') 
    ax1.xaxis.set_label_position('top') 
    ax0.tick_params(labelbottom=False,labeltop=True)
    ax1.tick_params(labelbottom=False,labeltop=True)
    ax2.tick_params(labelbottom=False,labeltop=False)
    ax3.tick_params(labelbottom=False,labeltop=False)
    ax0.set_ylabel('altitude / km')
    ax2.set_ylabel('altitude / km')
    
    # lid_00240_00102
    xloc = int(str(ds_lid['location'].values)[6:9]) * ds.dx00/1000
    yloc = (int(str(ds_lid['location'].values)[-3:]) - int(ds.ny/2)) * ds.dy00/1000
    if ds.ny==1:
        center_str = 'x: ' + str(xloc) + 'km, y: 0km' 
    else:
        center_str = 'x: ' + str(xloc) + 'km, y: ' + str(yloc) + 'km' 
    fig.text(CENTER_STR_X,CENTER_STR_Y,center_str,horizontalalignment='center',
            verticalalignment='center', ma='center', fontsize=12, weight='bold')

    # SAVE FIG ##
    fig.tight_layout()
    lid_dir = SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/Lidar'
    if not os.path.exists(lid_dir):
        os.makedirs(lid_dir)
    fig_name = str(ds_lid['location'].values) + '.png'
    fig.savefig(lid_dir + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'
    

def plot_lidar_location_fluxes(ds_lid, ds, SETTINGS):
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True, sharey=True, figsize=FIGSIZE_2)
    
    cmap = plt.get_cmap('RdBu_r')

    ## MFx ##
    clev = eval(SETTINGS['CLEV_MFX'])
    clev_l = eval(SETTINGS['CLEV_MFX'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    pcMesh0 = ax0.pcolormesh(ds_lid.time, ds_lid.zcr, 1000*ds_lid.mfx,
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh0, ax=ax0, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.set_label("MF$_x$ / mPa")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_MFX'] == 'CLEV_01' or SETTINGS['CLEV_MFX'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
    ## EFx ##
    clev = eval(SETTINGS['CLEV_EFX'])
    clev_l = eval(SETTINGS['CLEV_EFX'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    pcMesh1 = ax1.pcolormesh(ds_lid.time, ds_lid.zcr, ds_lid.efx,
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh1, ax=ax1, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.set_label("EF$_x$ / W m$^{-2}$")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_EFX'] == 'CLEV_01' or SETTINGS['CLEV_EFX'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    
    ## MF U ##
    clev = eval(SETTINGS['CLEV_EFZ'])
    clev_l = eval(SETTINGS['CLEV_EFZ'] + '_LABELS')
    norm = BoundaryNorm(boundaries=clev , ncolors=cmap.N, clip=True) 
    pcMesh2 = ax2.pcolormesh(ds_lid.time, ds_lid.zcr, -ds_lid.mfx_u,
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh2, ax=ax2, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.set_label("$\\bf{-MF}$ $\cdot$ $\\bf{U}$ / W m$^{-2}$")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_EFX'] == 'CLEV_01' or SETTINGS['CLEV_EFX'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    
    ## EFz ##
    pcMesh3 = ax3.pcolormesh(ds_lid.time, ds_lid.zcr, ds_lid.efz,
                            cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(pcMesh3, ax=ax3, orientation='horizontal', ticks=clev_l, pad=0.07)
    cbar.set_label("EF$_z$ / W m$^{-2}$")
    cbar.ax.tick_params(labelsize=9)
    if SETTINGS['CLEV_EFX'] == 'CLEV_01' or SETTINGS['CLEV_EFX'] == 'CLEV_05':
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    
    ## ISENTROPES ##
    ax0 = format_tz_plot(ds_lid, ds, SETTINGS, ax0)
    ax1 = format_tz_plot(ds_lid, ds, SETTINGS, ax1)
    ax2 = format_tz_plot(ds_lid, ds, SETTINGS, ax2)
    ax3 = format_tz_plot(ds_lid, ds, SETTINGS, ax3)
    
    ## LABELS ##
    ax0.set_xlabel('time / h') # change to longitudes, latitude
    ax1.set_xlabel('time / h') # change to longitudes, latitude
    ax0.xaxis.set_label_position('top') 
    ax1.xaxis.set_label_position('top') 
    ax0.tick_params(labelbottom=False,labeltop=True)
    ax1.tick_params(labelbottom=False,labeltop=True)
    ax2.tick_params(labelbottom=False,labeltop=False)
    ax3.tick_params(labelbottom=False,labeltop=False)
    ax0.set_ylabel('altitude / km')
    ax2.set_ylabel('altitude / km')
    
    # SAVE FIG ##
    fig.tight_layout()
    lid_dir = SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/Lidar'
    if not os.path.exists(lid_dir):
        os.makedirs(lid_dir)
    fig_name = str(ds_lid['location'].values) + '_fluxes' + '.png'
    fig.savefig(lid_dir + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'

    
def process_ds_lid(ds_lid, ds):
    # ds_lid = ds_lid.assign_coords({'tt':(ds_lid.t * ds.nlid * ds.dt00/3600)})
    ds_lid['time'] = ds_lid.t * ds.nlid * ds.dt00/3600
    ds_lid['time'] = ds_lid['time'].expand_dims({'z':ds_lid.z}, axis=1)
    ds_lid['zcr'] = ds_lid['zcr']/1000

    ds_lid['uprime'] = ds_lid['u'] - ds_lid['ue']
    ds_lid['thloc'] = ds_lid['th'] + ds_lid['the']
    ds_lid['ploc'] = ds_lid['p'] + ds_lid['pr0'] # [t,:,y,x].values
    ds_lid['tloc'] = ds_lid['thloc']*(ds_lid['ploc']/ds['pref00'])**ds.cap
    ds_lid['tte'] = ds_lid['the']*(ds_lid['pr0']/ds['pref00'])**ds.cap # T_env
    ds_lid['tprime'] = ds_lid['tloc'] - ds_lid['tte']
    
    # Forces
    ds_lid['mfx'] =  ds_lid['uprime'] * ds_lid['w'] * ds_lid['rh0'] # [t,:,y,x].values # h9=h3*h6
    ds_lid['efx'] = ds_lid['uprime'] * ds_lid['p'] # (h12 instead of p)  h14, EFx1
    ds_lid['efz'] = ds_lid['w'] * ds_lid['p'] # (h12 instead of p) h13, EFz1
    ds_lid['mfx_u'] = ds_lid['mfx'] * ds_lid['u']
    
    return ds_lid


def plot_surf(ds, SETTINGS, y=0):
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharex=True, figsize=FIGSIZE)
    z=0
    
    for t in range(0,len(ds.w[:,0,0,0]),2):
        if (t == len(ds.w[:,0,0,0])-1):
            linePlot0 = ax0.plot(ds.xcr[y], ds.u[t,z,y,:]-ds.ue[t,z,y,:], color='black', lw=2)
            linePlot1 = ax1.plot(ds.xcr[y], ds.w[t,z,y,:], color='black', lw=2)
            linePlot2 = ax2.plot(ds.xcr[y], ds.pprime[t,z,y,:], color='black', lw=2) # pprime
            linePlot3 = ax3.plot(ds.xcr[y], ds.pprime[t,z,y,:] * ds.dzdx_surf[y,:], color='black', lw=2) # pprime dz_surf/dx
        else:
            linePlot0 = ax0.plot(ds.xcr[y], ds.u[t,z,y,:]-ds.ue[t,z,y,:], color='darkgray', lw=1)
            linePlot1 = ax1.plot(ds.xcr[y], ds.w[t,z,y,:], color='darkgray', lw=1)
            linePlot2 = ax2.plot(ds.xcr[y], ds.pprime[t,z,y,:], color='darkgray', lw=1) # pprime
            linePlot3 = ax3.plot(ds.xcr[y], ds.pprime[t,z,y,:] * ds.dzdx_surf[y,:], color='darkgray', lw=1) # pprime dz_surf/dx

    ## PLOT LINEAR THEORY ##
    drag_linear_theory(ds)
    # drag_f : streamwise Coriolis force acting between undisturbed streamline and vertically displaced streamline
    # drag_vertMom: linear flux of vertical momentum
    # drag_angMom: vertical flux of angular momentum
    
    # Labels and Limits
    ax2.set_xlabel('streamwise x / km')
    ax3.set_xlabel('streamwise x / km')
    # ax0.set_xlabel('streamwise x / km')
    # ax1.set_xlabel('streamwise x / km')
    
    # xrange = eval(SETTINGS['XRANGE_ZOOM'])
    # ax0.set_xlim(xrange)
    # ax0.set_ylim([-1,1])
    # ax1.set_ylim([-1,1])
    # ax2.set_ylim([-10,10])
    # ax3.set_ylim([-10,10])
    
    ax0.set_ylabel("$u'_{SURF}$ / Pa", labelpad=-5)
    ax1.set_ylabel("$w'_{SURF}$ / Pa", labelpad=-5)
    ax2.set_ylabel("$p'_{SURF}$ / Pa", labelpad=-5)
    ax3.set_ylabel("$p' dz_{SURF}$/dx / Pa", labelpad=-5)
    
    # ax0.xaxis.set_major_locator(MultipleLocator(2))
    ax0.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_major_locator(MultipleLocator(10))
    ax0.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Grey origin lines
    ax0.axhline(y=0, color='grey')
    ax0.axvline(x=0, color='grey')
    ax1.axhline(y=0, color='grey')
    ax1.axvline(x=0, color='grey')
    ax2.axhline(y=0, color='grey')
    ax2.axvline(x=0, color='grey')
    ax3.axhline(y=0, color='grey')
    ax3.axvline(x=0, color='grey')
    
    # Save fig
    fig.tight_layout()
    fig_name = 'surf_y' + str(y) + '_t' +str(t) + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'


#### DRAG FROM LINEAR THEORY OF SMALL PERTURBATIONS ##### 
def drag_linear_theory(ds):
    # fig, (ax0,ax1) = plt.subplots(2,1,figsize=(12,9), sharex=True, sharey=True)
    fig, ax0 = plt.subplots(1,1,figsize=(7,5))

    # - Calculate drag for multiple points in time - # 
    # start = int(ds.nx/2-50)
    # end = int(ds.nx/2+50)
    if ds.irelx:
        n_sponge=int(ds.dxabL/ds.dx00)
    else:
        n_sponge=0

    x_regime = ds.xml * ds.bv / ds['ue'][0,0,0,0]
    x_regime = x_regime.values
    linear_drag_array = np.loadtxt("drag_gill_page279.txt",usecols=(0, 1), skiprows=0)
    ax0.plot(linear_drag_array[:,0],linear_drag_array[:,1], color='black', label='Linear analysis')

    y=0
    z=0
    tdrag=[9,10,11,12]
    setlabel=0
    for t in tdrag:
        # - Drag form pressure perturbation - #
        # drag = integrate.trapezoid(ds.pprime[t,z,y,:] * ds.dzdx_surf[y,:], 1000.*ds.xcr[y,:])
        drag = integrate.trapezoid(ds.pprime[t,z,y,n_sponge:ds.nx-n_sponge] * ds.dzdx_surf[y,n_sponge:ds.nx-n_sponge], 1000*ds.xcr[y,n_sponge:ds.nx-n_sponge]) # 0.3576
        # drag = integrate.trapezoid(ds.pprime[12,z,y,start:end] * ds.dzdx_surf[y,start:end], 1000*ds.xcr[y,start:end]) # 0.3576
        # drag = integrate.simps(ds.pprime[12,z,y,:] * ds.dzdx_surf[y,:], 1000*ds.xcr[y,:])

        # - Drag from vertical flux of angular momentum (vertical momentum flux + streamwise Coriolis force) - # 
        drag_vertMom = -ds['rh0'][0,0,0,0]*integrate.trapezoid((ds.u[t,z,y,:]-ds.ue[t,z,y,:])*ds.w[t,z,y,:], 1000.*ds.xcr[y,:])
        drag_vertMom_v = -ds['rh0'][0,0,0,0]*integrate.trapezoid((ds.v[t,z,y,:]-ds.ve[t,z,y,:])*ds.w[t,z,y,:], 1000.*ds.xcr[y,:])
        drag_f = ds.fcr0*np.sin(-ds.ang*np.pi/180) * ds['rh0'][0,0,0,0]*integrate.trapezoid((ds.v[t,z,y,:]-ds.ve[t,z,y,:])*1000.*ds.zcr[t,z,y,:], 1000.*ds.xcr[y,:])
        # f=1.031
    
        # - Normalized drag - #
        # normalized_drag = drag / (0.25*np.pi*ds['rh0'][0,0,0,0] * ds.bv * ds['ue'][0,0,0,0] * ds.amp**2)
        drag = drag / (ds['rh0'][0,0,0,0] * ds.bv * ds['ue'][0,0,0,0] * ds.amp**2)
        drag_vertMom = drag_vertMom / (ds['rh0'][0,0,0,0] * ds.bv * ds['ue'][0,0,0,0] * ds.amp**2)
        drag_vertMom_v = drag_vertMom_v / (ds['rh0'][0,0,0,0] * ds.bv * ds['ue'][0,0,0,0] * ds.amp**2)
        drag_f = drag_f / (ds['rh0'][0,0,0,0] * ds.bv * ds['ue'][0,0,0,0] * ds.amp**2)
        drag_angMom = drag_vertMom + drag_f 

        # - Plot - #
        tstr = str(t*12) + 'h' # h
        if (setlabel==0):
            ax0.scatter(x_regime, drag, marker='D', s=50, color='indianred', label="p'deta_dx")
            ax0.scatter(x_regime, drag_angMom, marker='D', s=50, color='y', label="u'w'+fv'eta'")
        else:
            ax0.scatter(x_regime, drag, marker='D', s=50, color='indianred')
            ax0.scatter(x_regime, drag_angMom, marker='D', s=50, color='y')
            

        setlabel+=1
        seperation = 10
        ax0.annotate(tstr, (x_regime+seperation, drag), color='indianred',transform=ax0.transAxes)
        ax0.annotate(tstr, (x_regime+seperation, drag_angMom), color='y', transform=ax0.transAxes) # (Smith79)

    # - Eventually text - # 
    # ax3.text(0.6, 0.6, 'drag$_{normalized}$: ' + str(normalized_drag.values.round(decimals=4)), bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 10}, transform=ax3.transAxes)
    # ax3.text(0.6, 0.9, 'drag$_{normalized}$: ' + str(ds['drag'].values.round(decimals=4)), transform=ax3.transAxes)
    # ax3.text(0.6, 0.8, 'drag$_{angMom}$: ' + str(ds['drag_angMom'].values.round(decimals=4)), transform=ax3.transAxes)
    # ax3.text(0.6, 0.7, 'drag$_{vertMom}$: ' + str(ds['drag_vertMom'].values.round(decimals=4)), transform=ax3.transAxes)
    # ax3.text(0.6, 0.6, 'drag$_f$: ' + str(ds['drag_f'].values.round(decimals=4)), transform=ax3.transAxes)
    # ax3.text(0.6, 0.5, 'drag$_v$: ' + str(ds['drag_vertMom_v'].values.round(decimals=4)), transform=ax3.transAxes)
    
    # - Formatting - #
    ax0.set_xscale('log')
    ax0.legend()
    ax0.set_ylabel(r'F ($\rho_0$ N U h$_m^2$)$^{-1}$')
    ax0.set_xlabel('L N U$^{-1}$')
    ax0.tick_params(labeltop=True)
    ax0.yaxis.set_minor_locator(AutoMinorLocator())

    # SAVE FIG ##
    fig.tight_layout()
    fig_name = 'drag_linearTheory' + '.png'
    fig.savefig(SETTINGS['FILE_LOCATION'] + '/' + SETTINGS['PLOT_FOLDER'] + '/' + fig_name, facecolor='w', edgecolor='w',
                format='png', dpi=150) # orientation='portrait'


#### LOAD AND MAIN #### 
def load_settings(SETTINGS_FILE):
    # global SETTINGS
    SETTINGS = {}
    with open(SETTINGS_FILE, 'r') as file:
        for line in file:
            try:
                line = line.strip()
                (key, val) = line.split(": ")
                SETTINGS[key] = val
            except:
                print('The following line could not be executed: ' + line)
                print('Variable might be missing!')
    return SETTINGS


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 0:    
        SETTINGS_FILE = args[0]
    else:
        SETTINGS_FILE = 'settings.txt'
    SETTINGS = load_settings(SETTINGS_FILE)
    if len(args) > 1:
        SETTINGS['FILE_LOCATION'] = args[1]
    vis_eulag(SETTINGS)