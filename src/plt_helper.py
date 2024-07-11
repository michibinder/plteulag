################################################################################
# Copyright 2023 German Aerospace Center                                       #
################################################################################
# This is free software you can redistribute/modify under the terms of the     #
# GNU Lesser General Public License 3 or later: http://www.gnu.org/licenses    #
################################################################################

import os
import matplotlib.dates as mdates
import time
from datetime import datetime
import numpy as np
import imageio.v2 as imageio
# import logging
# import warnings
# warnings.simplefilter("ignore", RuntimeWarning)
# warnings.filterwarnings('ignore', category=UserWarning, module='imageio_ffmpeg')
# logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)

"""Config"""
pbar_interval = 5 # %

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


def create_animation(image_folder):
    """Create animation (mp4) from pngs"""

    filenames        = sorted(os.listdir(image_folder))
    fps              = 10
    macro_block_size = 16

    # Increase the probesize to give FFmpeg more data to estimate the rate
    # writer_options = {'ffmpeg_params': ['-probesize', '100M']}  # Increase probesize to 5MB
    # writer_options = {'ffmpeg_params': ['-probesize', '5000000', '-analyzeduration', '5000000']}

    with imageio.get_writer(os.path.join(image_folder,"animation.mp4"), fps=fps) as writer: # duration=1000*1/fps
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