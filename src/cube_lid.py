#!/usr/bin/env python3

"""Build the lambda2 multiview + virtual-lidar MP4 for one PMAP cube output.

Usage:
    python3 cube_lid.py darwin_240718_400m_schu
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cube_lid")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyvista as pv
import xarray as xr

if os.path.exists('latex_default.mplstyle'):
    plt.style.use('latex_default.mplstyle')

from cmcrameri import cm
import plt_helper, cmaps

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

pv.global_theme.allow_empty_mesh = True
pv.OFF_SCREEN = True


DATA_ROOT = Path("/scratch/b/b309199")
REFERENCE_PRESSURE_PA = 100000.0
RD = 287.05
CP = 1004.0
KAPPA = RD / CP

TIME_NAME = "time"
Z_NAME = "z"
Y_NAME = "y"
X_NAME = "x"

U_NAME = "uvelx"
V_NAME = "uvely"
W_NAME = "uvelz"
DENSITY_NAME = "density"
THETA_NAME = "theta_total"
EXNER_NAME = "exner_total"

TIME_INDEX = -1
STRIDE = (1, 1, 1)

# Optional cube limits in coordinate units of cube.nc, typically meters.
# Set to None to use the full domain in that direction.

# X_LIMITS = (115000.0,130000.0)  # e.g. (110000.0, 145000.0)
# Y_LIMITS = (-93000.0, -78000.0)  # e.g. (-95000.0, -60000.0)
# Z_LIMITS = (55000.0, 70000.0)  # e.g. (52000.0, 76000.0)

X_LIMITS = (-30000.0,10000.0)  # e.g. (110000.0, 145000.0)
Y_LIMITS = (-75000.0, None)  # e.g. (-95000.0, -60000.0)
Z_LIMITS = (None, None)  # e.g. (52000.0, 76000.0)

# Optional time-index limits `(start, stop)` for rendering.
# Uses Python slice semantics: start inclusive, stop exclusive.
# Set to None to use the full available time range.
# TIME_LIMITS = (150, None)  # e.g. (50, 110)
TIME_LIMITS = (400, None)  # e.g. (50, 110)

# Virtual lidar location as actual coordinate values `(x, y)`.
# Use `None` for either coordinate to keep it centered along that dimension.
VIRTUAL_LIDAR_LOCATION = (123000.0, -84000.0)  # e.g. (125000.0, -70000.0)
VIRTUAL_LIDAR_LOCATION = (-10000.0, -60000.0)  # e.g. (125000.0, -70000.0)
QUARTER_CUT_REMOVED_CORNER = "xmin_ymin"

TIMESERIES_ALTITUDE_VALUE = 63000

NEGATIVE_LAMBDA2_ONLY = True
SINGLE_LAMBDA2_REFERENCE_TIME_INDEX = TIME_INDEX
SINGLE_LAMBDA2_PERCENTILE = 5.0
MANUAL_LAMBDA2_LEVEL = None

VORTICITY_COMPONENT = None
VORTICITY_COLORMAP = cm.batlow
VORTICITY_PERCENTILE_RANGE = (5.0, 99.5)

TPRIME_COLORMAP = cmaps.get_wave_cmap()
TPRIME_PERCENTILE = 99.0
W_SLICE_COLORMAP = cm.vik
W_SLICE_PERCENTILE = 99.0
SLICE_OPACITY = 1.0

LAMBDA2_CONTOUR_COLOR = "black"
LAMBDA2_CONTOUR_WIDTH = 4.0

SHOW_BOTTOM_W_CONTOURS = True
BOTTOM_W_CONTOUR_COUNT = 10
BOTTOM_W_CONTOUR_COLOR = "black"
BOTTOM_W_CONTOUR_WIDTH = 2.0
BOTTOM_W_CONTOUR_OPACITY = 0.85
EXCLUDE_ZERO_FROM_BOTTOM_W_CONTOURS = True

SMOOTH_SIGMA = 2.0

SHOW_MEAN_WIND_ARROW = False
MEAN_WIND_ARROW_SCALE = 0.18
MEAN_WIND_MIN_VISIBLE_FRACTION = 0.08
MEAN_WIND_ARROW_COLOR = "white"
MEAN_WIND_ARROW_OPACITY = 1.0

TIMESERIES_ALTITUDE_INDEX = None
TPRIME_CURTAIN_PERCENTILE = 99.0

LIDAR_LINE_COLOR = "black"
LIDAR_LINE_WIDTH = 7
LIDAR_LINE_N_DASHES = 12
LIDAR_LINE_DASH_FRACTION = 0.58

SLICE_REFERENCE_LINE_COLOR = "black"
SLICE_REFERENCE_LINE_WIDTH = 5
SLICE_REFERENCE_LINE_N_DASHES = 14
SLICE_REFERENCE_LINE_DASH_FRACTION = 0.55

SHOW_VORTICITY_SCALAR_BAR_IN_CLASSIC_VIEW = True
SHOW_TPRIME_SCALAR_BAR_IN_XZ_VIEW = True
SHOW_W_SCALAR_BAR_IN_XY_VIEW = True

VORTICITY_SCALAR_BAR_ARGS = dict(
    title_font_size=14,
    label_font_size=12,
    vertical=True,
    position_x=0.86,
    position_y=0.08,
    width=0.08,
    height=0.72,
)

TPRIME_SCALAR_BAR_ARGS = dict(
    title="T'",
    title_font_size=14,
    label_font_size=12,
    vertical=True,
    position_x=0.86,
    position_y=0.08,
    width=0.08,
    height=0.72,
)

W_SCALAR_BAR_ARGS = dict(
    title=W_NAME,
    title_font_size=14,
    label_font_size=12,
    vertical=True,
    position_x=0.86,
    position_y=0.08,
    width=0.08,
    height=0.72,
)

BACKGROUND = "white"
SHOW_OUTLINE = True
SHOW_BOUNDS_AXES = True
SHOW_AXES = False
SHOW_TIME_LABEL_3D = True
WINDOW_SIZE_3D = (950, 720)

USE_REFERENCE_DATETIME_FOR_LABELS = False
REFERENCE_DATETIME = None
LIDAR_TIME_TEXT_X = 0.02
LIDAR_TIME_TEXT_Y = 0.96

COMBINED_FIGURE_SIZE = (16.0, 12.0)
COMBINED_DPI = 150

FPS = 6
PARALLEL_FRAME_GENERATION = True
# ANIMATION_NCPUS = max(1, mp.cpu_count() - 2)
ANIMATION_NCPUS = min(50, mp.cpu_count() - 2)
CLEAR_EXISTING_FRAMES = True


if (Path(__file__).with_name("latex_default.mplstyle")).exists():
    plt.style.use(Path(__file__).with_name("latex_default.mplstyle"))


def configure_headless_rendering():
    """Prefer EGL, then OSMesa, then the default/X backend before first render."""
    backend_env = "VTK_DEFAULT_OPENGL_WINDOW"
    if os.environ.get(backend_env):
        print(f"[i]  Using preconfigured VTK backend: {os.environ[backend_env]}")
        return

    backend_candidates = [
        ("EGL", "vtkEGLRenderWindow"),
        ("OSMesa", "vtkOSOpenGLRenderWindow"),
        ("X/default", None),
    ]

    for backend_label, backend_value in backend_candidates:
        if probe_vtk_backend(backend_value):
            if backend_value is None:
                print("[i]  Falling back to VTK default/X render window backend.")
            else:
                os.environ[backend_env] = backend_value
                print(f"[i]  Selected VTK backend: {backend_label} ({backend_value})")
            return

    print("[w]  Could not validate EGL or OSMesa. Continuing with VTK default backend.")


def probe_vtk_backend(backend_value):
    probe_env = os.environ.copy()
    probe_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cube_lid")
    if backend_value is None:
        probe_env.pop("VTK_DEFAULT_OPENGL_WINDOW", None)
    else:
        probe_env["VTK_DEFAULT_OPENGL_WINDOW"] = backend_value

    probe_code = """
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cube_lid")
import pyvista as pv
pv.OFF_SCREEN = True
plotter = pv.Plotter(off_screen=True, window_size=(32, 32))
plotter.add_mesh(pv.Sphere(radius=0.5))
plotter.screenshot(filename=None, return_img=True)
plotter.close()
"""

    result = subprocess.run(
        [sys.executable, "-c", probe_code],
        env=probe_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def load_time_slice(ds, time_index, stride=(1, 1, 1), extra_var_names=None):
    z_stride, y_stride, x_stride = stride
    extra_var_names = extra_var_names or []

    x = np.asarray(ds[X_NAME].values)[::x_stride]
    y = np.asarray(ds[Y_NAME].values)[::y_stride]
    z = np.asarray(ds[Z_NAME].values)[::z_stride]

    arrays = {}
    for name in [U_NAME, V_NAME, W_NAME, *extra_var_names]:
        da = ds[name].isel({TIME_NAME: time_index}).transpose(Z_NAME, Y_NAME, X_NAME).load()
        arrays[name] = np.asarray(da.values)[::z_stride, ::y_stride, ::x_stride]

    return x, y, z, arrays


def normalize_limits(limits):
    if limits is None:
        return None
    lo, hi = limits
    lo = None if lo is None else float(lo)
    hi = None if hi is None else float(hi)
    if lo is None and hi is None:
        return None
    if lo is None:
        return None, hi
    if hi is None:
        return lo, None
    return min(lo, hi), max(lo, hi)


def apply_spatial_limits(ds):
    indexers = {}
    for coord_name, limits in ((X_NAME, X_LIMITS), (Y_NAME, Y_LIMITS), (Z_NAME, Z_LIMITS)):
        limits = normalize_limits(limits)
        if limits is not None:
            indexers[coord_name] = slice(limits[0], limits[1])

    if not indexers:
        return ds

    limited = ds.sel(indexers)
    for coord_name, coord_slice in indexers.items():
        if int(limited.sizes.get(coord_name, 0)) == 0:
            raise ValueError(f"Spatial limits for {coord_name!r} produced an empty selection: {coord_slice}")
    return limited


def select_time_indices(ds):
    total_time_indices = int(ds.sizes[TIME_NAME])
    if TIME_LIMITS is None:
        return list(range(total_time_indices))

    if len(TIME_LIMITS) != 2:
        raise ValueError("TIME_LIMITS must be None or a tuple/list of (start, stop).")

    start, stop = TIME_LIMITS
    start = 0 if start is None else int(start)
    stop = total_time_indices if stop is None else int(stop)

    start = max(0, min(start, total_time_indices))
    stop = max(0, min(stop, total_time_indices))
    if stop <= start:
        raise ValueError(
            f"TIME_LIMITS={TIME_LIMITS!r} produced an empty/invalid selection for total_time_indices={total_time_indices}."
        )

    return list(range(start, stop))


def compute_temperature_from_theta_exner(theta, exner):
    return np.asarray(theta, dtype=np.float64) * np.asarray(exner, dtype=np.float64)


def compute_pressure_from_exner(exner, p0=REFERENCE_PRESSURE_PA, rd=RD, cp=CP):
    exner = np.asarray(exner, dtype=np.float64)
    return p0 * np.power(exner, 1.0 / (rd / cp))


def compute_temperature_from_ideal_gas(density, pressure, rd=RD):
    density = np.asarray(density, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    return pressure / (density * rd)


def compute_tprime_from_temperature(temperature):
    temperature = np.asarray(temperature, dtype=np.float64)
    mean_xy = np.nanmean(temperature, axis=(1, 2), keepdims=True)
    return temperature - mean_xy


def get_nearest_index(coord_values, coord_value):
    coord_values = np.asarray(coord_values, dtype=float)
    if coord_value is None:
        return len(coord_values) // 2
    return int(np.nanargmin(np.abs(coord_values - float(coord_value))))


def load_virtual_lidar_tprime(ds, x_value=None, y_value=None):
    nx = int(ds.sizes[X_NAME])
    ny = int(ds.sizes[Y_NAME])
    x_values = np.asarray(ds[X_NAME].values, dtype=float)
    y_values = np.asarray(ds[Y_NAME].values, dtype=float)
    x_index = get_nearest_index(x_values, x_value)
    y_index = get_nearest_index(y_values, y_value)
    x_index = int(np.clip(x_index, 0, nx - 1))
    y_index = int(np.clip(y_index, 0, ny - 1))

    theta = ds[THETA_NAME].transpose(TIME_NAME, Z_NAME, Y_NAME, X_NAME)
    exner = ds[EXNER_NAME].transpose(TIME_NAME, Z_NAME, Y_NAME, X_NAME)
    temperature = theta * exner
    tprime = temperature - temperature.mean(dim=(Y_NAME, X_NAME))
    curtain = tprime.isel({Y_NAME: int(y_index), X_NAME: int(x_index)}).load()

    times = np.asarray(ds[TIME_NAME].values)
    z = np.asarray(ds[Z_NAME].values)
    x_val = float(x_values[int(x_index)])
    y_val = float(y_values[int(y_index)])

    return (
        times,
        z,
        np.asarray(curtain.values),
        int(x_index),
        int(y_index),
        x_val,
        y_val,
    )


def compute_lambda2_and_vorticity(u, v, w, x, y, z):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if min(len(x), len(y), len(z)) < 3:
        raise ValueError("Need at least 3 points along x, y, and z to compute gradients.")

    du_dz, du_dy, du_dx = np.gradient(u, z, y, x, edge_order=2)
    dv_dz, dv_dy, dv_dx = np.gradient(v, z, y, x, edge_order=2)
    dw_dz, dw_dy, dw_dx = np.gradient(w, z, y, x, edge_order=2)

    sxx = du_dx
    syy = dv_dy
    szz = dw_dz
    sxy = 0.5 * (du_dy + dv_dx)
    sxz = 0.5 * (du_dz + dw_dx)
    syz = 0.5 * (dv_dz + dw_dy)

    oxy = 0.5 * (du_dy - dv_dx)
    oxz = 0.5 * (du_dz - dw_dx)
    oyz = 0.5 * (dv_dz - dw_dy)

    vort_x = dw_dy - dv_dz
    vort_y = du_dz - dw_dx
    vort_z = dv_dx - du_dy
    vort_mag = np.sqrt(vort_x**2 + vort_y**2 + vort_z**2)

    if VORTICITY_COMPONENT is None:
        vort_scalars = vort_mag
        vort_name = "|ω|"
    elif str(VORTICITY_COMPONENT).lower() == "x":
        vort_scalars = vort_x
        vort_name = "ω_x"
    elif str(VORTICITY_COMPONENT).lower() == "y":
        vort_scalars = vort_y
        vort_name = "ω_y"
    elif str(VORTICITY_COMPONENT).lower() == "z":
        vort_scalars = vort_z
        vort_name = "ω_z"
    else:
        raise ValueError("vorticity_component must be None, 'x', 'y', or 'z'.")

    nz = u.shape[0]
    lambda2 = np.empty_like(u, dtype=np.float64)
    chunk_size_z = max(1, min(nz, 12))
    for z0 in range(0, nz, chunk_size_z):
        z1 = min(z0 + chunk_size_z, nz)
        sl = slice(z0, z1)

        s = np.zeros((z1 - z0, u.shape[1], u.shape[2], 3, 3), dtype=np.float64)
        o = np.zeros_like(s)

        s[..., 0, 0] = sxx[sl]
        s[..., 1, 1] = syy[sl]
        s[..., 2, 2] = szz[sl]
        s[..., 0, 1] = s[..., 1, 0] = sxy[sl]
        s[..., 0, 2] = s[..., 2, 0] = sxz[sl]
        s[..., 1, 2] = s[..., 2, 1] = syz[sl]

        o[..., 0, 1] = oxy[sl]
        o[..., 1, 0] = -oxy[sl]
        o[..., 0, 2] = oxz[sl]
        o[..., 2, 0] = -oxz[sl]
        o[..., 1, 2] = oyz[sl]
        o[..., 2, 1] = -oyz[sl]

        eigvals = np.linalg.eigvalsh(s @ s + o @ o)
        lambda2[sl] = eigvals[..., 1]

    return lambda2, vort_scalars, vort_mag, (vort_x, vort_y, vort_z), vort_name


def build_structured_grid(x, y, z, point_arrays, active_scalar_name):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
    grid = pv.StructuredGrid(xg, yg, zg)

    expected_shape = (len(z), len(y), len(x))
    for name, arr in point_arrays.items():
        arr = np.asarray(arr)
        if arr.shape != expected_shape:
            raise ValueError(f"Array {name!r} has shape {arr.shape}, expected {expected_shape}")
        grid.point_data[name] = np.transpose(arr, (2, 1, 0)).ravel(order="F")

    grid.set_active_scalars(active_scalar_name)
    return grid


def choose_single_lambda2_level(lambda2, negative_only=True, percentile=5.0, manual_level=None):
    if manual_level is not None:
        return float(manual_level)

    vals = np.asarray(lambda2)
    vals = vals[np.isfinite(vals)]
    if negative_only:
        vals = vals[vals < 0]
        if vals.size == 0:
            raise ValueError("No negative λ2 values found. Try negative_lambda2_only = False.")
    if not (0 <= percentile <= 100):
        raise ValueError("single_lambda2_percentile must be between 0 and 100.")
    return float(np.percentile(vals, percentile))


def choose_clim(data, percentile_range):
    vals = np.asarray(data)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No finite values found.")
    p_lo, p_hi = percentile_range
    vmin, vmax = np.percentile(vals, [p_lo, p_hi])
    if vmin == vmax:
        vmax = vmin + 1e-12
    return float(vmin), float(vmax)


def choose_symmetric_clim(data, percentile=99.0):
    vals = np.asarray(data)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No finite values found.")
    vmax = float(np.percentile(np.abs(vals), percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(vals)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1e-12
    return -vmax, vmax


def choose_symmetric_contour_levels(clim, count=10, exclude_zero=True):
    vmax = max(abs(float(clim[0])), abs(float(clim[1])))
    levels = np.linspace(-vmax, vmax, max(1, int(count)) + 2)
    if exclude_zero:
        levels = levels[np.abs(levels) > 1e-12]
    return np.asarray(levels, dtype=float)


def choose_altitude(z_values, altitude_index=None, altitude_value=None):
    z_values = np.asarray(z_values, dtype=float)
    if altitude_value is not None:
        idx = int(np.nanargmin(np.abs(z_values - float(altitude_value))))
    elif altitude_index is None:
        idx = len(z_values) // 2
    else:
        idx = int(np.clip(int(altitude_index), 0, len(z_values) - 1))
    return idx, float(z_values[idx])


def get_quarter_cut_geometry(x, y, z, x_cut_value, y_cut_value, removed_corner="xmax_ymax"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    x_cut = float(x_cut_value)
    y_cut = float(y_cut_value)

    if removed_corner == "xmax_ymax":
        remove_bounds = (x_cut, xmax, y_cut, ymax, zmin, zmax)
        yz_keep = ("y", y_cut, False)
        xz_keep = ("x", x_cut, False)
    elif removed_corner == "xmax_ymin":
        remove_bounds = (x_cut, xmax, ymin, y_cut, zmin, zmax)
        yz_keep = ("y", y_cut, True)
        xz_keep = ("x", x_cut, False)
    elif removed_corner == "xmin_ymax":
        remove_bounds = (xmin, x_cut, y_cut, ymax, zmin, zmax)
        yz_keep = ("y", y_cut, False)
        xz_keep = ("x", x_cut, True)
    elif removed_corner == "xmin_ymin":
        remove_bounds = (xmin, x_cut, ymin, y_cut, zmin, zmax)
        yz_keep = ("y", y_cut, True)
        xz_keep = ("x", x_cut, True)
    else:
        raise ValueError("quarter_cut_removed_corner must be one of: xmax_ymax, xmax_ymin, xmin_ymax, xmin_ymin")

    return {
        "remove_bounds": tuple(float(v) for v in remove_bounds),
        "yz_keep": yz_keep,
        "xz_keep": xz_keep,
    }


def clip_slice_to_exposed_halfplane(slice_mesh, axis, value, invert):
    if slice_mesh is None or slice_mesh.n_points == 0:
        return slice_mesh
    xmid, ymid, zmid = slice_mesh.center
    if axis == "x":
        return slice_mesh.clip(normal="x", origin=(float(value), float(ymid), float(zmid)), invert=bool(invert))
    if axis == "y":
        return slice_mesh.clip(normal="y", origin=(float(xmid), float(value), float(zmid)), invert=bool(invert))
    raise ValueError("axis must be 'x' or 'y'")


def compute_elapsed_seconds(value, time_start_value):
    value_arr = np.asarray(value)
    start_arr = np.asarray(time_start_value)
    if np.issubdtype(value_arr.dtype, np.datetime64) or np.issubdtype(start_arr.dtype, np.datetime64):
        value_ns = value_arr.astype("datetime64[ns]")
        start_ns = start_arr.astype("datetime64[ns]")
        return np.asarray((value_ns - start_ns) / np.timedelta64(1, "s"), dtype=np.float64)
    return np.asarray(value_arr, dtype=np.float64) - float(start_arr)


def format_elapsed_hms(total_seconds):
    total_seconds = max(0, int(np.rint(float(total_seconds))))
    hrs = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def format_reference_timestamp(elapsed_seconds):
    if not USE_REFERENCE_DATETIME_FOR_LABELS or REFERENCE_DATETIME is None:
        return None
    ts = pd.to_datetime(REFERENCE_DATETIME) + pd.to_timedelta(float(elapsed_seconds), unit="s")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def format_time_value(value, time_start_value, frame_index=None):
    elapsed_seconds = float(np.asarray(compute_elapsed_seconds(value, time_start_value)).reshape(-1)[0])
    label = f"Time: {format_elapsed_hms(elapsed_seconds)}s"
    if frame_index is not None:
        label += f" ({int(frame_index):03d})"
    reference_label = format_reference_timestamp(elapsed_seconds)
    if reference_label is not None:
        label = f"{label}\n{reference_label}"
    return label


def format_elapsed_tick_label(value, pos=None):
    return format_elapsed_hms(value)


def apply_rotated_default_camera(plotter):
    plotter.view_isometric()
    focal_point = np.asarray(plotter.camera.focal_point, dtype=float)
    position = np.asarray(plotter.camera.position, dtype=float)
    view_up = tuple(np.asarray(plotter.camera.up, dtype=float))
    rel = position - focal_point
    rel[0] *= -1.0
    rel[1] *= -1.0
    plotter.camera_position = [tuple(focal_point + rel), tuple(focal_point), view_up]


def add_mean_wind_arrow(plotter, x, y, z, u, v):
    if not SHOW_MEAN_WIND_ARROW:
        return

    u_mean = float(np.nanmean(u))
    v_mean = float(np.nanmean(v))
    mean_hvec = np.array([u_mean, v_mean, 0.0], dtype=float)
    mean_hspeed = float(np.linalg.norm(mean_hvec[:2]))
    if mean_hspeed <= 0 or not np.isfinite(mean_hspeed):
        return

    x_mid = float(0.5 * (np.nanmin(x) + np.nanmax(x)))
    y_mid = float(0.5 * (np.nanmin(y) + np.nanmax(y)))
    z0 = float(np.nanmin(z))
    origin = np.array([x_mid, y_mid, z0], dtype=float)

    x_span = float(np.nanmax(x) - np.nanmin(x))
    y_span = float(np.nanmax(y) - np.nanmin(y))
    horizontal_span = max(np.hypot(x_span, y_span), 1e-12)

    local_hspeed = np.sqrt(u**2 + v**2)
    characteristic_speed = float(np.nanpercentile(local_hspeed, 95))
    if not np.isfinite(characteristic_speed) or characteristic_speed <= 0:
        characteristic_speed = mean_hspeed

    direction = mean_hvec / mean_hspeed
    scaled_length = horizontal_span * MEAN_WIND_ARROW_SCALE * (mean_hspeed / characteristic_speed)
    arrow_length = max(scaled_length, horizontal_span * MEAN_WIND_MIN_VISIBLE_FRACTION)

    arrow = pv.Arrow(
        start=origin,
        direction=direction,
        scale=arrow_length,
        tip_length=0.22,
        tip_radius=0.05,
        shaft_radius=0.018,
    )
    plotter.add_mesh(arrow, color=MEAN_WIND_ARROW_COLOR, opacity=MEAN_WIND_ARROW_OPACITY)


def make_dashed_line_segments(p0, p1, n_dashes=12, dash_fraction=0.58):
    p0 = np.asarray(p0, dtype=float).reshape(3)
    p1 = np.asarray(p1, dtype=float).reshape(3)
    edges = np.linspace(0.0, 1.0, max(1, int(n_dashes)) + 1)
    dash_fraction = float(np.clip(dash_fraction, 0.05, 0.95))

    pts = []
    direction = p1 - p0
    for ta, tb in zip(edges[:-1], edges[1:]):
        tend = ta + dash_fraction * (tb - ta)
        pts.append(p0 + ta * direction)
        pts.append(p0 + tend * direction)
    return np.asarray(pts, dtype=float)


def make_dashed_vertical_segments(x0, y0, zmin, zmax, n_dashes=12, dash_fraction=0.58):
    return make_dashed_line_segments(
        [float(x0), float(y0), float(zmin)],
        [float(x0), float(y0), float(zmax)],
        n_dashes=n_dashes,
        dash_fraction=dash_fraction,
    )


def add_virtual_lidar_line(plotter, x0, y0, zmin, zmax):
    segments = make_dashed_vertical_segments(
        x0=x0,
        y0=y0,
        zmin=zmin,
        zmax=zmax,
        n_dashes=LIDAR_LINE_N_DASHES,
        dash_fraction=LIDAR_LINE_DASH_FRACTION,
    )
    plotter.add_lines(segments, color=LIDAR_LINE_COLOR, width=LIDAR_LINE_WIDTH)


def prepare_scene_data_for_time(ds, time_index, lambda2_level, stride=(1, 1, 1)):
    x, y, z, arrays = load_time_slice(
        ds,
        time_index=time_index,
        stride=stride,
        extra_var_names=[DENSITY_NAME, THETA_NAME, EXNER_NAME],
    )

    u = np.asarray(arrays[U_NAME], dtype=np.float64)
    v = np.asarray(arrays[V_NAME], dtype=np.float64)
    w = np.asarray(arrays[W_NAME], dtype=np.float64)
    density = np.asarray(arrays[DENSITY_NAME], dtype=np.float64)
    theta = np.asarray(arrays[THETA_NAME], dtype=np.float64)
    exner = np.asarray(arrays[EXNER_NAME], dtype=np.float64)

    temperature = compute_temperature_from_theta_exner(theta, exner)
    tprime = compute_tprime_from_temperature(temperature)
    lambda2, vort_scalars, vort_mag, vort_vector, vort_name = compute_lambda2_and_vorticity(u, v, w, x, y, z)

    if SMOOTH_SIGMA and SMOOTH_SIGMA > 0:
        if gaussian_filter is None:
            raise ImportError("scipy is required for smoothing; install scipy or set smooth_sigma = 0")
        lambda2 = gaussian_filter(lambda2, sigma=SMOOTH_SIGMA)
        vort_scalars = gaussian_filter(vort_scalars, sigma=SMOOTH_SIGMA)
        vort_mag = gaussian_filter(vort_mag, sigma=SMOOTH_SIGMA)
        w = gaussian_filter(w, sigma=SMOOTH_SIGMA)
        tprime = gaussian_filter(tprime, sigma=SMOOTH_SIGMA)

    point_arrays = {
        "lambda2": lambda2,
        "vorticity_scalar": vort_scalars,
        "vorticity_magnitude": vort_mag,
        "w": w,
        "tprime": tprime,
        "temperature": temperature,
        "density": density,
        "theta": theta,
        "exner": exner,
    }
    grid = build_structured_grid(x, y, z, point_arrays=point_arrays, active_scalar_name="lambda2")
    surface = grid.contour(isosurfaces=[float(lambda2_level)], scalars="lambda2")

    return {
        "x": x,
        "y": y,
        "z": z,
        "u": u,
        "v": v,
        "w": w,
        "grid": grid,
        "surface": surface,
        "vorticity_name": vort_name,
    }


def build_bottom_w_contours(x, y, z, w_field, contour_levels):
    if not SHOW_BOTTOM_W_CONTOURS or contour_levels is None or len(contour_levels) == 0:
        return None

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    w_field = np.asarray(w_field, dtype=float)
    if w_field.ndim != 3 or w_field.shape[0] < 1:
        return None

    w0 = w_field[0]
    xb, yb = np.meshgrid(x, y, indexing="ij")
    zb = np.full_like(xb, float(np.min(z)))
    bottom = pv.StructuredGrid(xb, yb, zb)
    bottom.point_data["w"] = np.asarray(w0.T, dtype=float).ravel(order="F")

    vals = np.asarray(bottom.point_data["w"])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    lo = float(np.min(vals))
    hi = float(np.max(vals))
    levels = np.asarray(contour_levels, dtype=float)
    levels = levels[(levels >= lo) & (levels <= hi)]
    if levels.size == 0:
        return None

    lines = bottom.contour(isosurfaces=levels, scalars="w")
    return None if lines.n_points == 0 else lines


def build_lambda2_slice_contours_from_surface(surface, normal, origin):
    if surface is None or surface.n_points == 0:
        return None
    lines = surface.slice(normal=normal, origin=origin)
    return None if lines.n_points == 0 else lines


def populate_plotter_variant(
    plotter,
    scene_data,
    variant,
    camera_position,
    vorticity_clim,
    tprime_clim,
    w_clim,
    bottom_w_contour_levels,
    lidar_x_val,
    lidar_y_val,
    slice_z_val,
    show_time_label=False,
    time_label_text=None,
):
    plotter.clear()
    plotter.set_background(BACKGROUND)

    x = scene_data["x"]
    y = scene_data["y"]
    z = scene_data["z"]
    u = scene_data["u"]
    v = scene_data["v"]
    grid = scene_data["grid"]
    surface = scene_data["surface"]
    w = scene_data["w"]
    vort_name = scene_data["vorticity_name"]

    if SHOW_OUTLINE:
        plotter.add_mesh(grid.outline(), color="black", line_width=1)

    if variant == "classic":
        if surface.n_points > 0:
            sb_args = dict(VORTICITY_SCALAR_BAR_ARGS)
            sb_args["title"] = vort_name
            plotter.add_mesh(
                surface,
                scalars="vorticity_scalar",
                cmap=VORTICITY_COLORMAP,
                clim=vorticity_clim,
                opacity=1.0,
                smooth_shading=True,
                show_scalar_bar=SHOW_VORTICITY_SCALAR_BAR_IN_CLASSIC_VIEW,
                scalar_bar_args=sb_args,
            )

    elif variant == "xz_slice":
        if lidar_x_val is None or lidar_y_val is None:
            raise ValueError("lidar_x_val and lidar_y_val are required for the xz_slice variant.")

        cut = get_quarter_cut_geometry(
            x=x,
            y=y,
            z=z,
            x_cut_value=float(lidar_x_val),
            y_cut_value=float(lidar_y_val),
            removed_corner=QUARTER_CUT_REMOVED_CORNER,
        )

        surface_cut = surface
        if surface_cut is not None and surface_cut.n_points > 0:
            surface_cut = surface_cut.clip_box(cut["remove_bounds"], invert=True)
            if surface_cut.n_points > 0:
                plotter.add_mesh(
                    surface_cut,
                    scalars="vorticity_scalar",
                    cmap=VORTICITY_COLORMAP,
                    clim=vorticity_clim,
                    opacity=1.0,
                    smooth_shading=True,
                    show_scalar_bar=False,
                )

        yz_slice = grid.slice(normal="x", origin=(float(lidar_x_val), float(np.mean(y)), float(np.mean(z))))
        yz_axis, yz_value, yz_invert = cut["yz_keep"]
        yz_slice = clip_slice_to_exposed_halfplane(yz_slice, yz_axis, yz_value, yz_invert)

        xz_slice = grid.slice(normal="y", origin=(float(np.mean(x)), float(lidar_y_val), float(np.mean(z))))
        xz_axis, xz_value, xz_invert = cut["xz_keep"]
        xz_slice = clip_slice_to_exposed_halfplane(xz_slice, xz_axis, xz_value, xz_invert)

        scalar_bar_drawn = False
        for cut_slice in (yz_slice, xz_slice):
            if cut_slice is not None and cut_slice.n_points > 0:
                plotter.add_mesh(
                    cut_slice,
                    scalars="tprime",
                    cmap=TPRIME_COLORMAP,
                    clim=tprime_clim,
                    opacity=float(SLICE_OPACITY),
                    smooth_shading=False,
                    show_scalar_bar=(SHOW_TPRIME_SCALAR_BAR_IN_XZ_VIEW and not scalar_bar_drawn),
                    scalar_bar_args=dict(TPRIME_SCALAR_BAR_ARGS),
                )
                scalar_bar_drawn = True

        yz_lines = build_lambda2_slice_contours_from_surface(
            surface,
            normal="x",
            origin=(float(lidar_x_val), float(np.mean(y)), float(np.mean(z))),
        )
        yz_lines = clip_slice_to_exposed_halfplane(yz_lines, yz_axis, yz_value, yz_invert)
        if yz_lines is not None and yz_lines.n_points > 0:
            plotter.add_mesh(yz_lines, color=LAMBDA2_CONTOUR_COLOR, line_width=LAMBDA2_CONTOUR_WIDTH, render_lines_as_tubes=False)

        xz_lines = build_lambda2_slice_contours_from_surface(
            surface,
            normal="y",
            origin=(float(np.mean(x)), float(lidar_y_val), float(np.mean(z))),
        )
        xz_lines = clip_slice_to_exposed_halfplane(xz_lines, xz_axis, xz_value, xz_invert)
        if xz_lines is not None and xz_lines.n_points > 0:
            plotter.add_mesh(xz_lines, color=LAMBDA2_CONTOUR_COLOR, line_width=LAMBDA2_CONTOUR_WIDTH, render_lines_as_tubes=False)

        if slice_z_val is not None:
            yz_ref = pv.PolyData(
                make_dashed_line_segments(
                    [float(lidar_x_val), float(np.nanmin(y)), float(slice_z_val)],
                    [float(lidar_x_val), float(np.nanmax(y)), float(slice_z_val)],
                    n_dashes=SLICE_REFERENCE_LINE_N_DASHES,
                    dash_fraction=SLICE_REFERENCE_LINE_DASH_FRACTION,
                )
            )
            yz_ref = clip_slice_to_exposed_halfplane(yz_ref, yz_axis, yz_value, yz_invert)
            if yz_ref is not None and yz_ref.n_points > 0:
                plotter.add_mesh(yz_ref, color=SLICE_REFERENCE_LINE_COLOR, line_width=SLICE_REFERENCE_LINE_WIDTH, render_lines_as_tubes=False)

            xz_ref = pv.PolyData(
                make_dashed_line_segments(
                    [float(np.nanmin(x)), float(lidar_y_val), float(slice_z_val)],
                    [float(np.nanmax(x)), float(lidar_y_val), float(slice_z_val)],
                    n_dashes=SLICE_REFERENCE_LINE_N_DASHES,
                    dash_fraction=SLICE_REFERENCE_LINE_DASH_FRACTION,
                )
            )
            xz_ref = clip_slice_to_exposed_halfplane(xz_ref, xz_axis, xz_value, xz_invert)
            if xz_ref is not None and xz_ref.n_points > 0:
                plotter.add_mesh(xz_ref, color=SLICE_REFERENCE_LINE_COLOR, line_width=SLICE_REFERENCE_LINE_WIDTH, render_lines_as_tubes=False)

    elif variant == "xy_slice":
        if slice_z_val is None:
            raise ValueError("slice_z_val is required for the xy_slice variant.")

        slice_mesh = grid.slice(normal="z", origin=(float(np.mean(x)), float(np.mean(y)), float(slice_z_val)))
        if slice_mesh.n_points > 0:
            plotter.add_mesh(
                slice_mesh,
                scalars="w",
                cmap=W_SLICE_COLORMAP,
                clim=w_clim,
                opacity=float(SLICE_OPACITY),
                smooth_shading=False,
                show_scalar_bar=SHOW_W_SCALAR_BAR_IN_XY_VIEW,
                scalar_bar_args=dict(W_SCALAR_BAR_ARGS),
            )
            lambda2_lines = build_lambda2_slice_contours_from_surface(
                surface,
                normal="z",
                origin=(float(np.mean(x)), float(np.mean(y)), float(slice_z_val)),
            )
            if lambda2_lines is not None:
                plotter.add_mesh(lambda2_lines, color=LAMBDA2_CONTOUR_COLOR, line_width=LAMBDA2_CONTOUR_WIDTH, render_lines_as_tubes=False)

            if lidar_x_val is not None and lidar_y_val is not None:
                x_ref = pv.PolyData(
                    make_dashed_line_segments(
                        [float(lidar_x_val), float(np.nanmin(y)), float(slice_z_val)],
                        [float(lidar_x_val), float(np.nanmax(y)), float(slice_z_val)],
                        n_dashes=SLICE_REFERENCE_LINE_N_DASHES,
                        dash_fraction=SLICE_REFERENCE_LINE_DASH_FRACTION,
                    )
                )
                y_ref = pv.PolyData(
                    make_dashed_line_segments(
                        [float(np.nanmin(x)), float(lidar_y_val), float(slice_z_val)],
                        [float(np.nanmax(x)), float(lidar_y_val), float(slice_z_val)],
                        n_dashes=SLICE_REFERENCE_LINE_N_DASHES,
                        dash_fraction=SLICE_REFERENCE_LINE_DASH_FRACTION,
                    )
                )
                plotter.add_mesh(x_ref, color=SLICE_REFERENCE_LINE_COLOR, line_width=SLICE_REFERENCE_LINE_WIDTH, render_lines_as_tubes=False)
                plotter.add_mesh(y_ref, color=SLICE_REFERENCE_LINE_COLOR, line_width=SLICE_REFERENCE_LINE_WIDTH, render_lines_as_tubes=False)

    else:
        raise ValueError("variant must be 'classic', 'xz_slice', or 'xy_slice'.")

    bottom_w_lines = build_bottom_w_contours(x, y, z, w_field=w, contour_levels=bottom_w_contour_levels)
    if bottom_w_lines is not None:
        plotter.add_mesh(
            bottom_w_lines,
            color=BOTTOM_W_CONTOUR_COLOR,
            line_width=BOTTOM_W_CONTOUR_WIDTH,
            opacity=BOTTOM_W_CONTOUR_OPACITY,
            render_lines_as_tubes=False,
        )

    add_mean_wind_arrow(plotter, x, y, z, u, v)

    if lidar_x_val is not None and lidar_y_val is not None:
        add_virtual_lidar_line(plotter, float(lidar_x_val), float(lidar_y_val), float(np.nanmin(z)), float(np.nanmax(z)))

    if SHOW_BOUNDS_AXES:
        plotter.show_bounds(grid="back", location="outer", xtitle="3D", ytitle="3D", ztitle="3D", n_xlabels=5, n_ylabels=5, n_zlabels=5, font_size=12)
    if SHOW_AXES:
        plotter.show_axes()
    plotter.add_bounding_box(color="black")

    if show_time_label and time_label_text is not None:
        plotter.add_text(time_label_text, position="upper_left", font_size=11, color="black")

    if camera_position is not None:
        plotter.camera_position = camera_position
    else:
        apply_rotated_default_camera(plotter)


def render_pyvista_variant_frame(
    plotter,
    scene_data,
    variant,
    camera_position,
    vorticity_clim,
    tprime_clim,
    w_clim,
    bottom_w_contour_levels,
    lidar_x_val,
    lidar_y_val,
    slice_z_val,
    time_label_text,
):
    populate_plotter_variant(
        plotter=plotter,
        scene_data=scene_data,
        variant=variant,
        camera_position=camera_position,
        vorticity_clim=vorticity_clim,
        tprime_clim=tprime_clim,
        w_clim=w_clim,
        bottom_w_contour_levels=bottom_w_contour_levels,
        lidar_x_val=lidar_x_val,
        lidar_y_val=lidar_y_val,
        slice_z_val=slice_z_val,
        show_time_label=SHOW_TIME_LABEL_3D,
        time_label_text=time_label_text,
    )
    return plotter.screenshot(filename=None, return_img=True)


def plot_virtual_lidar_panels(ax_curtain, ax_series, times, z, tprime_curtain, current_time_value, selected_altitude_value, tprime_clim, frame_index=None):
    times = np.asarray(times)
    z = np.asarray(z)
    curtain = np.asarray(tprime_curtain)
    if curtain.ndim != 2:
        raise ValueError("tprime_curtain must have shape (nt, nz)")

    altitude_idx = int(np.nanargmin(np.abs(z - float(selected_altitude_value))))
    tprime_series = curtain[:, altitude_idx]
    z_km = z / 1000.0
    selected_altitude_km = float(selected_altitude_value) / 1000.0
    time_start_value = times[0]
    x_vals = np.asarray(compute_elapsed_seconds(times, time_start_value), dtype=np.float64)
    current_x = float(np.asarray(compute_elapsed_seconds(current_time_value, time_start_value)).reshape(-1)[0])
    contour_levels = np.linspace(tprime_clim[0], tprime_clim[1], 61)

    mesh = ax_curtain.contourf(
        x_vals,
        z_km,
        curtain.T,
        levels=contour_levels,
        cmap=TPRIME_COLORMAP,
        extend="both",
    )
    formatter = mticker.FuncFormatter(format_elapsed_tick_label)
    ax_curtain.xaxis.set_major_formatter(formatter)
    ax_series.xaxis.set_major_formatter(formatter)

    ax_curtain.axvline(current_x, linestyle="--", linewidth=1.8, color="black")
    ax_curtain.axhline(selected_altitude_km, linestyle="--", linewidth=1.4, color="black", alpha=0.85)
    ax_curtain.set_ylabel(f"{Z_NAME} [km]")
    ax_curtain.text(
        LIDAR_TIME_TEXT_X,
        LIDAR_TIME_TEXT_Y,
        format_time_value(current_time_value, time_start_value, frame_index=frame_index),
        transform=ax_curtain.transAxes,
        weight="bold",
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "lw": 0.67, "facecolor": "white", "edgecolor": "black"},
    )
    ax_curtain.grid(True, alpha=0.22)
    plt.setp(ax_curtain.get_xticklabels(), visible=False)

    ax_series.plot(x_vals, tprime_series, linewidth=1.8, color="black")
    ax_series.axvline(current_x, linestyle="--", linewidth=1.8, color="black")
    ax_series.set_xlabel("Time since start")
    ax_series.set_ylabel("T'")
    ax_series.grid(True, alpha=0.28)
    return mesh


def render_combined_frame(
    plotters,
    ds,
    time_index,
    lambda2_level,
    vorticity_clim,
    tprime_clim,
    w_clim,
    bottom_w_contour_levels,
    times_lidar,
    z_lidar,
    tprime_curtain,
    camera_position,
    lidar_x_val,
    lidar_y_val,
    selected_altitude_value,
    time_axis_limits=None,
    figure_size=(16.0, 12.0),
    dpi=150,
):
    scene_data = prepare_scene_data_for_time(ds=ds, time_index=time_index, lambda2_level=lambda2_level, stride=STRIDE)
    time_values = np.asarray(ds[TIME_NAME].values)
    time_start_value = time_values[0]
    current_time_value = time_values[int(time_index)]
    time_label_text = format_time_value(current_time_value, time_start_value)

    img_classic = render_pyvista_variant_frame(
        plotter=plotters["classic"],
        scene_data=scene_data,
        variant="classic",
        camera_position=camera_position,
        vorticity_clim=vorticity_clim,
        tprime_clim=tprime_clim,
        w_clim=w_clim,
        bottom_w_contour_levels=bottom_w_contour_levels,
        lidar_x_val=lidar_x_val,
        lidar_y_val=lidar_y_val,
        slice_z_val=selected_altitude_value,
        time_label_text=time_label_text,
    )
    img_xz = render_pyvista_variant_frame(
        plotter=plotters["xz_slice"],
        scene_data=scene_data,
        variant="xz_slice",
        camera_position=camera_position,
        vorticity_clim=vorticity_clim,
        tprime_clim=tprime_clim,
        w_clim=w_clim,
        bottom_w_contour_levels=bottom_w_contour_levels,
        lidar_x_val=lidar_x_val,
        lidar_y_val=lidar_y_val,
        slice_z_val=selected_altitude_value,
        time_label_text=time_label_text,
    )
    img_xy = render_pyvista_variant_frame(
        plotter=plotters["xy_slice"],
        scene_data=scene_data,
        variant="xy_slice",
        camera_position=camera_position,
        vorticity_clim=vorticity_clim,
        tprime_clim=tprime_clim,
        w_clim=w_clim,
        bottom_w_contour_levels=bottom_w_contour_levels,
        lidar_x_val=lidar_x_val,
        lidar_y_val=lidar_y_val,
        slice_z_val=selected_altitude_value,
        time_label_text=time_label_text,
    )

    fig = plt.figure(figsize=figure_size, dpi=dpi, constrained_layout=True)
    outer = fig.add_gridspec(2, 2)

    ax_ul = fig.add_subplot(outer[0, 0])
    ax_ul.imshow(img_classic)
    ax_ul.axis("off")

    sub = outer[0, 1].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.05)
    ax_urtop = fig.add_subplot(sub[0, 0])
    ax_urbot = fig.add_subplot(sub[1, 0], sharex=ax_urtop)

    mesh = plot_virtual_lidar_panels(
        ax_curtain=ax_urtop,
        ax_series=ax_urbot,
        times=times_lidar,
        z=z_lidar,
        tprime_curtain=tprime_curtain,
        current_time_value=current_time_value,
        selected_altitude_value=selected_altitude_value,
        tprime_clim=tprime_clim,
        frame_index=time_index,
    )
    if time_axis_limits is not None:
        ax_urtop.set_xlim(time_axis_limits)
        ax_urbot.set_xlim(time_axis_limits)
    cbar = fig.colorbar(mesh, ax=ax_urtop, pad=0.02, extend="both")
    cbar.set_label("T'")

    ax_ll = fig.add_subplot(outer[1, 0])
    ax_ll.imshow(img_xz)
    ax_ll.axis("off")

    ax_lr = fig.add_subplot(outer[1, 1])
    ax_lr.imshow(img_xy)
    ax_lr.axis("off")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def frame_png_path(frames_dir, time_index):
    return Path(frames_dir) / f"frame_{int(time_index):04d}.png"


def clear_frame_directory(frames_dir):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(exist_ok=True)
    if not CLEAR_EXISTING_FRAMES:
        return frames_dir
    for png_path in frames_dir.glob("*.png"):
        png_path.unlink()
    return frames_dir


def get_parallel_context():
    methods = mp.get_all_start_methods()
    if "fork" in methods:
        return mp.get_context("fork")
    return None


_frame_worker_state = None


def initialize_frame_worker(worker_config, pbar=None):
    global _frame_worker_state
    _frame_worker_state = dict(worker_config)
    _frame_worker_state["ds"] = apply_spatial_limits(xr.open_dataset(_frame_worker_state["nc_path"]))
    _frame_worker_state["pbar"] = pbar


def render_frame_png_worker(time_index):
    state = _frame_worker_state
    if state is None:
        raise RuntimeError("Animation worker was not initialized.")

    plotters = {
        "classic": pv.Plotter(off_screen=True, window_size=tuple(state["window_size_3d"])),
        "xz_slice": pv.Plotter(off_screen=True, window_size=tuple(state["window_size_3d"])),
        "xy_slice": pv.Plotter(off_screen=True, window_size=tuple(state["window_size_3d"])),
    }
    for plotter in plotters.values():
        plotter.set_background(state["background"])

    png_path = frame_png_path(state["frames_dir"], time_index)
    try:
        combined_img = render_combined_frame(
            plotters=plotters,
            ds=state["ds"],
            time_index=int(time_index),
            lambda2_level=state["lambda2_level"],
            vorticity_clim=tuple(state["vorticity_clim"]),
            tprime_clim=tuple(state["tprime_clim"]),
            w_clim=tuple(state["w_clim"]),
            bottom_w_contour_levels=np.asarray(state["bottom_w_contour_levels"], dtype=float),
            times_lidar=np.asarray(state["times_lidar"]),
            z_lidar=np.asarray(state["z_lidar"]),
            tprime_curtain=np.asarray(state["tprime_curtain"]),
            camera_position=state["camera_position"],
            lidar_x_val=state["lidar_x_val"],
            lidar_y_val=state["lidar_y_val"],
            selected_altitude_value=state["selected_altitude_value"],
            time_axis_limits=tuple(state["time_axis_limits"]),
            figure_size=tuple(state["figure_size"]),
            dpi=state["dpi"],
        )
        imageio.imwrite(png_path, combined_img)
    finally:
        for plotter in plotters.values():
            plotter.close()

    pbar = state.get("pbar")
    if pbar is not None:
        plt_helper.show_progress(pbar["progress_counter"], pbar["lock"], pbar["stime"], pbar["ntasks"])
    return str(png_path)


def render_frames_serial(time_indices, worker_config):
    plotters = {
        "classic": pv.Plotter(off_screen=True, window_size=tuple(worker_config["window_size_3d"])),
        "xz_slice": pv.Plotter(off_screen=True, window_size=tuple(worker_config["window_size_3d"])),
        "xy_slice": pv.Plotter(off_screen=True, window_size=tuple(worker_config["window_size_3d"])),
    }
    for plotter in plotters.values():
        plotter.set_background(worker_config["background"])

    stime_local = time.time()
    try:
        for i, ti in enumerate(time_indices, start=1):
            combined_img = render_combined_frame(
                plotters=plotters,
                ds=worker_config["ds"],
                time_index=int(ti),
                lambda2_level=worker_config["lambda2_level"],
                vorticity_clim=tuple(worker_config["vorticity_clim"]),
                tprime_clim=tuple(worker_config["tprime_clim"]),
                w_clim=tuple(worker_config["w_clim"]),
                bottom_w_contour_levels=np.asarray(worker_config["bottom_w_contour_levels"], dtype=float),
                times_lidar=np.asarray(worker_config["times_lidar"]),
                z_lidar=np.asarray(worker_config["z_lidar"]),
                tprime_curtain=np.asarray(worker_config["tprime_curtain"]),
                camera_position=worker_config["camera_position"],
                lidar_x_val=worker_config["lidar_x_val"],
                lidar_y_val=worker_config["lidar_y_val"],
                selected_altitude_value=worker_config["selected_altitude_value"],
                time_axis_limits=tuple(worker_config["time_axis_limits"]),
                figure_size=tuple(worker_config["figure_size"]),
                dpi=worker_config["dpi"],
            )
            png_path = frame_png_path(worker_config["frames_dir"], ti)
            imageio.imwrite(png_path, combined_img)
            elapsed = time.time() - stime_local
            print(f"[{i:>4d}/{len(time_indices)}] wrote {png_path} in {elapsed/60:.2f} min total")
    finally:
        for plotter in plotters.values():
            plotter.close()


def render_frames_parallel(time_indices, worker_config, ncpus):
    ctx = get_parallel_context()
    if ctx is None:
        print("Parallel frame generation requires the 'fork' multiprocessing start method. Falling back to serial rendering.")
        render_frames_serial(time_indices, worker_config)
        return

    ncpus = max(1, min(int(ncpus), len(time_indices)))
    manager = ctx.Manager()
    pbar = {
        "progress_counter": manager.Value("i", 0),
        "lock": manager.Lock(),
        "stime": time.time(),
        "ntasks": len(time_indices),
    }

    print(f"[i]  CPUs available: {mp.cpu_count()}")
    print(f"[i]  CPUs for rendering: {ncpus}")

    worker_init_config = dict(worker_config)
    worker_init_config.pop("ds", None)
    with ctx.Pool(processes=ncpus, initializer=initialize_frame_worker, initargs=(worker_init_config, pbar)) as pool:
        for _ in pool.imap_unordered(render_frame_png_worker, time_indices):
            pass


def build_mp4_from_pngs(frames_dir, outfile, fps=10):
    outfile = Path(outfile)
    plt_helper.create_animation(str(frames_dir), outfile.name, fps=fps)
    generated = Path(frames_dir) / outfile.name
    if generated.resolve() != outfile.resolve():
        shutil.move(str(generated), str(outfile))
    return outfile


def resolve_nc_path(simulation_name: str) -> Path:
    simulation_name = os.path.basename(os.path.normpath(simulation_name))
    nc_path = DATA_ROOT / simulation_name / "cube.nc"
    if not nc_path.exists():
        raise FileNotFoundError(f"Could not find cube.nc for simulation '{simulation_name}' at {nc_path}")
    return nc_path


def prepare_worker_config(nc_path: Path):
    ds = xr.open_dataset(nc_path)
    ds = apply_spatial_limits(ds)
    print(ds)

    print()
    print("Applied spatial limits:")
    print("x:", normalize_limits(X_LIMITS))
    print("y:", normalize_limits(Y_LIMITS))
    print("z:", normalize_limits(Z_LIMITS))

    x_diag, y_diag, z_diag, arrays_diag = load_time_slice(
        ds,
        time_index=TIME_INDEX,
        stride=STRIDE,
        extra_var_names=[DENSITY_NAME, THETA_NAME, EXNER_NAME],
    )
    density_diag = arrays_diag[DENSITY_NAME]
    theta_diag = arrays_diag[THETA_NAME]
    exner_diag = arrays_diag[EXNER_NAME]

    temperature_poisson_diag = compute_temperature_from_theta_exner(theta_diag, exner_diag)
    pressure_diag = compute_pressure_from_exner(exner_diag)
    temperature_ideal_diag = compute_temperature_from_ideal_gas(density_diag, pressure_diag)
    delta_t_diag = temperature_ideal_diag - temperature_poisson_diag

    print()
    print("Thermodynamic check at time index", TIME_INDEX)
    print("Temperature from theta/exner: min/max =", float(np.nanmin(temperature_poisson_diag)), float(np.nanmax(temperature_poisson_diag)))
    print("Temperature from ideal gas : min/max =", float(np.nanmin(temperature_ideal_diag)), float(np.nanmax(temperature_ideal_diag)))
    print("Mean abs difference [K]   =", float(np.nanmean(np.abs(delta_t_diag))))
    print("Max abs difference [K]    =", float(np.nanmax(np.abs(delta_t_diag))))
    print()
    print("The script uses T = theta_total * exner_total as the primary temperature definition.")

    x_ref, y_ref, z_ref, arrays_ref = load_time_slice(
        ds,
        SINGLE_LAMBDA2_REFERENCE_TIME_INDEX,
        stride=STRIDE,
        extra_var_names=[DENSITY_NAME, THETA_NAME, EXNER_NAME],
    )
    u_ref = arrays_ref[U_NAME]
    v_ref = arrays_ref[V_NAME]
    w_ref = arrays_ref[W_NAME]
    theta_ref = arrays_ref[THETA_NAME]
    exner_ref = arrays_ref[EXNER_NAME]

    temperature_ref = compute_temperature_from_theta_exner(theta_ref, exner_ref)
    tprime_ref = compute_tprime_from_temperature(temperature_ref)
    lambda2_ref, vort_scalars_ref, _, _, vort_name_ref = compute_lambda2_and_vorticity(u_ref, v_ref, w_ref, x_ref, y_ref, z_ref)

    if SMOOTH_SIGMA and SMOOTH_SIGMA > 0:
        if gaussian_filter is None:
            raise ImportError("scipy is required for smoothing; install scipy or set smooth_sigma = 0")
        lambda2_ref = gaussian_filter(lambda2_ref, sigma=SMOOTH_SIGMA)
        vort_scalars_ref = gaussian_filter(vort_scalars_ref, sigma=SMOOTH_SIGMA)
        tprime_ref = gaussian_filter(tprime_ref, sigma=SMOOTH_SIGMA)
        w_ref_for_clim = gaussian_filter(w_ref, sigma=SMOOTH_SIGMA)
    else:
        w_ref_for_clim = w_ref

    lambda2_level = choose_single_lambda2_level(
        lambda2=lambda2_ref,
        negative_only=NEGATIVE_LAMBDA2_ONLY,
        percentile=SINGLE_LAMBDA2_PERCENTILE,
        manual_level=MANUAL_LAMBDA2_LEVEL,
    )
    vorticity_clim = choose_clim(vort_scalars_ref, percentile_range=VORTICITY_PERCENTILE_RANGE)
    tprime_clim = choose_symmetric_clim(tprime_ref, percentile=TPRIME_PERCENTILE)
    w_clim = choose_symmetric_clim(w_ref_for_clim, percentile=W_SLICE_PERCENTILE)
    bottom_w_contour_levels = choose_symmetric_contour_levels(w_clim, count=BOTTOM_W_CONTOUR_COUNT, exclude_zero=EXCLUDE_ZERO_FROM_BOTTOM_W_CONTOURS)

    print("Reference time index:", SINGLE_LAMBDA2_REFERENCE_TIME_INDEX)
    print("Chosen λ2 level:", lambda2_level)
    print(f"{vort_name_ref} color limits:", vorticity_clim)
    print("T' color limits:", tprime_clim)
    print("w color limits:", w_clim)
    print("Bottom w contour levels:", bottom_w_contour_levels)

    times_lidar, z_lidar, tprime_curtain, _, _, lidar_x_val, lidar_y_val = load_virtual_lidar_tprime(
        ds,
        x_value=VIRTUAL_LIDAR_LOCATION[0],
        y_value=VIRTUAL_LIDAR_LOCATION[1],
    )
    tprime_curtain_clim = choose_symmetric_clim(tprime_curtain, percentile=TPRIME_CURTAIN_PERCENTILE)
    tprime_clim = (
        -max(abs(tprime_clim[0]), abs(tprime_curtain_clim[0]), abs(tprime_curtain_clim[1])),
        max(abs(tprime_clim[1]), abs(tprime_curtain_clim[0]), abs(tprime_curtain_clim[1])),
    )
    _, timeseries_altitude_value_used = choose_altitude(
        z_lidar,
        altitude_index=TIMESERIES_ALTITUDE_INDEX,
        altitude_value=TIMESERIES_ALTITUDE_VALUE,
    )

    print("Virtual lidar coordinates:", (lidar_x_val, lidar_y_val))
    print("Selected altitude value:", timeseries_altitude_value_used)
    print("T' curtain shape:", tprime_curtain.shape)
    print("Updated common T' color limits:", tprime_clim)

    return {
        "ds": ds,
        "lambda2_level": float(lambda2_level),
        "vorticity_clim": tuple(vorticity_clim),
        "tprime_clim": tuple(tprime_clim),
        "w_clim": tuple(w_clim),
        "bottom_w_contour_levels": np.asarray(bottom_w_contour_levels, dtype=float),
        "times_lidar": np.asarray(times_lidar),
        "z_lidar": np.asarray(z_lidar),
        "tprime_curtain": np.asarray(tprime_curtain),
        "lidar_x_val": None if lidar_x_val is None else float(lidar_x_val),
        "lidar_y_val": None if lidar_y_val is None else float(lidar_y_val),
        "selected_altitude_value": float(timeseries_altitude_value_used),
        "camera_position": None,
    }


def run_animation(simulation_name: str):
    simulation_name = os.path.basename(os.path.normpath(simulation_name))
    nc_path = resolve_nc_path(simulation_name)
    print(f"[i]  Using cube file: {nc_path}")

    prepared = prepare_worker_config(nc_path)
    ds = prepared["ds"]
    frames_dir = clear_frame_directory(Path(f"frames_cube_lid_{simulation_name}"))
    outfile = Path(f"cube_lid_{simulation_name}.mp4")
    time_indices = select_time_indices(ds)
    all_time_values = np.asarray(ds[TIME_NAME].values)
    plotted_time_values = all_time_values[time_indices]
    time_axis_limits = (
        float(np.asarray(compute_elapsed_seconds(plotted_time_values[0], all_time_values[0])).reshape(-1)[0]),
        float(np.asarray(compute_elapsed_seconds(plotted_time_values[-1], all_time_values[0])).reshape(-1)[0]),
    )
    if time_axis_limits[0] == time_axis_limits[1]:
        time_axis_limits = (time_axis_limits[0], time_axis_limits[0] + 1.0)

    print()
    print(f"Rendering {len(time_indices)} of {int(ds.sizes[TIME_NAME])} available time indices.")
    print("Applied time-index limits:", TIME_LIMITS)
    print(
        "Virtual lidar x-limits use plotted time range:",
        f"{format_elapsed_hms(time_axis_limits[0])} to {format_elapsed_hms(time_axis_limits[1])}",
    )

    worker_config = {
        "nc_path": str(nc_path),
        "ds": ds,
        "frames_dir": str(frames_dir),
        "window_size_3d": tuple(WINDOW_SIZE_3D),
        "background": BACKGROUND,
        "lambda2_level": prepared["lambda2_level"],
        "vorticity_clim": prepared["vorticity_clim"],
        "tprime_clim": prepared["tprime_clim"],
        "w_clim": prepared["w_clim"],
        "bottom_w_contour_levels": prepared["bottom_w_contour_levels"],
        "times_lidar": prepared["times_lidar"],
        "z_lidar": prepared["z_lidar"],
        "tprime_curtain": prepared["tprime_curtain"],
        "camera_position": prepared["camera_position"],
        "lidar_x_val": prepared["lidar_x_val"],
        "lidar_y_val": prepared["lidar_y_val"],
        "selected_altitude_value": prepared["selected_altitude_value"],
        "time_axis_limits": tuple(time_axis_limits),
        "figure_size": tuple(COMBINED_FIGURE_SIZE),
        "dpi": int(COMBINED_DPI),
    }

    if PARALLEL_FRAME_GENERATION and len(time_indices) > 1:
        render_frames_parallel(time_indices, worker_config, ncpus=ANIMATION_NCPUS)
    else:
        render_frames_serial(time_indices, worker_config)

    outfile = build_mp4_from_pngs(frames_dir, outfile, fps=FPS)
    ds.close()
    print(f"Saved animation to {outfile.resolve()}")
    print(f"Saved PNG frames to {frames_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Render the cube lidar multiview MP4 for one simulation.")
    parser.add_argument("simulation_directory", help="Simulation name below /scratch/b/b309199, e.g. darwin_240718_400m_schu")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_headless_rendering()
    run_animation(args.simulation_directory)


if __name__ == "__main__":
    mp.freeze_support()
    main()
