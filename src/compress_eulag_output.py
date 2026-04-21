import sys
import os
import glob
import time
import xarray as xr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def compress_eulag_output(sim_dir: str):
    nc_files = sorted(list(Path(sim_dir).glob("*.nc")))
    print(f"NC Files: {nc_files}")
    stime0 = time.time()
    for fpath in nc_files:
        stime = time.time()
        size_before = os.path.getsize(fpath) / 1e9
        print(f"File: {os.path.basename(fpath)}, Size: {size_before:.3f} GB")
        
        with xr.open_dataset(fpath) as ds:
            if is_compressed(ds):
                print(f"{os.path.basename(fpath)} is already compressed -> skipping.")
                continue
            else:
                encoding = {
                    var: {"zlib": True, "complevel": 3, "shuffle": True}
                    for var in ds.data_vars
                }
                tmp_path = fpath.with_suffix(".tmp.nc")
                ds.to_netcdf(tmp_path, encoding=encoding, format="NETCDF4", mode="w")
        
        os.replace(tmp_path, fpath)  # Atomic replacement
        size_after = os.path.getsize(fpath) / 1e9
        print(f"-> Compressed from {size_before:.3f} GB to {size_after:.3f} GB in {(time.time()-stime)/60:.2f} min")
    
    print(f"Compression of {sim_dir} completed in {(time.time()-stime0)/60:.2f} min")


def is_compressed(ds):
    for var in ds.data_vars:
        encoding = ds[var].encoding
        if encoding.get("zlib", False):
            return True  # At least one variable is compressed
    return False


if __name__ == '__main__':
    compress_eulag_output(sys.argv[1])