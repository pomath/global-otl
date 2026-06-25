"""MPI script templates for LoadDef subprocess calls.

Each template is a Python script string with {placeholders} that gets
filled in by runner.py, written to a temp file, and executed via mpirun.
The scripts set up the LoadDef-compatible directory structure and call
the appropriate LoadDef functions.
"""

# Template for computing Love Numbers from an Earth model.
# Placeholders: loaddef_root, planet_model_path, output_dir
LOVE_NUMBERS_SCRIPT = """\
#!/usr/bin/env python
from __future__ import print_function
from mpi4py import MPI
import sys, os

sys.path.insert(0, "{loaddef_root}")
os.makedirs("{output_dir}/Love_Numbers/LLN", exist_ok=True)
os.makedirs("{output_dir}/Love_Numbers/PLN", exist_ok=True)
os.makedirs("{output_dir}/Love_Numbers/STR", exist_ok=True)
os.makedirs("{output_dir}/Love_Numbers/SHR", exist_ok=True)

from LOADGF.LN import compute_love_numbers

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# compute_love_numbers writes to ../output/Love_Numbers/ relative to cwd
# We set cwd so ../output/ resolves to our output_dir
os.chdir("{working_dir}")

planet_model = "{planet_model_path}"
file_ext = "{earth_model}.txt"

if rank == 0:
    print("Computing Love Numbers for", file_ext)
    compute_love_numbers.main(planet_model, rank, comm, size, file_out=file_ext)
else:
    compute_love_numbers.main(planet_model, rank, comm, size, file_out=file_ext)
"""

# Template for computing Green's Functions from Love Numbers.
# Placeholders: loaddef_root, lln_file, output_dir, earth_model, working_dir
GREENS_FUNCTIONS_SCRIPT = """\
#!/usr/bin/env python
from __future__ import print_function
from mpi4py import MPI
import sys, os

sys.path.insert(0, "{loaddef_root}")
os.makedirs("{output_dir}/Greens_Functions", exist_ok=True)

from LOADGF.GF import compute_greens_functions

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

os.chdir("{working_dir}")

lln_file = "{lln_file}"
file_out = "{earth_model}.txt"

if rank == 0:
    print("Computing Greens Functions for", file_out)
    compute_greens_functions.main(lln_file, rank, comm, size, grn_out=file_out)
else:
    compute_greens_functions.main(lln_file, rank, comm, size, grn_out=file_out)

comm.Barrier()
"""

# Template for OTL convolution (LoadDef current version).
# The new LoadDef API has load_convolution.main() without MPI params;
# MPI parallelization is over stations in the outer loop.
# Placeholders: loaddef_root, grn_file, loadfile_directory, loadfile_prefix,
#               lsmask_file, sta_file, rfm, earth_model, output_dir, working_dir
CONVOLUTION_SCRIPT = """\
#!/usr/bin/env python
from __future__ import print_function
from mpi4py import MPI
import sys, os
import numpy as np

sys.path.insert(0, "{loaddef_root}")
os.makedirs("{output_dir}/Convolution", exist_ok=True)

from CONVGF.CN import load_convolution
from CONVGF.utility import read_station_file
from CONVGF.utility import read_lsmask

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

os.chdir("{working_dir}")

rfm = "{rfm}"
pmod = "{earth_model}"
grn_file = "{grn_file}"
norm_flag = False
loadfile_directory = "{loadfile_directory}"
loadfile_prefix = "{loadfile_prefix}"
loadfile_format = "nc"
regular = True
ldens = 1030.0
lsmask_type = 1
lsmask_file = "{lsmask_file}"
sta_file = "{sta_file}"
# Only rank 0 does the work; workers idle (single-station mode)
if rank == 0:
    # Ensure ../output/Convolution/ exists relative to cwd (working/)
    os.makedirs("../output/Convolution", exist_ok=True)
    # Read land-sea mask
    lslat, lslon, lsmask = read_lsmask.main(lsmask_file)
    print("Finished reading land-sea mask.")

    # Read station file
    lat, lon, sta = read_station_file.main(sta_file)

    # Handle single vs multiple stations
    if isinstance(lat, float):
        numel = 1
    else:
        numel = len(lat)

    # Find load files
    load_files = []
    if os.path.isdir(loadfile_directory):
        for mfile in os.listdir(loadfile_directory):
            if mfile.startswith(loadfile_prefix):
                load_files.append(loadfile_directory + "/" + mfile)
    if not load_files:
        print("ERROR: No load files found with prefix", loadfile_prefix, "in", loadfile_directory)
        sys.exit(1)
    load_files = np.asarray(sorted(load_files))
    print("Found", len(load_files), "load files.")

    for jj in range(0, numel):
        if numel == 1:
            my_sta, my_lat, my_lon = sta, lat, lon
        else:
            my_sta, my_lat, my_lon = sta[jj], lat[jj], lon[jj]
        try:
            my_sta = my_sta.decode()
        except:
            pass
        my_sta = str(my_sta)
        print("Starting station:", my_sta)

        # cnv_out is just the suffix; LoadDef prepends ../output/Convolution/cn_OceanOnly_
        cnv_out = my_sta + "_" + rfm + "_" + loadfile_prefix + "_" + pmod + ".txt"

        eamp, epha, namp, npha, vamp, vpha = load_convolution.main(
            grn_file, norm_flag, load_files, loadfile_format, regular,
            lslat, lslon, lsmask, lsmask_type,
            my_lat, my_lon, my_sta, cnv_out,
            load_density=ldens)

        print("Station", my_sta, "complete.")

    print("Convolution complete.")

comm.Barrier()
"""
