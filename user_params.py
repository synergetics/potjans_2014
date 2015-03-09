#!/usr/bin/env python

from sim_params import *

if run_mode == 'production':
  # absolute path to which the output files should be written
  output_path = '.'

# path to the mpi shell script
# can be left out if set beforehand
mpi = './my_mpi_script.sh'

# path to NEST
nest_path = '/opt/bin/nest'

