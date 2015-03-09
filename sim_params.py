#!usrbinenv =


# Contains:
# - simulation parameters
# - recording parameters

###################################################
###       Simulation parameters   ###
###################################################

run_mode = 'test'     # (test) for writing files to
                      # Directory containing microcircuit.sli
                      # (production) for writing files
                      # To a chosen absolute path.

t_sim = 1000.0      # Simulated time (ms)
dt = 0.1            # Simulation step (ms). ault is 0.1 ms.
allgather = True    # Communication protocol

# Master seed for random number generators
# Actual seeds will be master_seed ... master_seed + 2*n_vp
#  ==>> different master seeds must be spaced by at least 2*n_vp + 1
# See Gewaltig et al. (2012) for details
master_seed = 123456    # Changes rng_seeds and grng_seed

n_mpi_procs = 1             # Number of MPI processes

n_threads_per_proc = 8      # Number of threads per MPI process
                            # Use for instance 24 for a full-scale simulation


n_vp = n_threads_per_proc * n_mpi_procs   # Number of virtual processes
                                          # This should be an integer multiple of
                                          # The number of MPI processes
                                          # See Morrison et al. (2005) Neural Comput
# Walltime = (8:0:0)    # Walltime for simulation

memory = '500mb'    # Total memory
                    # Use for instance 4gb for a full-scale simulation


###################################################
###       Recording parameters    ###
###################################################

overwrite_existing_files =  True

# Whether to record spikes from a fixed fraction of neurons in each population
# If false, a fixed number of neurons is recorded in each population.
# Record_fraction_neurons_spikes True with f_rec_spikes 1. records all spikes
record_fraction_neurons_spikes =  True

if record_fraction_neurons_spikes:
  frac_rec_spikes = 0.1
else:
  n_rec_spikes = 100

# Whether to record voltage from a fixed fraction of neurons in each population
record_fraction_neurons_voltage =  True

if record_fraction_neurons_voltage:
  frac_rec_voltage = 0.02
else:
  n_rec_voltage = 20.

# Whether to write any recorded cortical spikes to file
save_cortical_spikes =  True

# Whether to write any recorded membrane potentials to file
save_voltages =  True

# Whether to record thalamic spikes (only used when n_thal in
# Network_params.sli is nonzero)
record_thalamic_spikes =  True

# Whether to write any recorded thalamic spikes to file
save_thalamic_spikes =  True

# Name of file to which to write global IDs
GID_filename =  'population_GIDs.dat'

# Stem for spike detector file labels
spike_detector_label =  'spikes'

# Stem for voltmeter file labels
voltmeter_label =  'voltages'

# Stem for thalamic spike detector file labels
th_spike_detector_label =  'th_spikes'

# File name for standard output
std_out =  'output.txt'

# File name for error output
error_out =  'errors.txt'



