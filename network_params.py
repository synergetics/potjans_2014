# !/usr/bin/env python

import numpy as np

# Contains:
# - network parameters
# - single-neuron parameters
# - stimulus parameters

###################################################
###       Network parameters    ###
###################################################

#  area of network in mm^2; scales numbers of neurons
#  use 1 for the full-size network (77,169 neurons)
area = 0.1

#  whether to use full-scale in-degrees when downscaling the number of neurons
#  When preserve_K is false, the full-scale connection probabilities are used.
#  Note that this produces different dynamics compared to the original model.
preserve_K = True

layers = ['L23', 'L4', 'L5', 'L6']

full_scale_n_neurons = np.array([[20683,   #  layer 2/3 e
                                  5834],   #  layer 2/3 i
                                 [21915,   #  layer 4 e
                                  5479],   #  layer 4 i
                                 [4850,    #  layer 5 e
                                  1065],   #  layer 5 i
                                 [14395,   #  layer 6 e
                                  2948]]   #  layer 6 i
                                )

#  mean EPSP amplitude (mV) for all connections except L4e->L2/3e
PSP_e = .15
#  mean EPSP amplitude (mv) for L4e->L2/3e connections
#  see p. 801 of the paper, second paragraph under 'Model Parameterization',
#  and the caption to Supplementary Fig. 7
PSP_e_23_4 = PSP_e * 2
#  standard deviation of PSC amplitudes relative to mean PSC amplitudes
PSC_rel_sd = 0.1
#  IPSP amplitude relative to EPSP amplitude
g = -4.

# Probabilities for >=1 connection between neurons in the given populations
# Columns correspond to source populations; rows to target populations
# Source                2/3e      2/3i      4e       4i      5e       5i        6e      6i
conn_probs = np.array([[0.1009,  0.1689,  0.0437,  0.0818,  0.0323,  0.,      0.0076,  0.    ],
                       [0.1346,  0.1371,  0.0316,  0.0515,  0.0755,  0.,      0.0042,  0.    ],
                       [0.0077,  0.0059,  0.0497,  0.135,   0.0067,  0.0003,  0.0453,  0.    ],
                       [0.0691,  0.0029,  0.0794,  0.1597,  0.0033,  0.,      0.1057,  0.    ],
                       [0.1004,  0.0622,  0.0505,  0.0057,  0.0831,  0.3726,  0.0204,  0.    ],
                       [0.0548,  0.0269,  0.0257,  0.0022,  0.06,    0.3158,  0.0086,  0.    ],
                       [0.0156,  0.0066,  0.0211,  0.0166,  0.0572,  0.0197,  0.0396,  0.2252],
                       [0.0364,  0.001,   0.0034,  0.0005,  0.0277,  0.008,   0.0658,  0.1443]]
                      )

# Mean dendritic delays for excitatory and inhibitory transmission (ms)
delays = np.array([1.5, 0.75])
# Standard deviation relative to mean delays
delay_rel_sd = 0.5
# Connection pattern used in connection calls connecting populations
conn_dict = {'rule': 'fixed_total_number'}
# Weight distribution of connections between populations
weight_dict_exc = {'distribution': 'normal_clipped', 'low': 0.0}
weight_dict_inh = {'distribution': 'normal_clipped', 'high': 0.0}
# Delay distribution of connections between populations
delay_dict = {'distribution': 'normal_clipped', 'low': 0.1}
# Default synapse dictionary
syn_dict = {'model': 'static_synapse'}


###################################################
###          Single-neuron parameters   ###
###################################################


neuron_model = "iaf_psc_exp"    # Neuron model. For PSP-to-PSC conversion to
                                # Be correct, synapses should be current-based
                                # With an exponential time course
Vm0_mean = -58.0                # Mean of initial membrane potential (mV)
Vm0_std = 10.0                  # Std of initial membrane potential (mV)

# Neuron model parameters
model_params = {'tau_m': 10.,        # Membrane time constant (ms)
                'tau_syn_ex': 0.5,   # Excitatory synaptic time constant (ms)
                'tau_syn_in': 0.5,   # Inhibitory synaptic time constant (ms)
                't_ref': 2.,         # Absolute refractory period (ms)
                'E_L': -65.,         # Resting membrane potential (mV)
                'V_th': -50.,        # Spike threshold (mV)
                'C_m': 250.,         # Membrane capacitance (pF)
                'V_reset': -65.      # Reset potential (mV)
               }


###################################################
###           Stimulus parameters   ###
###################################################

# Rate of background Poisson input at each external input synapse (spikes/s)
bg_rate = 8.
# DC amplitude at each external input synapse (pA)
# This is relevant for reproducing Potjans & Diesmann (2012) Fig. 7.
dc_amplitude = 0.
# In-degrees for background input
K_bg = np.array([[1600,  # 2/3e
                1500],  # 2/3i
               [2100,   # 4e
                1900],  # 4i
               [2000,   # 5e
                1900],  # 5i
               [2900,   # 6e
                2100]]  # 6i
              )

# Optional additional thalamic input (Poisson)
# Set n_thal to 0 to avoid this input.
# For producing Potjans & Diesmann (2012) Fig. 10, n_thal=902 was used.
# Note that the thalamic rate here reproduces the simulation results
# Shown in the paper, and differs from the rate given in the text.
n_thal = 0.          # Size of thalamic population
th_start = 700.     # Onset of thalamic input (ms)
th_duration = 10.   # Duration of thalamic input (ms)
th_rate = 120.      # Rate of thalamic neurons (spikes/s)
PSP_ext = 0.15      # Mean EPSP amplitude (mV) for external input

# Connection probabilities for thalamic input
C_th = np.array([[0.0,       # 2/3e
                  0.0],       # 2/3i
                 [0.0983,     # 4e
                  0.0619],    # 4i
                 [0.0,        # 5e
                  0.0],       # 5i
                 [0.0512,     # 6e
                  0.0196]]    # 6i
                )

# Mean delay of thalamic input (ms)
delay_th = 1.5
# Standard deviation relative to mean delay of thalamic input
delay_th_rel_sd = 0.5

