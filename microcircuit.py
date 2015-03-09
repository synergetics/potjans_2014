# !/usr/bin/env python

import sys
sys.path.append('/opt/lib/python2.7/site-packages/')

import math
import numpy as np
import pylab
import nest
import nest.raster_plot
import nest.topology as tp
import logging as log

from network_params import *
from sim_params import *
from user_params import *

# Implementation of the multi-layered local cortical network model by

# Potjans, Tobias C., and Markus Diesmann. "The cell-type specific
# cortical microcircuit: relating structure and activity in a full-scale
# spiking network model." Cerebral Cortex (2014): bhs358.

# Uses user_params.sli, sim_params.sli, and network_params.sli

# function definitions:
# - CheckParameters
# - PrepareSimulation
# - DerivedParameters
# - CreateNetworkNodes
# - WriteGIDstoFile
# - ConnectNetworkNodes


# global variables, lol
n_layers = None
n_pops_per_layer = None
normal_rdvs = None

PSC_e = None
PSC_e_23_4 = None
PSP_i = None
PSC_i = None
PSC_ext = None
PSC_array = None
PSC_sd = None
PSC_th_sd = None
delays_sd = None
delay_th_sd = None
n_neurons_rec_spikes = None
n_neurons_rec_voltage = None
n_neurons = None

neuron_subnet_GIDs = None
spike_detector_GIDs = None
voltmeter_GIDs = None
poisson_GIDs = None
dc_GIDs = None
th_neuron_subnet_GID = None
th_poisson_GID = None
th_spike_detector_GID = None


def GetLocalNodes(subnets):
  if type(subnets) is not tuple:
    subnets = tuple(subnets)
  return nest.GetNodes(subnets, local_only=True)


def GetGlobalNodes(subnets):
  if type(subnets) is not tuple:
    subnets = tuple(subnets)
  return nest.GetNodes(subnets, local_only=False)


def CheckParameters():
  global n_layers
  global n_pops_per_layer

  if neuron_model != 'iaf_psc_exp':
    if nest.Rank() != 0:
      log.warn('Unexpected neuron type: script is tuned to "iaf_psc_exp" neurons.')

  # number of layers
  n_layers = len(full_scale_n_neurons)
  # number of populations in each layer
  n_pops_per_layer = np.shape(full_scale_n_neurons)[1]

  # if np.shape(conn_probs)[0] != n_layers*n_pops_per_layer or \
  #   np.shape(conn_probs)[1] != n_layers*n_pops_per_layer:
  #   raise ValueError('conn_probs_dimensions')

  if record_fraction_neurons_spikes:
    if frac_rec_spikes > 1:
      raise ValueError('frac_rec_spikes')
  else:
    if n_rec_spikes > area * min(map(min, full_scale_n_neurons)):
      raise ValueError('n_rec_spikes')

  if record_fraction_neurons_voltage:
    if frac_rec_voltage > 1:
      raise ValueError('frac_rec_voltage')
  else:
    if n_rec_voltage > area * min(map(min, full_scale_n_neurons)):
      raise ValueError('n_rec_voltage')


def PrepareSimulation():
  global normal_rdvs

  nest.ResetKernel()

  nest.SetKernelStatus({
    'resolution': dt,
    'total_num_virtual_procs': n_vp,
    'communicate_allgather': allgather,
    'overwrite_files': overwrite_existing_files,
    'rng_seeds': range(master_seed, master_seed + n_vp),    # local RNG seeds
    'grng_seed': master_seed + n_vp                         # global RNG seed
  })
  if run_mode == 'production':
    nest.SetKernelStatus({'data_path': output_path})

  seed_offset =  master_seed + n_vp
  normal_rdvs = [np.random.RandomState(s) for s in range(seed_offset, seed_offset + n_vp)]


def DerivedParameters():
  global PSC_e
  global PSC_e_23_4
  global PSP_i
  global PSC_i
  global PSC_ext
  global PSC_array
  global PSC_sd
  global PSC_th_sd
  global delays_sd
  global delay_th_sd
  global n_neurons_rec_spikes
  global n_neurons_rec_voltage
  global n_neurons

  # compute numbers of neurons for the given surface area
  n_neurons = np.array(map(lambda x: map(int, x), full_scale_n_neurons*area))

  m = model_params
  # compute PSC amplitude from PSP amplitude
  # factor for transforming PSP amplitude to PSC amplitude
  re = m['tau_m'] / m['tau_syn_ex']
  de = m['tau_syn_ex'] - m['tau_m']
  ri = m['tau_m'] / m['tau_syn_in']
  di = m['tau_syn_in'] - m['tau_m']

  PSC_e_over_PSP_e = (((m['C_m'])**(-1)*m['tau_m']*m['tau_syn_ex']/de*(re**(m['tau_m']/de)-re**(m['tau_syn_ex']/de)))**(-1))

  PSC_i_over_PSP_i = (((m['C_m'])**(-1)*m['tau_m']*m['tau_syn_in']/di*(ri**(m['tau_m']/di)-ri**(m['tau_syn_in']/di)))**(-1))

  PSC_e =  PSC_e_over_PSP_e * PSP_e
  PSC_e_23_4 =  PSC_e_over_PSP_e * PSP_e_23_4
  PSP_i =  PSP_e * g
  PSC_i =  PSC_i_over_PSP_i * PSP_i

  # PSC amplitude for all external input
  PSC_ext =  PSC_e_over_PSP_e * PSP_ext

  # array of synaptic current amplitudes
  PSC_array = np.tile(np.array([PSC_e, PSC_i]), (4,2,4,1))
  PSC_array[0, 0, 1, 0] = PSC_e_23_4

  # standard deviations of synaptic current amplitudes
  PSC_sd =  np.array([PSC_e, PSC_i]) * PSC_rel_sd
  PSC_th_sd =  PSC_ext * PSC_rel_sd

  # standard deviations of delays
  delays_sd =  delays * delay_rel_sd
  delay_th_sd =  delay_th * delay_th_rel_sd

  # numbers of neurons from which to record spikes and membrane potentials
  if record_fraction_neurons_spikes:
    n_neurons_rec_spikes = frac_rec_spikes*n_neurons
  else:
    n_neurons_rec_spikes = np.tile(n_rec_spikes, (n_layers, n_pops_per_layer, 1))

  if record_fraction_neurons_voltage:
    n_neurons_rec_voltage = frac_rec_voltage*n_neurons
  else:
    n_neurons_rec_voltage = np.tile(n_rec_voltage, (n_layers, n_pops_per_layer, 1))


def CreateNetworkNodes():
  global neuron_subnet_GIDs
  global spike_detector_GIDs
  global voltmeter_GIDs
  global poisson_GIDs
  global dc_GIDs
  global th_neuron_subnet_GID
  global th_poisson_GID
  global th_spike_detector_GID

  # create and configure neurons
  nest.SetDefaults(neuron_model, model_params)
  # arrays of GIDs:
  # neuron subnets
  neuron_subnet_GIDs = np.tile(0, (n_layers, n_pops_per_layer, 1))
  # spike detectors
  spike_detector_GIDs = np.tile(0, (n_layers, n_pops_per_layer, 1))
  # voltmeters
  voltmeter_GIDs = np.tile(0, (n_layers, n_pops_per_layer, 1))
  # Poisson generators
  poisson_GIDs = np.tile(0, (n_layers, n_pops_per_layer, 1))
  # DC generators
  dc_GIDs  = np.tile(0, (n_layers, n_pops_per_layer, 1))

  for layer_index in xrange(n_layers):
    nest.ChangeSubnet((0,)) # change to the root node

    layer_subnet = nest.Create('subnet')

    for population_index in xrange(n_pops_per_layer):
      nest.ChangeSubnet(layer_subnet)

      population_subnet = nest.Create('subnet')
      nest.ChangeSubnet(population_subnet)

      # create neurons
      neuron_subnet = nest.Create('subnet')
      nest.ChangeSubnet(neuron_subnet)
      neuron_subnet_GIDs[layer_index][population_index] = neuron_subnet
      nest.Create(neuron_model, n_neurons[layer_index][population_index])

      # initialize membrane potentials
      ctr = 0
      for n in GetLocalNodes(neuron_subnet)[0]:
        nest.SetStatus((n,), {'V_m': normal_rdvs[ctr].normal(Vm0_mean, Vm0_std)})

      nest.ChangeSubnet(population_subnet)

      # create and configure stimulus and recording devices
      device_subnet = nest.Create('subnet')
      nest.ChangeSubnet(device_subnet)
      this_spike_detector = nest.Create('spike_detector')
      # Set spike detector label for filenames. The GID of the spike
      # detector and the process number are appended automatically.
      nest.SetStatus(this_spike_detector, {
        'label': spike_detector_label + '_' + str(layer_index) + '_' + str(population_index),
        'to_file': save_cortical_spikes
      })
      spike_detector_GIDs[layer_index][population_index] = this_spike_detector

      this_voltmeter = nest.Create('voltmeter')
      nest.SetStatus(this_voltmeter, {
        'label': voltmeter_label + '_' + str(layer_index) + '_' + str(population_index),
        'to_file': save_voltages
      })
      voltmeter_GIDs[layer_index][population_index] = this_voltmeter

      this_poisson_generator = nest.Create('poisson_generator')
      this_K_bg = K_bg[layer_index][population_index]
      nest.SetStatus(this_poisson_generator, {
        'rate': this_K_bg * bg_rate
      })
      poisson_GIDs[layer_index][population_index] = this_poisson_generator

      this_dc_generator = nest.Create('dc_generator')
      nest.SetStatus(this_dc_generator, {
        'amplitude': this_K_bg * dc_amplitude
      })
      dc_GIDs[layer_index][population_index] = this_dc_generator

  # create and configure thalamic neurons (parrots) and their Poisson inputs
  nest.ChangeSubnet((0,))
  if n_thal > 0:
    th_subnet = nest.Create('subnet')
    nest.ChangeSubnet(th_subnet)

    th_neuron_subnet_GID = nest.Create('subnet')
    nest.ChangeSubnet(th_neuron_subnet_GID)
    nest.Create('parrot_neuron')

    nest.ChangeSubnet(th_subnet)
    th_device_subnet = nest.Create('subnet')
    nest.ChangeSubnet(th_device_subnet)
    th_poisson_GID = nest.Create('poisson_generator')
    nest.SetStatus(th_poisson_GID, {
      'rate': th_rate,
      'start': th_start,
      'stop': th_start + th_duration
    })

    if record_thalamic_spikes:
      th_spike_detector_GID = nest.Create('spike_detector')
      nest.SetStatus(th_spike_detector_GID, {
        'label': th_spike_detector_label,
        'to_file': save_thalamic_spikes
      })


def WriteGIDstoFile():
  if run_mode == 'test':
    f = GID_filename

  if run_mode == 'production':
    f = output_path + '/' + GID_filename

  with open(f, 'w') as f:
    for n in neuron_subnet_GIDs.flatten():
      GIDs = nest.GetNodes((n,))
      f.write(str(min(GIDs[0])) + '\t' + str(max(GIDs[0])) + '\n')

    f.close()


def ConnectNetworkNodes():
  global neuron_subnet_GIDs
  global spike_detector_GIDs
  global voltmeter_GIDs
  global poisson_GIDs
  global dc_GIDs
  global th_neuron_subnet_GID
  global th_poisson_GID
  global th_spike_detector_GID

  # target layer
  for target_layer in xrange(n_layers):
    for target_pop in xrange(n_pops_per_layer):
      # get neuron IDs
      target_nodes = GetGlobalNodes(neuron_subnet_GIDs[target_layer][target_pop])

      n_targets = n_neurons[target_layer][target_pop]
      full_scale_n_targets = full_scale_n_neurons[target_layer][target_pop]

      # source layer
      for source_layer in xrange(n_layers):
        # source population
        for source_pop in xrange(n_pops_per_layer):
          ### local connections

          # get neuron IDs
          source_nodes = GetGlobalNodes(neuron_subnet_GIDs[source_layer][source_pop])
          n_sources = n_neurons[source_layer][source_pop]
          full_scale_n_sources = full_scale_n_neurons[source_layer][source_pop]

          # get connection probability
          # pick row (target) in conn_probs
          r = (target_layer * n_pops_per_layer) + target_pop
          # pick column (source) in conn_probs
          c = (source_layer * n_pops_per_layer) + source_pop
          this_conn =  conn_probs[r][c]# probability for this connection

          # Compute numbers of synapses assuming binomial degree
          # distributions and allowing for multapses (see Potjans
          # and Diesmann 2012 Cereb Cortex Eq. 1)
          if preserve_K:
            prod = full_scale_n_sources * full_scale_n_targets
            n_syn_temp = np.log(1.-this_conn)/np.log((prod-1.)/prod)
            this_n_synapses = int((n_syn_temp * n_targets) / full_scale_n_targets)
          else:
            prod = n_sources * n_targets
            this_n_synapses = int(np.log(1.-this_conn)/np.log((prod-1.)/prod))

          if this_n_synapses > 0:
            mean_weight = PSC_array[target_layer][target_pop][source_layer][source_pop]
            # Create label for target and source populations
            conn_label = 'layers' + str(target_layer) + '_' + 'populations' + str(target_pop) + '-' + \
                        'layers' + str(source_layer) + '_' + 'populations' + str(source_pop)

            # fill the weight dictionary for Connect and insert it into the synapse dictionary
            if mean_weight > 0:
              weight_dict_exc['mu'] = mean_weight
              weight_dict_exc['sigma'] = np.abs(PSC_sd[source_pop])
              syn_dict['weight'] = weight_dict_exc
            else:
              weight_dict_inh['mu'] = mean_weight
              weight_dict_inh['sigma'] = np.abs(PSC_sd[source_pop])
              syn_dict['weight'] = weight_dict_inh

            # fill the delay dictionary for Connect and insert it into the synapse dictionary
            delay_dict['mu'] = delays[source_pop]
            delay_dict['sigma'] = np.abs(delays_sd[source_pop])
            syn_dict['delay'] = delay_dict

            # fill the connectivity dictionary with the number of synapses to be used
            conn_dict['N'] = this_n_synapses
            conn_dict['rule'] = 'fixed_total_number'

            # Connect the populations
            nest.Connect(source_nodes[0], target_nodes[0], conn_dict, syn_dict)

      if n_thal > 0:
        # connections from thalamic neurons

        source_nodes = GetGlobalNodes(th_neuron_subnet_GID)
        this_conn = C_th[target_layer][target_pop]

        # Compute numbers of synapses assuming binomial degree
        # distributions and allowing for multapses (see Potjans and
        # Diesmann 2012 Cereb Cortex Eq. 1)
        if preserve_K:
          prod = n_thal * full_scale_n_targets
          n_syn_temp = np.log(1.-this_conn)/np.log((prod-1.)/prod)
          this_n_synapses = int(full_scale_n_targets / (n_syn_temp * n_targets))
        else:
          prod = n_thal * n_targets
          this_n_synapses = int(np.log(1.-this_conn)/np.log((prod-1.)/prod))

        if this_n_synapses > 0:
          # create label for current target population
          th_conn_label = layers[target_layer] + '_' + populations[target_pop]

          # fill the weight dictionary for Connect
          weight_dict_exc['mu'] = PSC_ext
          weight_dict_exc['sigma'] = np.abs(PSC_th_sd)
          # insert the weight dictionary into the synapse dictionary
          syn_dict['weight'] = weight_dict_exc

          # fill the delay dictionary for Connect
          delay_dict['mu'] = delay_th
          delay_dict['sigma'] = np.abs(delay_th_sd)
          # insert the delay dictionary into the synapse dictionary
          syn_dict['delay'] = delay_dict

          conn_dict['N'] = this_n_synapses
          conn_dict['rule'] = 'fixed_total_number'

          nest.Connect(source_nodes, target_nodes, conn_dict, syn_dict)

      # Connect devices
      # record from a continuous range of IDs
      # (appropriate for networks without topology)
      # print tuple(spike_detector_GIDs[target_layer][target_pop])
      # print target_nodes[:int(n_neurons_rec_spikes[target_layer][target_pop])][0]
      nest.Connect(target_nodes[:int(n_neurons_rec_spikes[target_layer][target_pop])][0],
        tuple(spike_detector_GIDs[target_layer][target_pop]),
        'all_to_all')

      nest.Connect(tuple(voltmeter_GIDs[target_layer][target_pop]),
        tuple(target_nodes[:int(n_neurons_rec_voltage[target_layer][target_pop])])[0],
        'all_to_all')

      nest.Connect(tuple(poisson_GIDs[target_layer][target_pop]),
        target_nodes[0],
        'all_to_all',
        {'weight': PSC_ext, 'delay': delays[0]})

      nest.Connect(tuple(dc_GIDs[target_layer][target_pop]),
        target_nodes[0],
        'all_to_all')

  if n_thal > 0:
    # Connect thalamic poisson_generator to thalamic neurons (parrots)
    nest.Connect(th_poisson_GID, GetGlobalNodes(th_neuron_subnet_GID))

  if record_thalamic_spikes and n_thal > 0:
    # Connect thalamic neurons to spike detector
    nest.Connect(GetGlobalNodes(th_neuron_subnet_GID), th_spike_detector_GID)


if __name__ == '__main__':
  print "------------------------------------------------------"
  print "Starting simulation"
  print "------------------------------------------------------"

  CheckParameters()

  PrepareSimulation()

  DerivedParameters()

  CreateNetworkNodes()

  WriteGIDstoFile()

  ConnectNetworkNodes()

  nest.Simulate(t_sim)

  # for s in np.array(spike_detector_GIDs).flatten():
  #   nest.raster_plot.from_device((s,))
  #   pylab.show()
