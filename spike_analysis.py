
# Merges spike files, produces raster plots, calculates and plots firing rates

import numpy as np
import glob
import matplotlib.pyplot as plt
import os

from sim_params import *

datapath = '.'

T = t_sim
T_start = 200.   # starting point of analysis (to avoid transients)

# load GIDs

gidfile = open(os.path.join(datapath , 'population_GIDs.dat'), 'r')
gids = []
for l in gidfile:
  a = l.split()
  gids.append([int(a[0]),int(a[1])])
print 'Global IDs:'
print gids
print

# number of populations

num_pops = len(gids)
print 'Number of populations:'
print num_pops
print

# first GID in each population

raw_first_gids = [gids[i][0] for i in np.arange(len(gids))]

# population sizes

pop_sizes = [gids[i][1]-gids[i][0]+1 for i in np.arange(len(gids))]

# numbers of neurons for which spikes were recorded

if record_fraction_neurons_spikes == True:
  rec_sizes = [int(pop_sizes[i]*frac_rec_spikes) for i in xrange(len(pop_sizes))]
else:
  rec_sizes = [n_rec_spikes]*len(pop_sizes)

# first GID of each population once device GIDs are dropped

first_gids=[int(1 + np.sum(pop_sizes[:i])) for i in np.arange(len(pop_sizes))]

# last GID of each population once device GIDs are dropped

last_gids = [int(np.sum(pop_sizes[:i+1])) for i in np.arange(len(pop_sizes))]

# convert lists to a nicer format, i.e. [[2/3e, 2/3i], []....]

Pop_sizes =[pop_sizes[i:i+2] for i in xrange(0,len(pop_sizes),2)]
print 'Population sizes:'
print Pop_sizes
print

Raw_first_gids =[raw_first_gids[i:i+2] for i in xrange(0,len(raw_first_gids),2)]

First_gids = [first_gids[i:i+2] for i in xrange(0,len(first_gids),2)]

Last_gids = [last_gids[i:i+2] for i in xrange(0,len(last_gids),2)]

# total number of neurons in the simulation

num_neurons = last_gids[len(last_gids)-1]
print 'Total number of neurons:'
print num_neurons
print

# load spikes from gdf files, correct GIDs and merge them in population files,
# and store spike trains

# will contain neuron id resolved spike trains
neuron_spikes = [[] for i in np.arange(num_neurons+1)]
# container for population-resolved spike data
spike_data= [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]

counter = 0

for layer in ['0','1','2','3']:
  for population in ['0','1']:
    output = os.path.join(datapath, 'population_spikes-{}-{}.gdf'.format(layer, population))
    file_pattern = os.path.join(datapath, 'spikes_{}_{}*'.format(layer, population))
    files = glob.glob(file_pattern)
    print 'Merge '+str(len(files))+' spike files from L'+layer+'P'+population
    if files:
      merged_file = open(output,'w')
      for f in files:
        data = open(f,'r')
        for l in data :
          a = l.split()
          a[0] = int(a[0])
          a[1] = float(a[1])
          raw_first_gid = Raw_first_gids[int(layer)][int(population)]
          first_gid = First_gids[int(layer)][int(population)]
          a[0] = a[0] - raw_first_gid + first_gid

          if(a[1] > T_start):    # discard data in the start-up phase
            spike_data[counter][0].append(num_neurons-a[0])
            spike_data[counter][1].append(a[1]-T_start)
            neuron_spikes[a[0]].append(a[1]-T_start)

            converted_line = str(a[0]) + '\t' + str(a[1]) +'\n'
            merged_file.write(converted_line)

    data.close()
    merged_file.close()
    counter +=1


import matplotlib.cm as cm
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
for i in range(len(tableau20)):
  r, g, b = tableau20[i]
  tableau20[i] = (r / 255., g / 255., b / 255.)

clrs = tableau20
plt.ion()

# raster plot

plt.figure(1)
counter = 1
for j in np.arange(num_pops):
    for i in np.arange(first_gids[j],first_gids[j]+rec_sizes[j]):
      if len(neuron_spikes[i]):
        print neuron_spikes[i], np.ones_like(neuron_spikes[i])+sum(rec_sizes)-counter
        plt.plot(neuron_spikes[i],np.ones_like(neuron_spikes[i])+sum(rec_sizes)-counter,'k o',ms=1, mfc=clrs[j],mec=clrs[j])
      counter+=1
plt.xlim(0,T-T_start)
plt.ylim(0,sum(rec_sizes))
plt.xlabel(r'time (ms)')
plt.ylabel(r'neuron id')
plt.savefig(os.path.join(datapath, 'rasterplot.png'))


# firing rates

rates = []
temp = 0

for i in np.arange(num_pops):
  for j in np.arange(first_gids[i], last_gids[i]):
    temp += len(neuron_spikes[j])
  rates.append(temp/(rec_sizes[i]*(T-T_start))*1e3)
  temp = 0

# print
# print 'Firing rates:'
# print rates

plt.figure(2)
ticks= np.arange(num_pops)
plt.bar(ticks, rates, width=0.9, color=clrs)
xticklabels = ['L2/3e','L2/3i','L4e','L4i','L5e','L5i','L6e','L6i']
plt.setp(plt.gca(), xticks=ticks+0.5, xticklabels=xticklabels)
plt.xlabel(r'subpopulation')
plt.ylabel(r'firing rate (spikes/s)')
plt.savefig(os.path.join(datapath, 'firing_rates.png'))

plt.show()
