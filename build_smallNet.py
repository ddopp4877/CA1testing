from bmtk.builder import NetworkBuilder
import numpy as np
import os
from neuron import h


seed = 999
rng = np.random.default_rng(seed)

import shutil
fileList = os.listdir()
if 'network' in fileList:
    shutil.rmtree('network')

net = NetworkBuilder("myNet")

net.add_nodes(N = 1, pop_name="AAC",
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:axoaxoniccell',
              morphology=None)

net.add_nodes(N = 1, pop_name="Pyr",
              mem_potential='e',
              model_type='biophysical',
              model_template='hoc:pyramidalcell',
              morphology=None)

#from bmtk.simulator.bionet.pyfunction_cache import add_synapse_model
#Chn2Pyr = h.chn2pyr(sec_x, sec=sec_id)
#add_synapse_model(Chn2Pyr, 'chn2pyr', overwrite=False)

net.add_edges(source={'pop_name': 'AAC'}, target={'pop_name': 'Pyr'},
              connection_rule=12,
              syn_weight=5.0e-03,
              dynamics_params='AMPA_ExcToExc.json',
              model_template='Exp2Syn',
              delay=2.0,
              target_sections=['soma'],
              distance_range = [-10000.0, 10000.0])
"""
net.add_edges(source={'pop_name':'AAC'}, target={'pop_name':'Pyr'},
              connection_rule=1,
              syn_weight=1.2e-4,
              delay = 0.1,
              dynamics_params='CHN2PN.json',
              model_template = 'chn3pyr',
              target_sections=['soma'],
              distance_range = [-10000.0, 10000.0],
              sec_id = 6,
              sec_x = 0.5
              )
"""

#(net,'AAC','Pyr',[ 0.072,     400],'CHN2PN.json', [-10000.0, 10000.0],'axonal', 6, 0.5)
net.build()
net.save_nodes(output_dir='network')
net.save_edges(output_dir='network')

for item in net.nodes():
    print(item)

