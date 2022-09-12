from bmtk.utils.sim_setup import build_env_bionet

import os


build_env_bionet(
    overwrite_config=True,
    base_dir='./',       # Where to save the scripts and config files
    config_file='config.json', # Where main config will be saved.
    network_dir='./network',     # Location of directory containing network files
    tstop=2000.0, dt=0.1,      # Run a simulation for 2000 ms at 0.1 ms intervals
    report_vars=['v', 'ecp'],  # Tells simulator we want to record membrane potential and calcium traces
    current_clamp={
        'gids' : 0,
        'amp': 0.120,
        'delay': 200.0,
        'duration': 10.0
    },
    compile_mechanisms=False   # Will try to compile NEURON mechanisms
)
