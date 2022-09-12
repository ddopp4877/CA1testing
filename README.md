build_smallNet.py
build_env.py
add "network": "$BASE_DIR\\circuit_config.json" to bottom of simulation_config.json -- need a fix for this
run_network.py
# get some error about unknown property morphology - can set to None when building network, but should probably find a way to make an swc first.
# a biophysical cell seems to expect morphology

in pyfunction_cache.py
you pass _PyFunctions an object, and it automatically adds a dictionary called __syn_weights = {} and a method called
add_synaptic_weight()
which takes a function as an argument and adds it (along with its unique key) to the dictionary of __syn_weights

passy _PyFunctions an object, and it automatically adds a dictionary called __synapse_models,
and a method called add_synapse_model()
 which creates a key with the given name and assigns to it a given function
 in synapses.py, it takes as arguments the function Bg2Pyr, 
 and the string 'bg2pyr' which is the function and the name, respectively.