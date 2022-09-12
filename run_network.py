
from bmtk.simulator import bionet
from bmtk.simulator.core import simulation_config 
from bmtk.utils.sim_setup import BioNetEnvBuilder


conf = bionet.Config.from_json('simulation_config.json')
#conf.update({"network":"$BASE_DIR\\circuit_config.json"})
#basically just need to make your own json that bioNetwork can read. It just needs a dictionary.
conf.copy_to_output()
conf.build_env()
net = bionet.BioNetwork.from_config(conf)



sim = bionet.BioSimulator.from_config(conf, network=net)
sim.run()

