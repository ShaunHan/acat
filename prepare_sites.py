from allocat.adsorption_sites import *
from ase.io import read, write
import pickle

slab = read('NiPt3_311_surface_small.traj')
slab.calc = None

sas = SlabAdsorptionSites(slab, surface='fcc311', 
                          sites_on_subsurface=True,
                          show_composition=True)

with open('adsorption_sites_NiPt3_311.pkl', 'wb') as output:
    pickle.dump(sas, output, pickle.HIGHEST_PROTOCOL)       
