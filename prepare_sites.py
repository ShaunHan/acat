from act.adsorption_sites import *
from ase.io import read, write
import pickle

slab = read('Ni3Pt_111_slab.traj')

sas = SlabAdsorptionSites(slab, surface='fcc111', 
                          sites_on_subsurface=True,
                          show_composition=True)

for s in sas.site_list:
    print(s)

with open('Ni3Pt_111_sites.pkl', 'wb') as output:
    pickle.dump(sas, output, pickle.HIGHEST_PROTOCOL)       
