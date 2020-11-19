from cabins.adsorbate_coverage import *
from cabins.adsorption_sites import *
from ase.io import read, write, Trajectory
from ase.calculators.emt import EMT
from ase.optimize import BFGS, FIRE
from ase.geometry import find_mic
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import random
import pickle
from ase.visualize import view


# Read input trajectory of the previous generation
structures = read('NiPt3_311_1_emt.traj', index=':')
# Write output trajectory of the current generation
traj = Trajectory('NiPt3_311_2_emt.traj', mode='a')
# Designate adsorbates
monodentate_adsorbates = ['H','C','O','OH','CH','CO','H2O','CH2','COH','CH3','H2CO']
bidentate_adsorbates = ['HCO','CHOH','CH2O','CH3O','H2COH','CH3OH']
adsorbates = monodentate_adsorbates + bidentate_adsorbates
# Provide energy (eV) of the clean slab
Eslab = 10.148632842226348

# Read AdsorptionSite object for the slab from pickle 
with open('adsorption_sites_NiPt3_311.pkl', 'rb') as input:
    sas = pickle.load(input)

Emin = min([struct.info['data']['Eads'] for struct in structures])
Ecutoff = 0.25 # Set a very large cutoff for the 1st generation
starting_images = []
old_labels_list = []
old_graph_list = []

for struct in structures:
    # Remain only the configurations with enegy below a threshold
    if struct.info['data']['Eads'] < Emin + Ecutoff:
#        sac = SlabAdsorbateCoverage(struct, sas)
#        fsl = sac.full_site_list
#        labs = sac.labels
#        G = sac.get_site_graph()
#                                                                    
#        if labs in old_labels_list: 
#            if old_graph_list:
#                # Remove duplicates after DFT relaxation based 
#                # on isomorphism (very likely to happen)
#                nm = iso.categorical_node_match('label', '0')
#                if any(H for H in old_graph_list if 
#                nx.isomorphism.is_isomorphic(G, H, node_match=nm)):
#                    continue           
# 
#        old_labels_list.append(labs)
#        old_graph_list.append(G)
        starting_images.append(struct)

site_list = sas.site_list.copy()
site_nblist = sas.get_neighbor_site_list()

labels_list = []
graph_list = []

random.shuffle(starting_images) 
for image in starting_images: 
    sac = SlabAdsorbateCoverage(image, sas)
    fsl = sac.full_site_list
    nbstids = []
    selfids = []
    for j, st in enumerate(fsl):
        if st['occupied'] == 1:
                nbstids += site_nblist[j]
                selfids.append(j)
    nbsids = [v for v in nbstids if v not in selfids]

    # Only add one adsorabte to a site at least 2 shells away from
    # currently occupied sites
    newsites, newnbids = [], []
    for i, s in enumerate(fsl):
        if i not in nbstids:
            newnbs = site_nblist[i]
            nbs = [n for n in newnbs if n not in nbstids]
            if not nbs:
                continue 
            newsites.append(s)
            newnbids.append(nbs)
#    newsites = [s for i, s in enumerate(fsl) if i not in nbstids]

    for k, nst in enumerate(newsites): 
        atoms = image.copy()
        #TODO: select adsorbate with probablity weighed by performance
        adsorbate = random.choice(adsorbates) 

        if adsorbate in bidentate_adsorbates:
        # Rotate a bidentate adsorbate to the direction of a randomly 
        # choosed neighbor site
            nbst = fsl[random.choice(newnbids[k])]
            pos = nst['position'] 
            nbpos = nbst['position'] 
            rotation = find_mic(np.array([nbpos-pos]), atoms.cell)[0][0]
            add_adsorbate_to_site(atoms, adsorbate, nst, rotation=rotation)        

        else:
            add_adsorbate_to_site(atoms, adsorbate, nst)        

        nsac = SlabAdsorbateCoverage(atoms, sas)
        nfsl = nsac.full_site_list

        # Make sure there no new site too close to previous sites after 
        # adding the adsorbate. Useful when adding large molecules
        if any(s for i, s in enumerate(nfsl) if (s['occupied'] == 1)
        and (i in nbsids)):
            print('Site too close')
            continue
        labs = nsac.labels
        G = nsac.get_site_graph()
        if labs in labels_list: 
            if graph_list:
                # Skip duplicates based on isomorphism 
                nm = iso.categorical_node_match('label', '0')
                potential_graphs = [g for i, g in enumerate(graph_list) 
                                    if labels_list[i] == labs]
                if any(H for H in potential_graphs if 
                nx.isomorphism.is_isomorphic(G, H, node_match=nm)):
                    continue            

#        # MC step
#        p_normal = np.minimum(1, (np.exp(-(new_e - e) / (kB * T)))) 
#        if np.random.rand() < p_normal:
        labels_list.append(labs)
        graph_list.append(G)

        atoms.calc = EMT()                 
#        opt = FIRE(atoms, logfile=None)
#        opt.run(fmax=0.1)
        Etot = atoms.get_potential_energy()
        Eads = Etot - Eslab 
        atoms.info['data'] = {'Eads': Eads}
        traj.write(atoms)
