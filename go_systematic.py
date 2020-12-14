from act.adsorbate_coverage import *
from act.adsorption_sites import *
from ase.io import read, write, Trajectory
from ase.calculators.emt import EMT
from ase.optimize import BFGS, FIRE
from ase.geometry import find_mic
from ase.calculators.lammpslib import LAMMPSlib
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import random
import pickle


# Read input trajectory of the previous generation
structures = read('NiPt3_311_1_reax.traj', index=':')
# Write output trajectory of the current generation
traj = Trajectory('NiPt3_311_2_reax.traj', mode='a')
# Designate adsorbates
monodentate_adsorbates = ['H','C','O','OH','CH','CO','OH2','CH2','COH','CH3','OCH2','OCH3']
bidentate_adsorbates = ['CHO','CHOH','CH2O','CH3O','CH2OH','CH3OH']
adsorbates = monodentate_adsorbates + bidentate_adsorbates
# Provide energy (eV) of the clean slab
Eslab = -325.32297835924203

# Read AdsorptionSite object for the slab from pickle 
with open('adsorption_sites_NiPt3_311.pkl', 'rb') as input:
    sas = pickle.load(input)

#Emin = min([struct.info['data']['Eads'] for struct in structures])
Ecut = 1.6 # Set a very large cutoff for the 1st generation
starting_images = []
old_labels_list = []
old_graph_list = []

for struct in structures:
    # Remain only the configurations with enegy below a threshold
#    if struct.info['data']['Eads'] < Ecut:
#        sac = SlabAdsorbateCoverage(struct, sas)
#        hsl = sac.hetero_site_list
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
bidentate_nblist = sas.get_neighbor_site_list(neighbor_number=1)
site_nblist = sas.get_neighbor_site_list(neighbor_number=2)

labels_list = []
graph_list = []

random.shuffle(starting_images) 
for image in starting_images: 
    sac = SlabAdsorbateCoverage(image, sas)
    hsl = sac.hetero_site_list
    nbstids = []
    selfids = []
    for j, st in enumerate(hsl):
        if st['occupied'] == 1:
                nbstids += site_nblist[j]
                selfids.append(j)
    nbsids = [v for v in nbstids if v not in selfids]

    # Only add one adsorabte to a site at least 2 shells away from
    # currently occupied sites
    newsites, binbids = [], []
    for i, s in enumerate(hsl):
        if i not in nbstids:
            binbs = bidentate_nblist[i]
            binbis = [n for n in binbs if n not in nbstids]
            if not binbis:
                continue 
            newsites.append(s)
            binbids.append(binbis)

    for k, nst in enumerate(newsites): 
        for adsorbate in adsorbates:
            if adsorbate in bidentate_adsorbates:
                nis = binbids[k]
            else:
                nis = [0]
            for ni in nis:
                # Prohibit adsorbates with more than 1 atom from entering subsurf sites
                if len(adsorbate) > 1 and nst['site'] == 'subsurf':
                    continue

                atoms = image.copy()
                if adsorbate in bidentate_adsorbates:
                    # Rotate a bidentate adsorbate to all possible directions of
                    # a neighbor site
                    nbst = hsl[ni]
                    pos = nst['position'] 
                    nbpos = nbst['position'] 
                    rotation = find_mic(np.array([nbpos-pos]), atoms.cell)[0][0]
                    add_adsorbate_to_site(atoms, adsorbate, nst, rotation=rotation)        
 
                else:
                    add_adsorbate_to_site(atoms, adsorbate, nst)        
 
                nsac = SlabAdsorbateCoverage(atoms, sas)
                nhsl = nsac.hetero_site_list
 
                # Make sure there no new site too close to previous sites after 
                # adding the adsorbate. Useful when adding large molecules
                if any(s for i, s in enumerate(nhsl) if (s['occupied'] == 1)
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
                            print('Duplicate found')
                            continue             
                labels_list.append(labs)
                graph_list.append(G)
 
                atoms.calc = EMT()                 
#                opt = FIRE(atoms, logfile=None)
#                opt.run(fmax=0.1)
#                sas.update_positions(atoms)
                Etot = atoms.get_potential_energy()
                Eads = Etot - Eslab 
                atoms.info['data'] = {'Eads': Eads}
                traj.write(atoms)
