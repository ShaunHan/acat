from allocat.adsorbate_coverage import *
from allocat.adsorption_sites import *
from ase.io import read, write, Trajectory
from ase.calculators.emt import EMT
from ase.optimize import BFGS, FIRE
from ase.geometry import find_mic
from ase.calculators.lammpslib import LAMMPSlib
from ase.formula import Formula
from ase.data import atomic_numbers, atomic_masses_legacy
import gaussian_process as gp
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import random
import pickle


# Read input trajectory of the previous generation
structures = read('NiPt3_311_6_reax.traj', index=':')
# Write output trajectory of the current generation
traj = Trajectory('NiPt3_311_7_reax.traj', mode='w')
# Designate adsorbates
monodentate_adsorbates = ['H','C','O','OH','CH','CO','OH2','CH2','COH','CH3']#,'OCH2','OCH3']
bidentate_adsorbates = ['CHO']#,'CHOH','CH2O','CH3O','CH2OH','CH3OH']
adsorbates = monodentate_adsorbates + bidentate_adsorbates
# Weights of adding each adsorbate according to collision theory
adsorbate_weights = [1/np.sqrt(np.sum([atomic_masses_legacy[atomic_numbers[s]] 
                     for s in list(Formula(ads))])) for ads in adsorbates]
# Scaled by partial pressure of each species
adsorbate_weights = [w*p for w,p in zip(adsorbate_weights,adsorbate_pressures)]
# Provide energy (eV) of the clean slab
Eslab = -319.5669510338692
# Provide number of new structures generated in this generation
Ngen = 200

# Reaxff model
model_dir = '/home/energy/shuha/project/4-NiPt_with_adsorbate'

header=['units real',
        'atom_style charge',
        'atom_modify map array sort 0 0']

cmds = ['pair_style reax/c NULL checkqeq yes safezone 1.6 mincap 100',
        'pair_coeff * * {}/ffield_NiPt.reax H C O Ni Pt'.format(model_dir),
        'fix 1 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c']

calc = LAMMPSlib(lmpcmds = cmds, lammps_header=header, 
                 atom_types={'H':1,'C':2,'O':3,'Ni':4,'Pt':5}, 
                 keep_alive=True, log_file='lammpslib.log')

# Read AdsorptionSite object for the slab from pickle 
with open('adsorption_sites_NiPt3_311.pkl', 'rb') as input:
    sas = pickle.load(input)

#Emin = min([struct.info['data']['Eads_dft'] for struct in structures])
Ecut = 1.6 # Set a very large cutoff for the 1st generation
starting_images = []
old_labels_list = []
old_graph_list = []

for struct in structures:
    # Remain only the configurations with enegy below a threshold
#    if struct.info['data']['Eads_dft'] < Ecut:  #+Emin?
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
bidentate_nblist = sas.get_neighbor_site_list(neighbor_number=1)
site_nblist = sas.get_neighbor_site_list(neighbor_number=2)

labels_list = []
graph_list = []

Nnew = 0
while Nnew < Ngen:
    image = random.choice(starting_images) 
    sac = SlabAdsorbateCoverage(image, sas)
    fsl = sac.full_site_list
    nbstids = []
    selfids = []
    for j, st in enumerate(fsl):
        if st['occupied'] == 1:
                nbstids += site_nblist[j]
                selfids.append(j)
    nbsids = [v for v in nbstids if v not in selfids]

    # Select adsorbate with probablity weighted by 1/sqrt(mass)
    adsorbate = random.choices(k=1, population=adsorbates,
                               weights=adsorbate_weights)[0] 

    # Only add one adsorabte to a site at least 2 shells 
    # away from currently occupied sites
    nsids = [i for i, s in enumerate(fsl) if i not in nbstids]
    # Prohibit adsorbates with more than 1 atom from entering subsurf sites
    subsurf_site = True
    nsi = None
    while subsurf_site: 
        nsi = random.choice(nsids)
        subsurf_site = (len(adsorbate) > 1 and fsl[nsi]['site'] == 'subsurf')
    nst = fsl[nsi]

    binbs = bidentate_nblist[nsi]    
    binbids = [n for n in binbs if n not in nbstids]
    if not binbids:
        continue
    atoms = image.copy()
    if adsorbate in bidentate_adsorbates:
        # Rotate a bidentate adsorbate to the direction of a randomly 
        # choosed neighbor site
        nbst = fsl[random.choice(binbids)]
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
                print('Duplicate found')
                continue            
    labels_list.append(labs)
    graph_list.append(G)
    
    atoms.calc = calc                 
#    opt = FIRE(atoms, logfile=None)
#    opt.run(fmax=0.1)
#    sas.update_positions(atoms)
    Etot = atoms.get_potential_energy()
    Eads = Etot - Eslab 
    atoms.info['data'] = {'Eads_dft': Eads}
    traj.write(atoms)
    Nnew += 1
