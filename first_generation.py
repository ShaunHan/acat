from asac.adsorbate_coverage import *
from asac.adsorption_sites import *
from asac.utilities import get_mic
from ase.io import read, write, Trajectory
from ase.optimize import BFGS, FIRE
import numpy as np
import random
import pickle


root = '/home/energy/shuha/project/6-rgo/1-Ni3Pt_surface/111/' 
slab = read(root + 'Ni3Pt_111_slab.traj')
traj = Trajectory('Ni3Pt_111_1.traj', mode='w')

monodentate_adsorbates = ['H','C','O','OH','CH','CO','OH2','CH2','COH','CH3']#,'OCH2','OCH3']
bidentate_adsorbates = ['CHO']#,'CHOH','CH2O','CH3O','CH2OH','CH3OH']
adsorbates = monodentate_adsorbates + bidentate_adsorbates

Eslab = slab.get_potential_energy() 
with open(root + 'Ni3Pt_111_sites.pkl', 'rb') as input:
    sas = pickle.load(input)

site_list = sas.site_list
site_nblist = sas.get_neighbor_site_list(neighbor_number=1)
unis = sas.get_unique_sites(unique_composition=True)
print(unis)

for uni in unis:
    for adsorbate in monodentate_adsorbates:
        if len(adsorbate) > 1 and uni[0] == 'subsurf':
            continue
        atoms = slab.copy()    
        add_adsorbate(atoms, adsorbate,     
                      site=uni[0], 
                      geometry=uni[1], 
                      composition= uni[2], 
                      site_list=site_list)
        atoms.calc = calc
#        opt = FIRE(atoms, logfile=None)
#        opt.run(fmax=0.1)
#        sas.update_positions(atoms)
        Etot = atoms.get_potential_energy()
        Eads = Etot - Eslab
        atoms.info['data'] = {'Eads_dft': Eads}
        atoms.info['data'] = {'Eads_krr': None}
        sac = SlabAdsorbateCoverage(atoms, sas)
        labels = sac.labels
        atoms.info['data']['labels'] = labels
        traj.write(atoms)


for uni in unis:
    for adsorbate in bidentate_adsorbates:
        if uni[0] == 'subsurf':
            continue
        atoms = slab.copy()
        si, site = next(((i, s) for i, s in enumerate(site_list) if 
                    s['site'] == uni[0] and s['geometry'] == uni[1] 
                    and s['composition'] == uni[2]), None)
        nbstids = site_nblist[si]
        nbsite = site_list[random.choice(nbstids)]
        pos = site['position'] 
        nbpos = nbsite['position'] 
        rotation = get_mic(nbpos, pos, atoms.cell)
        add_adsorbate_to_site(atoms, adsorbate, site, rotation=rotation) 
        atoms.calc = calc
#        opt = FIRE(atoms, logfile=None)
#        opt.run(fmax=0.1)
#        sas.update_positions(atoms)
        Etot = atoms.get_potential_energy()
        Eads = Etot - Eslab
        atoms.info['data'] = {'Eads_dft': Eads}
        atoms.info['data'] = {'Eads_krr': None}
        sac = SlabAdsorbateCoverage(atoms, sas)
        labels = sac.labels
        atoms.info['data']['labels'] = labels
        traj.write(atoms)
