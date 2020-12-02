from allocat.adsorbate_coverage import *
from allocat.adsorption_sites import *
from ase.io import read, write, Trajectory
from ase.calculators.emt import EMT
from ase.optimize import BFGS, FIRE
from ase.geometry import find_mic
from ase.calculators.lammpslib import LAMMPSlib
import numpy as np
import random
import pickle


slab = read('NiPt3_311_surface_small.traj')

monodentate_adsorbates = ['H','C','O','OH','CH','CO','OH2','CH2','COH','CH3']#,'OCH2','OCH3']
bidentate_adsorbates = ['CHO']#,'CHOH','CH2O','CH3O','CH2OH','CH3OH']
adsorbates = monodentate_adsorbates + bidentate_adsorbates

traj = Trajectory('NiPt3_311_1_reax.traj', mode='w')

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
slab.calc = calc
#opt = FIRE(slab, logfile=None)
#opt.run(fmax=0.1)
Eslab = slab.get_potential_energy()
print(Eslab)

with open('adsorption_sites_NiPt3_311.pkl', 'rb') as input:
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
        rotation = find_mic(np.array([nbpos-pos]), atoms.cell)[0][0]
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
