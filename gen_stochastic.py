from asac.adsorption_sites import *
from asac.adsorbate_coverage import *
from asac.utilities import get_mic
from ase.io import read, write, Trajectory
from ase.calculators.emt import EMT
from ase.optimize import BFGS, FIRE
from ase.calculators.lammpslib import LAMMPSlib
from ase.formula import Formula
from ase.data import atomic_numbers, atomic_masses_legacy
import gaussian_process as gp
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import argparse
import random
import pickle


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Stochastic global optimization", 
                    fromfile_prefix_chars="+")
    parser.add_argument(
        "-i",
        type=str,
        help="Path to the input trajectory file",
    )
    parser.add_argument(
        "-o",
        type=str,
        help="Name of the output trajectory file",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=100,
        help="Number of generated structures",
    )
    parser.add_argument(
        "-cutoff",
        type=float,
        default=2.0,
        help="Energy cutoff of adsorption energy difference",
    )
    parser.add_argument(
        "-surface",
        type=str,
        default=None,
        help="Surface type (crytal structure + miller index)",
    )
    parser.add_argument(
        "-adsorbates",
        nargs="+", 
        required=True,
        help="Adsorbate candidates",
    )
    parser.add_argument(
        "-fix_surfcomp",
        action="store_true",
        help="Fix the elemental composition of surface atoms",
    )
    parser.add_argument(
        "-adsorption_sites",
        type=str,
        default=None,
        help="Path to the AdsorptionSites pickle file. Automatically \
        identify adsorption sites if not specified",
    )
    parser.add_argument(
        "-assign_weights",
        action="store_true",
        help="Assign weights (probabilities) of adding each species",
    )
    parser.add_argument(
        "-pressures",
        nargs="+", 
        required=False,
        help="Partial pressures (ratios) of each species. Probabilities \
        of adding each species are scaled accordingly. Same partial \
        pressure ratios are used if not specified",
    )

    return parser.parse_args(arg_list)


def main():
    args = get_arguments()

    # Read input trajectory of the previous generation
    structures = read(args.i, index=':')
    # Write output trajectory of the current generation
    traj = Trajectory(args.o, mode='w')
    # Designate adsorbates
#    mono_adsorbates = ['H','C','O','OH','CH','CO','OH2','CH2','COH','CH3']#,'OCH2','OCH3']
#    multi_adsorbates = ['CHO']#,'CHOH','CH2O','CH3O','CH2OH','CH3OH']
#    adsorbates = mono_adsorbates + multi_adsorbates
    adsorbates = args.adsorbates
    mono_adsorbates, multi_adsorbates = [], []
    for ads in adsorbates:
        if (len(list(Formula(ads))) < 4 or ads in ['CH3','NH3','OCH2','OCH3']) and \
        (ads not in ['CHO','O3','C3']):
            mono_adsorbates.append(ads)
        else:
            multi_adsorbates.append(ads)

    if args.assign_weights:
        # Weights of adding each adsorbate according to collision theory
        adsorbate_weights = [1/np.sqrt(np.sum([atomic_masses_legacy[atomic_numbers[s]] 
                             for s in list(Formula(ads))])) for ads in adsorbates]
        if args.pressures:
            # Scaled by partial pressure of each species
            adsorbate_pressures = args.pressures
            adsorbate_weights = [w*float(p) for w,p in zip(adsorbate_weights,adsorbate_pressures)]
            # Provide energy (eV) of the clean slab
    Eslab = -319.5669510338692
    # Provide number of new structures generated in this generation
    Ngen = args.n

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

    if args.fix_surfcomp:
        if args.adsorption_sites:
            # Read AdsorptionSite object for the slab from pickle 
            with open(args.adsorption_sites, 'rb') as input:
                sas = pickle.load(input)
        else:
            sas = SlabAdsorptionSites(structures[0], surface=args.surface,
                                      sites_on_subsurf=True,
                                      composition_effect=True)

    Ecut = args.cutoff # Set a very large cutoff for the 1st generation
    starting_images = []
    old_labels_list, old_graph_list = [], []
    for struct in structures:
        # Remain only the configurations with enegy below a threshold
    #    if struct.info['data']['Eads_dft'] < Ecut:  #+Emin?
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

    labels_list, graph_list = [], []
    Nnew = 0
    while Nnew < Ngen:
        image = random.choice(starting_images)
        if not args.fix_surfcomp: 
            sas = SlabAdsorptionSites(image, args.surface,
                                      allow_6fold=True,
                                      composition_effect=True)
        sac = SlabAdsorbateCoverage(image, sas)
        hsl = sac.hetero_site_list
        nbstids = []
        selfids = []
        for j, st in enumerate(hsl):
            if st['occupied'] == 1:
                nbstids += site_nblist[j]
                selfids.append(j)
        nbsids = [v for v in nbstids if v not in selfids]
 
        # Select adsorbate with probablity weighted by 1/sqrt(mass)
        if not args.assign_weights:
            adsorbate = random.choice(adsorbates)
        else:
            adsorbate = random.choices(k=1, population=adsorbates,
                                       weights=adsorbate_weights)[0] 
 
        # Only add one adsorabte to a site at least 2 shells 
        # away from currently occupied sites
        nsids = [i for i, s in enumerate(hsl) if i not in nbstids]
        # Prohibit adsorbates with more than 1 atom from entering subsurf sites
        subsurf_site = True
        nsi = None
        while subsurf_site: 
            nsi = random.choice(nsids)
            subsurf_site = (len(adsorbate) > 1 and hsl[nsi]['geometry'] == 'subsurf')
        nst = hsl[nsi]
 
        binbs = bidentate_nblist[nsi]    
        binbids = [n for n in binbs if n not in nbstids]
        if not binbids:
            continue
        atoms = image.copy()
        if adsorbate in multi_adsorbates:
            # Rotate a bidentate adsorbate to the direction of a randomly 
            # choosed neighbor site
            nbst = hsl[random.choice(binbids)]
            pos = nst['position'] 
            nbpos = nbst['position'] 
            orientation = get_mic(nbpos, pos, atoms.cell)
            add_adsorbate_to_site(atoms, adsorbate, nst, orientation=orientation)        
 
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
        
        atoms.calc = calc                 
#        opt = FIRE(atoms, logfile=None)
#        opt.run(fmax=0.1)
#        sas.update_positions(atoms)
        Etot = atoms.get_potential_energy()
        Eads = Etot - Eslab 
        atoms.info['data'] = {'Eads_dft': Eads}
        traj.write(atoms)
        Nnew += 1


if __name__ == "__main__":
    main()
