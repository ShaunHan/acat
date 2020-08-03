from ..adsorption_sites import multi_label_counter
from ase.io import read, write
import pickle
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist
from ase.data import chemical_symbols
import pickle
import json
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
import os 


def process_data(atoms, adsorbate, second_shell=False):
    '''The input atoms object must be a structure consisting of a relaxed slab + unrelaxed adsorbates.
       Set second_shell=False if you don't want to label second shell atoms.'''

    np_indices = [a.index for a in atoms if a.symbol not in 'SCHON']
    np_atoms = atoms[np_indices]

    # Use pymatgen to get all neighbors
    structure = AseAtomsAdaptor.get_structure(np_atoms)
    all_nbrs_init = structure.get_all_neighbors(r=12., include_index=True)
    num_of_nbrs = [len(nbrs) for nbrs in all_nbrs_init]

    # CGCNN requies a minimum of 9 neighbors per atom
    nnbr=9
    if np.min(num_of_nbrs) < nnbr: 
        raise ValueError('Each atom must have at least 9 neighbor atoms')
    
    # Prepare features
    all_nbrs_init = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs_init]
    nbr_fea_idx_init = np.array([list(map(lambda x: x[2],nbr[:nnbr])) for nbr in all_nbrs_init]).tolist()
    nbr_fea_init = np.array([list(map(lambda x: x[1], nbr[:nnbr])) for nbr in all_nbrs_init]).tolist()
    tags = multi_label_counter(atoms, adsorbate, second_shell) 
    atoms_init = {'positions': np_atoms.get_positions().tolist(),
                  'cell': atoms.get_cell().tolist(),
                  'pbc': atoms.get_pbc(),
                  'numbers': np_atoms.get_atomic_numbers().tolist(), 
                  'tags': tags,
                  'nbr_fea_idx': nbr_fea_idx_init,
                  'nbr_fea': nbr_fea_init,
                  'prop': -0.77}

    for k,v in atoms_init.items():
        print(str(k)+': '  + str(v))
