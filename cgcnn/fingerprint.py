from ..adsorption_sites import label_occupied_sites 
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


pbcs = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 1, 0], [-1, 1, 0], [0, -1, 0], [1, -1, 0], [-1, -1, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 1], [0, 1, 1], [1, 1, 1], [-1, 1, 1], [0, -1, 1], [1, -1, 1], [-1, -1, 1], [0, 0, -1], [1, 0, -1], [-1, 0, -1], [0, 1, -1], [1, 1, -1], [-1, 1, -1], [0, -1, -1], [1, -1, -1], [-1, -1, -1]]


def multi_label_binarizer(labeled_atoms):
    '''Encoding the labels into binaries. This can be further used as a fingerprint.
       Atoms that constitute an occupied adsorption site will be labeled as 1.
       One atom can encompass multiple 1s if it contributes to multiple sites.

       Note: Please provide only the labeled atoms object.'''
    
    output = []
    for atom in labeled_atoms:
        if atom.symbol not in 'SCHON':
            if atom.tag == 0:
                output.append(np.zeros(10).astype(int).tolist())
            else:
                line = str(atom.tag)
                strings = [line[k:k+2] for k in range(0, len(line), 2)]
                lst = np.zeros(10).astype(int).tolist()
                for idx in [int(s[1]) for s in strings]:
                    lst[idx] = int(1)
                output.append(lst)

    return output


def process_data(atoms):
    '''The input atoms object must be a structure consisting of a relaxed slab + unrelaxed adsorbates.'''

    # Set second_shell=False if you don't want to label second shell atoms
    labeled_atoms = label_occupied_sites(atoms, ['CO','N'], second_shell=True)
    np_indices = [a.index for a in labeled_atoms if a.symbol not in 'SCHON']
    np_atoms = labeled_atoms[np_indices]

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
    tags = multi_label_binarizer(np_atoms) 
    atoms_init = {'positions': np_atoms.get_positions().tolist(),
                  'cell': np_atoms.get_cell().tolist(),
                  'pbc':False,
                  'numbers': np_atoms.get_atomic_numbers().tolist(), 
                  'tags': tags,
                  'nbr_fea_idx': nbr_fea_idx_init,
                  'nbr_fea': nbr_fea_init,
                  'prop': -0.77}

    for k,v in atoms_init.items():
        print(str(k)+': '  + str(v))
