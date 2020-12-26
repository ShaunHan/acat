from .adsorption_sites import * 
from .utilities import * 
from ase.io import read, write
from ase.build import molecule
from ase.data import covalent_radii, atomic_numbers
from ase.geometry import find_mic, get_duplicate_atoms
from ase.formula import Formula
from ase.visualize import view
from ase import Atom, Atoms
from asap3 import FullNeighborList
from collections import defaultdict 
from collections import Iterable, Counter
from operator import itemgetter
import networkx as nx
import numpy as np
import random
import copy
import re


# Set global variables
adsorbate_elements = 'SCHON'
heights_dict = {'ontop': 2., 
                'bridge': 2., 
                'short-bridge': 2.,
                'long-bridge': 2.,
                'fcc': 2., 
                'hcp': 2.,
                '3fold': 2., 
                '4fold': 2.,
                '5fold': 2.,
                '6fold': 0.}

# Make your own adsorbate list. Make sure you always sort the 
# indices of the atoms in the same order as the symbol. 
# First element always starts from bonded index or the 
# bonded element with smaller atomic number if multi-dentate.
                  # Monodentate (vertical)
adsorbate_list = ['H','C','O','CH','OH','CO','CH2','OH2','COH','CH3','OCH','OCH2','OCH3', 
                  # Multidentate (lateral)
                  'CHO','CHOH','CH2O','CH3O','CH2OH','CH3OH','CHOOH','COOH','CHOO','CO2'] 
adsorbate_formula_dict = {k: ''.join(list(Formula(k))) for k in adsorbate_list}

# Make your own bidentate fragment dict
adsorbate_fragment_dict = {'CO': ['C','O'],     # Possible
                           'OC': ['O','C'],     # to 
                           'COH': ['C','OH'],   # bidentate
                           'OCH': ['O','CH'],   # on
                           'OCH2': ['O','CH2'], # rugged 
                           'OCH3': ['O','CH3'], # surfaces
                           'CHO': ['CH','O'],
                           'CHOH': ['CH','OH'],
                           'CH2O': ['CH2','O'],
                           'CH3O': ['CH3','O'],
                           'CH2OH': ['CH2','OH'],
                           'CH3OH': ['CH3','OH'],
                           'CHOOH': ['CH','O','OH'],
                           'COOH': ['C','O','OH'],
                           'CHOO': ['CH','O','O'],
                           'CO2': ['C','O','O']}

# Make your own adsorbate molecules
def adsorbate_molecule(adsorbate):
    # The ase.build.molecule module has many issues.       
    # Adjust positions, angles and indexing for your needs.
    if adsorbate  == 'CO':
        ads = molecule(adsorbate)[::-1]
    elif adsorbate == 'OH2':
        ads = molecule('H2O')
        ads.rotate(180, 'y')
    elif adsorbate == 'CH2':
        ads = molecule('NH2')
        ads[0].symbol = 'C'
        ads.rotate(180, 'y')
    elif adsorbate == 'COH':
        ads = molecule('H2COH')
        del ads[-2:]
        ads.rotate(90, 'y')
    elif adsorbate == 'CHO':
        ads = molecule('HCO')[[0,2,1]] 
    elif adsorbate == 'OCH2':
        ads = molecule('H2CO')
        ads.rotate(180, 'y')
    elif adsorbate == 'OCH3':
        ads = molecule('CH3O')[[1,0,2,3,4]]
        ads.rotate(90, '-x')
    elif adsorbate == 'CH2O':
        ads = molecule('H2CO')[[1,2,3,0]]
        ads.rotate(90, 'y')
    elif adsorbate == 'CH3O':
        ads = molecule('CH3O')[[0,2,3,4,1]]
        ads.rotate(30, 'y')
    elif adsorbate == 'CHOH':
        ads = molecule('H2COH')
        del ads[-1]
        ads = ads[[0,3,1,2]]
    elif adsorbate == 'CH2OH':
        ads = molecule('H2COH')[[0,3,4,1,2]]
    elif adsorbate == 'CH3OH':
        ads = molecule('CH3OH')[[0,2,4,5,1,3]]
        ads.rotate(-30, 'y')
    elif adsorbate == 'CHOOH':
        ads = molecule('HCOOH')[[1,4,2,0,3]]
    elif adsorbate == 'COOH':
        ads = molecule('HCOOH')
        del ads[-1]
        ads = ads[[1,2,0,3]]
        ads.rotate(90, '-x')
        ads.rotate(15, '-y')
    elif adsorbate == 'CHOO':
        ads = molecule('HCOOH')
        del ads[-2]
        ads = ads[[1,3,2,0]]
        ads.rotate(90, 'x')
        ads.rotate(15, 'y')
    elif adsorbate == 'CO2':
        ads = molecule(adsorbate)
        ads.rotate(-90, 'y')
    else:
        ads = molecule(adsorbate)
    return ads


class NanoparticleAdsorbateCoverage(NanoparticleAdsorptionSites):
    """dmax: maximum bond length [Ã] that should be considered as an adsorbate"""       

    def __init__(self, atoms, adsorption_sites=None, dmax=2.5):
 
        self.atoms = atoms.copy()
        self.positions = atoms.positions
        self.symbols = atoms.symbols
        self.ads_ids = [a.index for a in atoms if 
                        a.symbol in adsorbate_elements]
        assert len(self.ads_ids) > 0 
        self.ads_atoms = atoms[self.ads_ids]
        self.cell = atoms.cell
        self.pbc = atoms.pbc
        self.dmax = dmax

        self.make_ads_neighbor_list()
        self.ads_connectivity_matrix = self.get_ads_connectivity() 
        self.identify_adsorbates()

        if adsorption_sites:
            nas = adsorption_sites
        else:
            nas = NanoparticleAdsorptionSites(atoms, 
                                              allow_6fold=True,
                                              composition_effect=True,
                                              subsurf_effect=False)    
        self.nas = nas
        self.slab = nas.atoms
        self.allow_6fold = nas.allow_6fold
        self.composition_effect = nas.composition_effect
        if nas.subsurf_effect:
            raise NotImplementedError

        self.metals = nas.metals
        self.surf_ids = nas.surf_ids
        self.hetero_site_list = nas.site_list.copy()
        self.clean_list()
        self.unique_sites = nas.get_unique_sites(unique_composition=
                                                 self.composition_effect) 
        self.label_dict = self.get_bimetallic_label_dict() \
                          if self.composition_effect else \
                          self.get_monometallic_label_dict()

        self.label_list = ['0'] * len(self.hetero_site_list)
        self.site_connectivity_matrix = self.get_site_connectivity()
        self.label_occupied_sites()
        self.labels = self.get_labels()

    def identify_adsorbates(self):
        G = nx.Graph()
        adscm = self.ads_connectivity_matrix
      
        if adscm.size != 0:
            np.fill_diagonal(adscm, 1)
            rows, cols = np.where(adscm == 1)

            edges = zip([self.ads_ids[row] for row in rows.tolist()], 
                        [self.ads_ids[col] for col in cols.tolist()])
            G.add_edges_from(edges)                        
            SG = (G.subgraph(c) for c in nx.connected_components(G))
 
            adsorbates = []
            for sg in SG:
                nodes = sorted(list(sg.nodes))
                adsorbates += [nodes]
        else:
            adsorbates = [self.ads_ids]
        self.ads_list = adsorbates

    def clean_list(self):
        sl = self.hetero_site_list
        entries = ['occupied', 'adsorbate', 'adsorbate_indices', 
                   'fragment', 'fragment_indices', 'bonded_index', 
                   'bond_length', 'label', 'dentate']
        for d in sl:
            for k in entries:
                if k in d:
                    del d[k]

    def get_ads_connectivity(self):
        """Generate a connections matrix for adsorbate atoms."""
        return get_connectivity_matrix(self.ads_nblist) 

    def get_site_connectivity(self):
        """Generate a connections matrix for adsorption sites."""
        sl = self.hetero_site_list
        conn_mat = []
        for i, sti in enumerate(sl):
            conn_x = []
            for j, stj in enumerate(sl): 
                overlap = len(set(sti['indices']).intersection(stj['indices']))
                if i == j:
                    conn_x.append(0.)
                elif overlap > 0:
                    if self.allow_6fold: 
                        if '6fold' in [sti['site'], stj['site']]: 
                            if overlap == 3:                            
                                conn_x.append(1.)
                            else:
                                conn_x.append(0.)
                        else:
                            conn_x.append(1.)
                    else:
                        conn_x.append(1.)                     
                else:
                    conn_x.append(0.)
            conn_mat.append(conn_x)   

        return np.asarray(conn_mat) 

    def label_occupied_sites(self):
        hsl = self.hetero_site_list
        ll = self.label_list
        ads_list = self.ads_list
        ndentate_dict = {}
 
        for adsid in self.ads_ids:
            if self.symbols[adsid] == 'H':
                if [adsid] not in ads_list:
                    continue

            adspos = self.positions[adsid]
            bls = np.linalg.norm(np.asarray([s['position'] - 
                                 adspos for s in hsl]), axis=1)
            stid, bl = min(enumerate(bls), key=itemgetter(1))
            st = hsl[stid]
            if bl > self.dmax:
                continue

            adsids = next((l for l in ads_list if adsid in l), None)
            adsi = tuple(sorted(adsids))
            if 'occupied' in st:
                if bl >= st['bond_length']:
                    continue
                elif self.symbols[adsid] != 'H':
                    ndentate_dict[adsi] -= 1 
            st['bonded_index'] = adsid
            st['bond_length'] = bl

            symbols = str(self.symbols[adsids])
            adssym = next((k for k, v in adsorbate_formula_dict.items() 
                           if v == symbols), symbols)
            st['adsorbate'] = adssym
            st['fragment'] = adssym
            st['adsorbate_indices'] = adsi 
            if adsi in ndentate_dict:
                ndentate_dict[adsi] += 1
            else:
                ndentate_dict[adsi] = 1
            st['occupied'] = 1            

        # Get dentate numbers and coverage  
        self.n_occupied, n_surf_occupied, self.n_subsurf_occupied = 0, 0, 0
        for st in hsl:
            if 'occupied' not in st:
                st['bonded_index'] = st['bond_length'] = None
                st['adsorbate'] = st['fragment'] = None
                st['adsorbate_indices'] = None
                st['occupied'] = st['dentate'] = 0
                st['fragment_indices'] = None
                st['label'] = 0
                continue
            self.n_occupied += 1
            if st['site'] == '6fold':
                self.n_subsurf_occupied += 1
            else:
                n_surf_occupied += 1
            adsi = st['adsorbate_indices']
            if adsi in ndentate_dict:              
                st['dentate'] = ndentate_dict[adsi]
            else:
                st['dentate'] = 0
        self.coverage = n_surf_occupied / len(self.surf_ids)

        # Identify bidentate fragments and assign labels 
        for j, st in enumerate(hsl):
            if st['occupied'] == 1:
                if st['dentate'] > 1:
                    bondid = st['bonded_index']
                    bondsym = self.symbols[bondid]
                    adssym = st['adsorbate']
                    if adssym in adsorbate_fragment_dict:
                        fsym = next((f for f in adsorbate_fragment_dict[adssym] 
                                     if f[0] == bondsym), None)
                        st['fragment'] = fsym
                        flen = len(list(Formula(fsym)))
                        adsids = st['adsorbate_indices']
                        ibond = adsids.index(bondid)
                        fsi = adsids[ibond:ibond+flen]
                        st['fragment_indices'] = fsi
                    else:
                        st['fragment'] = adssym
                        st['fragment_indices'] = st['adsorbate_indices']
                else:
                    st['fragment_indices'] = st['adsorbate_indices'] 
                signature = [st['site'], st['surface']]                     
                if self.composition_effect:
                    signature.append(st['composition'])
                stlab = self.label_dict['|'.join(signature)]
                label = str(stlab) + st['fragment']
                st['label'] = label
                ll[j] = label

    def make_ads_neighbor_list(self, dx=.3, neighbor_number=1):
        """Generate a periodic neighbor list (defaultdict).""" 
        self.ads_nblist = neighbor_shell_list(self.ads_atoms, dx, 
                                              neighbor_number, mic=False)

    def get_labels(self):
        ll = self.label_list
        labs = [lab for lab in ll if lab != '0']
        return sorted(labs)

    def get_site_graph(self):                                         
        ll = self.label_list
        scm = self.site_connectivity_matrix

        G = nx.Graph()                                                  
        # Add nodes from label list
        G.add_nodes_from([(i, {'label': ll[i]}) for 
                           i in range(scm.shape[0])])
        # Add edges from surface connectivity matrix
        rows, cols = np.where(scm == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G.add_edges_from(edges)

        return G

    def get_subsurf_coverage(self):
        nsubsurf = len(self.nas.get_subsurface())
        return self.n_subsurf_occupied / nsubsurf

    # Use this label dictionary when site compostion is 
    # not considered. Useful for monometallic surfaces.
    def get_monometallic_label_dict(self):
        return {'ontop|vertex': 1,
                'ontop|edge': 2,
                'ontop|fcc111': 3,
                'ontop|fcc100': 4,
                'bridge|edge': 5,
                'bridge|fcc111': 6,
                'bridge|fcc100': 7,
                'fcc|fcc111': 8,
                'hcp|fcc111': 9,
                '4fold|fcc100': 10,
                '6fold|fcc111': 11}

    def get_bimetallic_label_dict(self): 
        ma, mb = self.metals[0], self.metals[1]
 
        return {'ontop|vertex|{}'.format(ma): 1, 
                'ontop|vertex|{}'.format(mb): 2,
                'ontop|edge|{}'.format(ma): 3,
                'ontop|edge|{}'.format(mb): 4,
                'ontop|fcc111|{}'.format(ma): 5,
                'ontop|fcc111|{}'.format(mb): 6,
                'ontop|fcc100|{}'.format(ma): 7,
                'ontop|fcc100|{}'.format(mb): 8,
                'bridge|edge|{}{}'.format(ma,ma): 9, 
                'bridge|edge|{}{}'.format(ma,mb): 10,
                'bridge|edge|{}{}'.format(mb,mb): 11,
                'bridge|fcc111|{}{}'.format(ma,ma): 12,
                'bridge|fcc111|{}{}'.format(ma,mb): 13,
                'bridge|fcc111|{}{}'.format(mb,mb): 14,
                'bridge|fcc100|{}{}'.format(ma,ma): 15,
                'bridge|fcc100|{}{}'.format(ma,mb): 16,
                'bridge|fcc100|{}{}'.format(mb,mb): 17,
                'fcc|fcc111|{}{}{}'.format(ma,ma,ma): 18,
                'fcc|fcc111|{}{}{}'.format(ma,ma,mb): 19, 
                'fcc|fcc111|{}{}{}'.format(ma,mb,mb): 20,
                'fcc|fcc111|{}{}{}'.format(mb,mb,mb): 21,
                'hcp|fcc111|{}{}{}'.format(ma,ma,ma): 22,
                'hcp|fcc111|{}{}{}'.format(ma,ma,mb): 23,
                'hcp|fcc111|{}{}{}'.format(ma,mb,mb): 24,
                'hcp|fcc111|{}{}{}'.format(mb,mb,mb): 25,
                '4fold|fcc100|{}{}{}{}'.format(ma,ma,ma,ma): 26,
                '4fold|fcc100|{}{}{}{}'.format(ma,ma,ma,mb): 27, 
                '4fold|fcc100|{}{}{}{}'.format(ma,ma,mb,mb): 28,
                '4fold|fcc100|{}{}{}{}'.format(ma,mb,ma,mb): 29, 
                '4fold|fcc100|{}{}{}{}'.format(ma,mb,mb,mb): 30,
                '4fold|fcc100|{}{}{}{}'.format(mb,mb,mb,mb): 31,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,ma): 32,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,mb): 33,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,ma,ma,mb,ma,ma): 34,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,ma,ma,ma,mb,mb): 35,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,ma): 36,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,mb): 37,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,ma): 38,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,mb): 39,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,ma,mb,mb,ma,ma): 40,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,ma,mb,ma,mb,mb): 41,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,ma): 42,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,mb): 43,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,ma): 44,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,mb): 45,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,mb,mb,mb,ma,ma): 46,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,mb,mb,ma,mb,mb): 47,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,ma): 48,
                '6fold|fcc111|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,mb): 49,
                '6fold|fcc111|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,ma): 50,
                '6fold|fcc111|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,mb): 51,
                '6fold|fcc111|{}{}{}{}{}{}'.format(mb,mb,mb,mb,ma,ma): 52,
                '6fold|fcc111|{}{}{}{}{}{}'.format(mb,mb,mb,ma,mb,mb): 53,
                '6fold|fcc111|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,ma): 54,
                '6fold|fcc111|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,mb): 55}

 
class SlabAdsorbateCoverage(SlabAdsorptionSites):

    """dmax: maximum bond length [Ã] that should be considered as an adsorbate"""        

    def __init__(self, atoms, adsorption_sites=None, surface=None, dmax=2.5):
 
        self.atoms = atoms.copy()
        self.positions = atoms.positions
        self.symbols = atoms.symbols
        self.ads_ids = [a.index for a in atoms if 
                        a.symbol in adsorbate_elements]
        assert len(self.ads_ids) > 0 
        self.ads_atoms = atoms[self.ads_ids]
        self.cell = atoms.cell
        self.pbc = atoms.pbc
        self.dmax = dmax

        self.make_ads_neighbor_list()
        self.ads_connectivity_matrix = self.get_ads_connectivity() 
        self.identify_adsorbates()
        if adsorption_sites:
            sas = adsorption_sites
        else:
            sas = SlabAdsorptionSites(atoms, surface, 
                                      allow_6fold=True,
                                      composition_effect=True,
                                      subsurf_effect=False)    
        self.sas = sas
        self.slab = sas.atoms
        self.surface = sas.surface
        self.allow_6fold = sas.allow_6fold
        self.composition_effect = sas.composition_effect
        if sas.subsurf_effect:
            raise NotImplementedError

        self.metals = sas.metals
        self.surf_ids = sas.surf_ids
        self.subsurf_ids = sas.subsurf_ids
        self.connectivity_matrix = sas.connectivity_matrix
        self.hetero_site_list = sas.site_list.copy()
        self.clean_list()
        self.unique_sites = sas.get_unique_sites(unique_composition=
                                                 self.composition_effect) 
        self.label_dict = self.get_bimetallic_label_dict() \
                          if self.composition_effect else \
                          self.get_monometallic_label_dict()

        self.label_list = ['0'] * len(self.hetero_site_list)
        self.site_connectivity_matrix = self.get_site_connectivity()
        self.label_occupied_sites()
        self.labels = self.get_labels()

    def identify_adsorbates(self):
        G = nx.Graph()
        adscm = self.ads_connectivity_matrix
      
        if adscm.size != 0:
            np.fill_diagonal(adscm, 1)
            rows, cols = np.where(adscm == 1)

            edges = zip([self.ads_ids[row] for row in rows.tolist()], 
                        [self.ads_ids[col] for col in cols.tolist()])
            G.add_edges_from(edges)                        
            SG = (G.subgraph(c) for c in nx.connected_components(G))
 
            adsorbates = []
            for sg in SG:
                nodes = sorted(list(sg.nodes))
                adsorbates += [nodes]
        else:
            adsorbates = [self.ads_ids]
        self.ads_list = adsorbates

    def clean_list(self):
        sl = self.hetero_site_list
        entries = ['occupied', 'adsorbate', 'adsorbate_indices', 
                   'fragment', 'fragment_indices', 'bonded_index', 
                   'bond_length', 'label', 'dentate']
        for d in sl:
            for k in entries:
                if k in d:
                    del d[k]

    def get_ads_connectivity(self):
        """Generate a connections matrix for adsorbate atoms."""
        return get_connectivity_matrix(self.ads_nblist) 

    def get_site_connectivity(self):
        """Generate a connections matrix for adsorption sites."""
        sl = self.hetero_site_list
        conn_mat = []
        for i, sti in enumerate(sl):
            conn_x = []
            for j, stj in enumerate(sl):
                overlap = len(set(sti['indices']).intersection(stj['indices']))
                if i == j:
                    conn_x.append(0.)
                elif overlap > 0:
                    if self.allow_6fold:         
                        if 'subsurf' in [sti['geometry'], stj['geometry']]: 
                            if overlap == 3:
                                conn_x.append(1.)
                            else:
                                conn_x.append(0.)
                        else:
                            conn_x.append(1.)
                    else:
                        conn_x.append(1.) 
                else:
                    conn_x.append(0.)
            conn_mat.append(conn_x)   

        return np.asarray(conn_mat) 

    def label_occupied_sites(self):
        hsl = self.hetero_site_list
        ll = self.label_list
        ads_list = self.ads_list
        ndentate_dict = {} 
        for adsid in self.ads_ids:
            if self.symbols[adsid] == 'H':
                if [adsid] not in ads_list:
                    continue

            adspos = self.positions[adsid]
            _, bls = find_mic(np.asarray([s['position'] - adspos for s in hsl]), 
                              cell=self.cell, pbc=True) 
            stid, bl = min(enumerate(bls), key=itemgetter(1))
            st = hsl[stid]
            if bl > self.dmax:
                continue

            adsids = next((l for l in ads_list if adsid in l), None)
            adsi = tuple(sorted(adsids))
            if 'occupied' in st:
                if bl >= st['bond_length']:
                    continue
                elif self.symbols[adsid] != 'H':
                    ndentate_dict[adsi] -= 1 
            st['bonded_index'] = adsid
            st['bond_length'] = bl

            symbols = str(self.symbols[adsids])
            adssym = next((k for k, v in adsorbate_formula_dict.items() 
                           if v == symbols), symbols)
            st['adsorbate'] = adssym
            st['fragment'] = adssym
            st['adsorbate_indices'] = adsi 
            if adsi in ndentate_dict:
                ndentate_dict[adsi] += 1
            else:
                ndentate_dict[adsi] = 1
            st['occupied'] = 1        

        # Get dentate numbers and coverage  
        self.n_occupied, n_surf_occupied, n_subsurf_occupied = 0, 0, 0
        for st in hsl:
            if 'occupied' not in st:
                st['bonded_index'] = st['bond_length'] = None
                st['adsorbate'] = st['fragment'] = None
                st['adsorbate_indices'] = None
                st['occupied'] = st['dentate'] = 0
                st['fragment_indices'] = None
                st['label'] = 0
                continue
            self.n_occupied += 1
            if st['geometry'] == 'subsurf':
                n_subsurf_occupied += 1
            else:
                n_surf_occupied += 1
            adsi = st['adsorbate_indices']
            if adsi in ndentate_dict:              
                st['dentate'] = ndentate_dict[adsi]
            else:
                st['dentate'] = 0
        self.coverage = n_surf_occupied / len(self.surf_ids)
        self.subsurf_coverage = n_subsurf_occupied / len(self.subsurf_ids)

        # Identify bidentate fragments and assign labels 
        for j, st in enumerate(hsl):
            if st['occupied'] == 1:
                if st['dentate'] > 1:
                    bondid = st['bonded_index']
                    bondsym = self.symbols[bondid] 
                    adssym = st['adsorbate']
                    if adssym in adsorbate_fragment_dict:
                        fsym = next((f for f in adsorbate_fragment_dict[adssym] 
                                     if f[0] == bondsym), None)
                        st['fragment'] = fsym
                        flen = len(list(Formula(fsym)))
                        adsids = st['adsorbate_indices']
                        ibond = adsids.index(bondid)
                        fsi = adsids[ibond:ibond+flen]
                        st['fragment_indices'] = fsi
                    else:
                        st['fragment'] = adssym
                        st['fragment_indices'] = st['adsorbate_indices']
                else:
                    st['fragment_indices'] = st['adsorbate_indices'] 
                signature = [st['site'], st['geometry']]                     
                if self.composition_effect:
                    signature.append(st['composition'])
                stlab = self.label_dict['|'.join(signature)]
                label = str(stlab) + st['fragment']
                st['label'] = label
                ll[j] = label

    def make_ads_neighbor_list(self, dx=.3, neighbor_number=1):
        """Generate a periodic neighbor list (defaultdict).""" 
        self.ads_nblist = neighbor_shell_list(self.ads_atoms, dx, 
                                              neighbor_number, mic=True)

    def get_labels(self):
        ll = self.label_list
        labs = [lab for lab in ll if lab != '0']
        return sorted(labs)

    def get_site_graph(self):                                         
        ll = self.label_list
        scm = self.site_connectivity_matrix

        G = nx.Graph()                                                  
        # Add nodes from label list
        G.add_nodes_from([(i, {'label': ll[i]}) for 
                           i in range(scm.shape[0])])
        # Add edges from surface connectivity matrix
        rows, cols = np.where(scm == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G.add_edges_from(edges)

        return G

    def get_surface_bond_count_matrix(self, species):
        hsl = self.hetero_site_list
        cm = self.connectivity_matrix
        atoms = self.atoms
        numbers = atoms.numbers
        symbols = atoms.symbols
        specs = species
        specs.sort(key=lambda x: atomic_numbers[x])
        ncols = len(specs) + 1
        sbcm = np.zeros((len(atoms), ncols))
        for st in hsl:
            frags = list(Formula(st['fragment']))
            counts = Counter(frags)
            for i in st['indices']:
                for j, spec in enumerate(specs):
                    sbcm[i,j] += counts[spec]
        top_ids = self.surf_ids + self.subsurf_ids if \
                  self.allow_6fold else self.surf_ids
        for si in top_ids:
            nbids = np.where(cm[si]==1)[0]
            nbs = [symbols[i] for i in nbids]
            nbcounts = Counter(nbs)
            for j, spec in enumerate(specs):
                sbcm[si,j] += nbcounts[spec]
            sbcm[si,ncols-1] = numbers[si] 

        return sbcm[top_ids]

    # Use this label dictionary when site compostion is 
    # not considered. Useful for monometallic surfaces.
    def get_monometallic_label_dict(self): 
        if self.surface in ['fcc111','hcp0001']:
            return {'ontop|terrace': 1,
                    'bridge|terrace': 2,
                    'fcc|terrace': 3,
                    'hcp|terrace': 4,
                    '6fold|subsurf': 5}
    
        elif self.surface in ['fcc100','bcc100']:
            return {'ontop|terrace': 1,
                    'bridge|terrace': 2,
                    '4fold|terrace': 3}
    
        elif self.surface in ['fcc110','hcp10m10-h']:
            return {'ontop|step': 1,
                    'bridge|step': 2, 
                    'bridge|sc-tc-h': 3,
                    'fcc|sc-tc-h': 4,
                    'hcp|sc-tc-h': 5,
                    '4fold|terrace': 6,
                    '5fold|terrace': 7,
                    '6fold|subsurf': 8}
    
        elif self.surface == 'fcc211':
            return {'ontop|step': 1,      
                    'ontop|terrace': 2,
                    'ontop|corner': 3, 
                    'bridge|step': 4,
                    'bridge|corner': 5,
                    'bridge|sc-tc-h': 6,
                    'bridge|tc-cc-h': 7,
                    'bridge|sc-cc-t': 8,
                    'fcc|sc-tc-h': 9,
                    'hcp|sc-tc-h': 10,
                    'fcc|tc-cc-h': 11,
                    'hcp|tc-cc-h': 12,
                    '4fold|sc-cc-t': 13,
                    '6fold|subsurf': 14}
    
        elif self.surface == 'fcc311':
            return {'ontop|step': 1,
                    'ontop|terrace': 2,
                    'bridge|step': 3,
                    'bridge|terrace': 4,
                    'bridge|sc-tc-h': 5,
                    'bridge|sc-tc-t': 6,
                    'fcc|sc-tc-h': 7,
                    'hcp|sc-tc-h': 8,
                    '4fold|sc-tc-t': 9,
                    '6fold|subsurf': 10}

        elif self.surface == 'fcc322':
            return {'ontop|step': 1,
                    'ontop|terrace': 2,
                    'ontop|corner': 3,
                    'bridge|step': 4,
                    'bridge|terrace': 5,
                    'bridge|corner': 6,
                    'bridge|sc-tc-h': 7,
                    'bridge|tc-cc-h': 8,
                    'bridge|sc-cc-t': 9,
                    'fcc|terrace': 10,
                    'hcp|terrace': 11,
                    'fcc|sc-tc-h': 12,
                    'hcp|sc-tc-h': 13,                    
                    'fcc|tc-cc-h': 14,
                    'hcp|tc-cc-h': 15,
                    '4fold|sc-cc-t': 16,
                    '6fold|subsurf': 17}

        elif self.surface in ['fcc221','fcc332']:
            return {'ontop|step': 1,
                    'ontop|terrace': 2,
                    'ontop|corner': 3,
                    'bridge|step': 4,
                    'bridge|terrace': 5,
                    'bridge|corner': 6,
                    'bridge|sc-tc-h': 7,
                    'bridge|tc-cc-h': 8,
                    'bridge|sc-cc-h': 9,
                    'fcc|terrace': 10,
                    'hcp|terrace': 11,
                    'fcc|sc-tc-h': 12,
                    'hcp|sc-tc-h': 13, 
                    'fcc|tc-cc-h': 14,
                    'hcp|tc-cc-h': 15,
                    'fcc|sc-cc-h': 16,
                    'hcp|sc-cc-h': 17,
                    '6fold|subsurf': 18}

        elif self.surface == 'fcc331':
            return {'ontop|step': 1,
                    'ontop|terrace': 2,
                    'bridge|step': 3,
                    'bridge|sc-tc-h': 4,
                    'bridge|tc-cc-h': 5,
                    'bridge|sc-cc-h': 6,
                    'fcc|sc-tc-h': 7,
                    'hcp|sc-tc-h': 8,
                    'fcc|tc-cc-h': 9,
                    'hcp|tc-cc-h': 10,
                    'fcc|sc-cc-h': 11,
                    'hcp|sc-cc-h': 12,
                    '4fold|corner': 13,
                    '5fold|corner': 14,
                    '6fold|subsurf': 15}

        elif self.surface == 'bcc110':
            return {'ontop|terrace': 1,
                    'long-bridge|terrace': 2,
                    'short-bridge|terrace': 3,
                    '3fold|terrace': 4}
                  
        elif self.surface == 'bcc111':           
            return {'ontop|step': 1,                       
                    'ontop|terrace': 2,        
                    'ontop|corner': 3,
                    'short-bridge|sc-tc-o': 4,
                    'short-bridge|tc-cc-o': 5,
                    'long-bridge|sc-cc-o': 6,
                    '3fold|sc-tc-cc-o': 7}

        elif self.surface == 'bcc210':
            return {'ontop|step': 1,     
                    'ontop|terrace': 2,
                    'ontop|corner': 3, 
                    'bridge|step': 4,
                    'bridge|terrace': 5,
                    'bridge|corner': 6,
                    'bridge|sc-tc-o': 7,
                    'bridge|tc-cc-o': 8,
                    'bridge|sc-cc-t': 9,
                    '3fold|sc-tc-o': 10,
                    '3fold|tc-cc-o': 11,
                    '4fold|sc-cc-t': 12}

        elif self.surface == 'bcc211':
            return {'ontop|step': 1,
                    'bridge|step': 2, 
                    'bridge|sc-tc-o': 3,
                    '3fold|sc-tc-o': 4,
                    '4fold|terrace': 5,
                    '5fold|terrace': 6}

        elif self.surface == 'bcc310':
            return {'ontop|step': 1,
                    'ontop|terrace': 2,
                    'bridge|step': 3,
                    'bridge|terrace': 4,
                    'bridge|sc-tc-o': 5,
                    'bridge|sc-tc-t': 6,
                    '3fold|sc-tc-o': 7,
                    '4fold|sc-tc-t': 8}

        elif self.surface == 'hcp10m10-t':
            return {'ontop|step': 1,
                    'ontop|terrace': 2,
                    'bridge|step': 3,
                    'bridge|terrace': 4,
                    'bridge|sc-tc-t': 5,
                    '5fold|subsurf': 6,}

        elif self.surface == 'hcp10m11':
            return {'ontop|step': 1,
                    'ontop|terrace': 2,
                    'bridge|step': 3,
                    'bridge|terrace': 4,
                    'bridge|sc-tc-h': 5,
                    'fcc|sc-tc-h': 6,
                    'hcp|sc-tc-h': 7,
                    '4fold|subsurf': 8,
                    '5fold|subsurf': 9,
                    '6fold|subsurf': 10}
 
        elif self.surface == 'hcp10m12':       
            return {'ontop|step': 1,
                    'ontop|terrace': 2,
                    'ontop|corner': 3,
                    'bridge|step': 4,
                    'bridge|terrace': 5,
                    'bridge|corner': 6,
                    'bridge|sc-tc-h': 7,
                    'bridge|tc-cc-t': 8,
                    'bridge|sc-cc-h': 9,
                    'fcc|sc-tc-h': 10,
                    'hcp|sc-tc-h': 11,
                    'fcc|sc-cc-h': 12,
                    'hcp|sc-cc-h': 13,
                    '4fold|tc-cc-t': 14,
                    '6fold|subsurf': 15}
 
    def get_bimetallic_label_dict(self): 
        ma, mb = self.metals[0], self.metals[1]
 
        if self.surface in ['fcc111','hcp0001']:
            return {'ontop|terrace|{}'.format(ma): 1, 
                    'ontop|terrace|{}'.format(mb): 2,
                    'bridge|terrace|{}{}'.format(ma,ma): 3, 
                    'bridge|terrace|{}{}'.format(ma,mb): 4,
                    'bridge|terrace|{}{}'.format(mb,mb): 5, 
                    'fcc|terrace|{}{}{}'.format(ma,ma,ma): 6,
                    'fcc|terrace|{}{}{}'.format(ma,ma,mb): 7, 
                    'fcc|terrace|{}{}{}'.format(ma,mb,mb): 8,
                    'fcc|terrace|{}{}{}'.format(mb,mb,mb): 9,
                    'hcp|terrace|{}{}{}'.format(ma,ma,ma): 10,
                    'hcp|terrace|{}{}{}'.format(ma,ma,mb): 11,
                    'hcp|terrace|{}{}{}'.format(ma,mb,mb): 12,
                    'hcp|terrace|{}{}{}'.format(mb,mb,mb): 13,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,ma): 14,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,mb): 15,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,ma,ma): 16,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,mb,mb): 17,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,ma): 18,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,mb): 19,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,ma): 20,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,mb): 21,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,ma,ma): 22,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,mb,mb): 23,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,ma): 24,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,mb): 25,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,ma): 26,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,mb): 27,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,ma,ma): 28,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,mb,mb): 29,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,ma): 30,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,mb): 31,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,ma): 32,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,mb): 33,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,ma,ma): 34,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,mb,mb): 35,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,ma): 36,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,mb): 37}
 
        elif self.surface in ['fcc100','bcc100']:
            return {'ontop|terrace|{}'.format(ma): 1, 
                    'ontop|terrace|{}'.format(mb): 2,
                    'bridge|terrace|{}{}'.format(ma,ma): 3, 
                    'bridge|terrace|{}{}'.format(ma,mb): 4,
                    'bridge|terrace|{}{}'.format(mb,mb): 5, 
                    '4fold|terrace|{}{}{}{}'.format(ma,ma,ma,ma): 6,
                    '4fold|terrace|{}{}{}{}'.format(ma,ma,ma,mb): 7, 
                    '4fold|terrace|{}{}{}{}'.format(ma,ma,mb,mb): 8,
                    '4fold|terrace|{}{}{}{}'.format(ma,mb,ma,mb): 9, 
                    '4fold|terrace|{}{}{}{}'.format(ma,mb,mb,mb): 10,
                    '4fold|terrace|{}{}{}{}'.format(mb,mb,mb,mb): 11}
    
        elif self.surface in ['fcc110','hcp10m10-h']:
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'bridge|step|{}{}'.format(ma,ma): 3,
                    'bridge|step|{}{}'.format(ma,mb): 4,
                    'bridge|step|{}{}'.format(mb,mb): 5,
                    'bridge|sc-tc-h|{}{}'.format(ma,ma): 6,
                    'bridge|sc-tc-h|{}{}'.format(ma,mb): 7,
                    'bridge|sc-tc-h|{}{}'.format(mb,mb): 8,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,ma): 9,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,mb): 10, 
                    'fcc|sc-tc-h|{}{}{}'.format(ma,mb,mb): 11,
                    'fcc|sc-tc-h|{}{}{}'.format(mb,mb,mb): 12,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,ma): 13,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,mb): 14,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,mb,mb): 15,
                    'hcp|sc-tc-h|{}{}{}'.format(mb,mb,mb): 16,
                    '4fold|terrace|{}{}-{}{}'.format(ma,ma,ma,ma): 17,
                    '4fold|terrace|{}{}-{}{}'.format(ma,ma,ma,mb): 18,
                    '4fold|terrace|{}{}-{}{}'.format(ma,ma,mb,mb): 19,
                    '4fold|terrace|{}{}-{}{}'.format(ma,mb,ma,ma): 20,
                    '4fold|terrace|{}{}-{}{}'.format(ma,mb,ma,mb): 21,
                    '4fold|terrace|{}{}-{}{}'.format(ma,mb,mb,mb): 22,
                    '4fold|terrace|{}{}-{}{}'.format(mb,mb,ma,ma): 23,
                    '4fold|terrace|{}{}-{}{}'.format(mb,mb,ma,mb): 24,
                    '4fold|terrace|{}{}-{}{}'.format(mb,mb,mb,mb): 25,
                    # neighbor elements count clockwise from shorter bond ma
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,ma,ma,ma,ma): 26,
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,ma,ma,ma,mb): 27,
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,ma,ma,mb,mb): 28,
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,ma,mb,ma,mb): 29,
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,ma,mb,mb,ma): 30,
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,ma,mb,mb,mb): 31,
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,mb,mb,mb,mb): 32,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,ma,ma,ma,ma): 33,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,ma,ma,ma,mb): 34,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,ma,ma,mb,mb): 35,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,ma,mb,ma,mb): 36,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,ma,mb,mb,ma): 37,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,ma,mb,mb,mb): 38,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,mb,mb,mb,mb): 39,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,ma): 40,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,mb): 41,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,ma,ma): 42,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,mb,mb): 43,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,ma): 44,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,mb): 45,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,ma): 46,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,mb): 47,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,ma,ma): 48,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,mb,mb): 49,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,ma): 50,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,mb): 51,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,ma): 52,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,mb): 53,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,ma,ma): 54,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,mb,mb): 55,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,ma): 56,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,mb): 57,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,ma): 58,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,mb): 59,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,ma,ma): 60,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,mb,mb): 61,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,ma): 62,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,mb): 63}
    
        elif self.surface == 'fcc211':
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'ontop|terrace|{}'.format(ma): 3,
                    'ontop|terrace|{}'.format(mb): 4,
                    'ontop|corner|{}'.format(mb): 5,
                    'ontop|corner|{}'.format(mb): 6,
                    'bridge|step|{}{}'.format(ma,ma): 7, 
                    'bridge|step|{}{}'.format(ma,mb): 8,
                    'bridge|step|{}{}'.format(mb,mb): 9,
                    'bridge|corner|{}{}'.format(ma,ma): 10,
                    'bridge|corner|{}{}'.format(ma,mb): 11,
                    'bridge|corner|{}{}'.format(mb,mb): 12,
                    'bridge|sc-tc-h|{}{}'.format(ma,ma): 13,
                    'bridge|sc-tc-h|{}{}'.format(ma,mb): 14,
                    'bridge|sc-tc-h|{}{}'.format(mb,mb): 15,
                    # terrace bridge is equivalent to tc-cc-h bridge
                    'bridge|tc-cc-h|{}{}'.format(ma,ma): 16,
                    'bridge|tc-cc-h|{}{}'.format(ma,mb): 17,
                    'bridge|tc-cc-h|{}{}'.format(mb,mb): 18,
                    'bridge|sc-cc-t|{}{}'.format(ma,ma): 19,
                    'bridge|sc-cc-t|{}{}'.format(ma,mb): 20,
                    'bridge|sc-cc-t|{}{}'.format(mb,mb): 21,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,ma): 22,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,mb): 23, 
                    'fcc|sc-tc-h|{}{}{}'.format(ma,mb,mb): 24,
                    'fcc|sc-tc-h|{}{}{}'.format(mb,mb,mb): 25,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,ma): 26,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,mb): 27,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,mb,mb): 28,
                    'hcp|sc-tc-h|{}{}{}'.format(mb,mb,mb): 29,
                    'fcc|tc-cc-h|{}{}{}'.format(ma,ma,ma): 30,
                    'fcc|tc-cc-h|{}{}{}'.format(ma,ma,mb): 31,
                    'fcc|tc-cc-h|{}{}{}'.format(ma,mb,mb): 32,
                    'fcc|tc-cc-h|{}{}{}'.format(mb,mb,mb): 33,
                    'hcp|tc-cc-h|{}{}{}'.format(ma,ma,ma): 34,
                    'hcp|tc-cc-h|{}{}{}'.format(ma,ma,mb): 35,
                    'hcp|tc-cc-h|{}{}{}'.format(ma,mb,mb): 36,
                    'hcp|tc-cc-h|{}{}{}'.format(mb,mb,mb): 37,
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,ma,ma,ma): 38,
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,ma,ma,mb): 39, 
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,ma,mb,mb): 40,
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,mb,ma,mb): 41, 
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,mb,mb,mb): 42,
                    '4fold|sc-cc-t|{}{}{}{}'.format(mb,mb,mb,mb): 43,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,ma): 44,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,mb): 45,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,ma,ma): 46,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,mb,mb): 47,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,ma): 48,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,mb): 49,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,ma): 50,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,mb): 51,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,ma,ma): 52,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,mb,mb): 53,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,ma): 54,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,mb): 55,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,ma): 56,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,mb): 57,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,ma,ma): 58,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,mb,mb): 59,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,ma): 60,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,mb): 61,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,ma): 62,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,mb): 63,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,ma,ma): 64,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,mb,mb): 65,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,ma): 66,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,mb): 67}
                     
        elif self.surface == 'fcc311':
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'ontop|terrace|{}'.format(ma): 3,
                    'ontop|terrace|{}'.format(mb): 4,
                    'bridge|step|{}{}'.format(ma,ma): 5,
                    'bridge|step|{}{}'.format(ma,mb): 6,
                    'bridge|step|{}{}'.format(mb,mb): 7,
                    'bridge|terrace|{}{}'.format(ma,ma): 8,
                    'bridge|terrace|{}{}'.format(ma,mb): 9,
                    'bridge|terrace|{}{}'.format(mb,mb): 10,
                    'bridge|sc-tc-h|{}{}'.format(ma,ma): 11,
                    'bridge|sc-tc-h|{}{}'.format(ma,mb): 12,
                    'bridge|sc-tc-h|{}{}'.format(mb,mb): 13,
                    'bridge|sc-tc-t|{}{}'.format(ma,ma): 14,
                    'bridge|sc-tc-t|{}{}'.format(ma,mb): 15,
                    'bridge|sc-tc-t|{}{}'.format(mb,mb): 16,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,ma): 17,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,mb): 18,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,mb,mb): 19,
                    'fcc|sc-tc-h|{}{}{}'.format(mb,mb,mb): 20,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,ma): 21,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,mb): 22,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,mb,mb): 23,
                    'hcp|sc-tc-h|{}{}{}'.format(mb,mb,mb): 24,
                    '4fold|sc-tc-t|{}{}{}{}'.format(ma,ma,ma,ma): 25,
                    '4fold|sc-tc-t|{}{}{}{}'.format(ma,ma,ma,mb): 26, 
                    '4fold|sc-tc-t|{}{}{}{}'.format(ma,ma,mb,mb): 27,
                    '4fold|sc-tc-t|{}{}{}{}'.format(ma,mb,ma,mb): 28, 
                    '4fold|sc-tc-t|{}{}{}{}'.format(ma,mb,mb,mb): 29,
                    '4fold|sc-tc-t|{}{}{}{}'.format(mb,mb,mb,mb): 30,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,ma): 31,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,mb): 32,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,ma,ma): 33,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,mb,mb): 34,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,ma): 35,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,mb): 36,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,ma): 37,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,mb): 38,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,ma,ma): 39,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,mb,mb): 40,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,ma): 41,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,mb): 42,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,ma): 43,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,mb): 44,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,ma,ma): 45,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,mb,mb): 46,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,ma): 47,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,mb): 48,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,ma): 49,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,mb): 50,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,ma,ma): 51,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,mb,mb): 52,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,ma): 53,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,mb): 54}

        elif self.surface == 'fcc322':                                                      
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'ontop|terrace|{}'.format(ma): 3,
                    'ontop|terrace|{}'.format(mb): 4,
                    'ontop|corner|{}'.format(mb): 5,
                    'ontop|corner|{}'.format(mb): 6,
                    'bridge|step|{}{}'.format(ma,ma): 7, 
                    'bridge|step|{}{}'.format(ma,mb): 8,
                    'bridge|step|{}{}'.format(mb,mb): 9,
                    'bridge|terrace|{}{}'.format(ma,ma): 10,
                    'bridge|terrace|{}{}'.format(ma,mb): 11,
                    'bridge|terrace|{}{}'.format(mb,mb): 12,
                    'bridge|corner|{}{}'.format(ma,ma): 13,
                    'bridge|corner|{}{}'.format(ma,mb): 14,
                    'bridge|corner|{}{}'.format(mb,mb): 15,
                    'bridge|sc-tc-h|{}{}'.format(ma,ma): 16,
                    'bridge|sc-tc-h|{}{}'.format(ma,mb): 17,
                    'bridge|sc-tc-h|{}{}'.format(mb,mb): 18,                    
                    'bridge|tc-cc-h|{}{}'.format(ma,ma): 19,
                    'bridge|tc-cc-h|{}{}'.format(ma,mb): 20,
                    'bridge|tc-cc-h|{}{}'.format(mb,mb): 21,
                    'bridge|sc-cc-t|{}{}'.format(ma,ma): 22,
                    'bridge|sc-cc-t|{}{}'.format(ma,mb): 23,
                    'bridge|sc-cc-t|{}{}'.format(mb,mb): 24,
                    'fcc|terrace|{}{}{}'.format(ma,ma,ma): 25,
                    'fcc|terrace|{}{}{}'.format(ma,ma,mb): 26,
                    'fcc|terrace|{}{}{}'.format(ma,mb,mb): 27,
                    'fcc|terrace|{}{}{}'.format(mb,mb,mb): 28,
                    'hcp|terrace|{}{}{}'.format(ma,ma,ma): 29,
                    'hcp|terrace|{}{}{}'.format(ma,ma,mb): 30,
                    'hcp|terrace|{}{}{}'.format(ma,mb,mb): 31,
                    'hcp|terrace|{}{}{}'.format(mb,mb,mb): 32,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,ma): 33,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,mb): 34, 
                    'fcc|sc-tc-h|{}{}{}'.format(ma,mb,mb): 35,
                    'fcc|sc-tc-h|{}{}{}'.format(mb,mb,mb): 36,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,ma): 37,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,mb): 38,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,mb,mb): 39,
                    'hcp|sc-tc-h|{}{}{}'.format(mb,mb,mb): 40,
                    'fcc|tc-cc-h|{}{}{}'.format(ma,ma,ma): 41,
                    'fcc|tc-cc-h|{}{}{}'.format(ma,ma,mb): 42,
                    'fcc|tc-cc-h|{}{}{}'.format(ma,mb,mb): 43,
                    'fcc|tc-cc-h|{}{}{}'.format(mb,mb,mb): 44,
                    'hcp|tc-cc-h|{}{}{}'.format(ma,ma,ma): 45,
                    'hcp|tc-cc-h|{}{}{}'.format(ma,ma,mb): 46,
                    'hcp|tc-cc-h|{}{}{}'.format(ma,mb,mb): 47,
                    'hcp|tc-cc-h|{}{}{}'.format(mb,mb,mb): 48,
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,ma,ma,ma): 49,
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,ma,ma,mb): 50, 
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,ma,mb,mb): 51,
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,mb,ma,mb): 52, 
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,mb,mb,mb): 53,
                    '4fold|sc-cc-t|{}{}{}{}'.format(mb,mb,mb,mb): 54,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,ma): 55,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,mb): 56,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,ma,ma): 57,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,mb,mb): 58,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,ma): 59,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,mb): 60,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,ma): 61,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,mb): 62,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,ma,ma): 63,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,mb,mb): 64,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,ma): 65,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,mb): 66,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,ma): 67,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,mb): 68,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,ma,ma): 69,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,mb,mb): 70,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,ma): 71,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,mb): 72,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,ma): 73,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,mb): 74,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,ma,ma): 75,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,mb,mb): 76,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,ma): 77,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,mb): 78}

        elif self.surface in ['fcc221','fcc332']:                                          
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'ontop|terrace|{}'.format(ma): 3,
                    'ontop|terrace|{}'.format(mb): 4,
                    'ontop|corner|{}'.format(mb): 5,
                    'ontop|corner|{}'.format(mb): 6,
                    'bridge|step|{}{}'.format(ma,ma): 7, 
                    'bridge|step|{}{}'.format(ma,mb): 8,
                    'bridge|step|{}{}'.format(mb,mb): 9,
                    'bridge|terrace|{}{}'.format(ma,ma): 10,
                    'bridge|terrace|{}{}'.format(ma,mb): 11,
                    'bridge|terrace|{}{}'.format(mb,mb): 12,
                    'bridge|corner|{}{}'.format(ma,ma): 13,
                    'bridge|corner|{}{}'.format(ma,mb): 14,
                    'bridge|corner|{}{}'.format(mb,mb): 15,
                    'bridge|sc-tc-h|{}{}'.format(ma,ma): 16,
                    'bridge|sc-tc-h|{}{}'.format(ma,mb): 17,
                    'bridge|sc-tc-h|{}{}'.format(mb,mb): 18, 
                    'bridge|tc-cc-h|{}{}'.format(ma,ma): 19,
                    'bridge|tc-cc-h|{}{}'.format(ma,mb): 20,
                    'bridge|tc-cc-h|{}{}'.format(mb,mb): 21,
                    'bridge|sc-cc-h|{}{}'.format(ma,ma): 22,
                    'bridge|sc-cc-h|{}{}'.format(ma,mb): 23,
                    'bridge|sc-cc-h|{}{}'.format(mb,mb): 24,
                    'fcc|terrace|{}{}{}'.format(ma,ma,ma): 25,
                    'fcc|terrace|{}{}{}'.format(ma,ma,mb): 26,
                    'fcc|terrace|{}{}{}'.format(ma,mb,mb): 27,
                    'fcc|terrace|{}{}{}'.format(mb,mb,mb): 28,
                    'hcp|terrace|{}{}{}'.format(ma,ma,ma): 29,
                    'hcp|terrace|{}{}{}'.format(ma,ma,mb): 30,
                    'hcp|terrace|{}{}{}'.format(ma,mb,mb): 31,
                    'hcp|terrace|{}{}{}'.format(mb,mb,mb): 32,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,ma): 33,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,mb): 34, 
                    'fcc|sc-tc-h|{}{}{}'.format(ma,mb,mb): 35,
                    'fcc|sc-tc-h|{}{}{}'.format(mb,mb,mb): 36,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,ma): 37,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,mb): 38,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,mb,mb): 39,
                    'hcp|sc-tc-h|{}{}{}'.format(mb,mb,mb): 40,
                    'fcc|tc-cc-h|{}{}{}'.format(ma,ma,ma): 41,
                    'fcc|tc-cc-h|{}{}{}'.format(ma,ma,mb): 42,
                    'fcc|tc-cc-h|{}{}{}'.format(ma,mb,mb): 43,
                    'fcc|tc-cc-h|{}{}{}'.format(mb,mb,mb): 44,
                    'hcp|tc-cc-h|{}{}{}'.format(ma,ma,ma): 45,
                    'hcp|tc-cc-h|{}{}{}'.format(ma,ma,mb): 46,
                    'hcp|tc-cc-h|{}{}{}'.format(ma,mb,mb): 47,
                    'hcp|tc-cc-h|{}{}{}'.format(mb,mb,mb): 48,
                    'fcc|sc-cc-h|{}{}{}'.format(ma,ma,ma): 49,
                    'fcc|sc-cc-h|{}{}{}'.format(ma,ma,mb): 50,
                    'fcc|sc-cc-h|{}{}{}'.format(ma,mb,mb): 51,
                    'fcc|sc-cc-h|{}{}{}'.format(mb,mb,mb): 52,
                    'hcp|sc-cc-h|{}{}{}'.format(ma,ma,ma): 53,
                    'hcp|sc-cc-h|{}{}{}'.format(ma,ma,mb): 54,
                    'hcp|sc-cc-h|{}{}{}'.format(ma,mb,mb): 55,
                    'hcp|sc-cc-h|{}{}{}'.format(mb,mb,mb): 56,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,ma): 57,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,mb): 58,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,ma,ma): 59,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,mb,mb): 60,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,ma): 61,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,mb): 62,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,ma): 63,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,mb): 64,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,ma,ma): 65,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,mb,mb): 66,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,ma): 67,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,mb): 68,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,ma): 69,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,mb): 70,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,ma,ma): 71,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,mb,mb): 72,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,ma): 73,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,mb): 74,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,ma): 75,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,mb): 76,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,ma,ma): 77,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,mb,mb): 78,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,ma): 79,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,mb): 80}

        elif self.surface == 'fcc331':                               
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'ontop|terrace|{}'.format(ma): 3,
                    'ontop|terrace|{}'.format(mb): 4,
                    'bridge|step|{}{}'.format(ma,ma): 5, 
                    'bridge|step|{}{}'.format(ma,mb): 6,
                    'bridge|step|{}{}'.format(mb,mb): 7,
                    'bridge|sc-tc-h|{}{}'.format(ma,ma): 8,
                    'bridge|sc-tc-h|{}{}'.format(ma,mb): 9,
                    'bridge|sc-tc-h|{}{}'.format(mb,mb): 10,
                    'bridge|tc-cc-h|{}{}'.format(ma,ma): 11,
                    'bridge|tc-cc-h|{}{}'.format(ma,mb): 12,
                    'bridge|tc-cc-h|{}{}'.format(mb,mb): 13,
                    'bridge|sc-cc-h|{}{}'.format(ma,ma): 14,
                    'bridge|sc-cc-h|{}{}'.format(ma,mb): 15,
                    'bridge|sc-cc-h|{}{}'.format(mb,mb): 16,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,ma): 17,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,mb): 18, 
                    'fcc|sc-tc-h|{}{}{}'.format(ma,mb,mb): 19,
                    'fcc|sc-tc-h|{}{}{}'.format(mb,mb,mb): 20,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,ma): 21,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,mb): 22,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,mb,mb): 23,
                    'hcp|sc-tc-h|{}{}{}'.format(mb,mb,mb): 24,
                    'fcc|tc-cc-h|{}{}{}'.format(ma,ma,ma): 25,
                    'fcc|tc-cc-h|{}{}{}'.format(ma,ma,mb): 26,
                    'fcc|tc-cc-h|{}{}{}'.format(ma,mb,mb): 27,
                    'fcc|tc-cc-h|{}{}{}'.format(mb,mb,mb): 28,
                    'hcp|tc-cc-h|{}{}{}'.format(ma,ma,ma): 29,
                    'hcp|tc-cc-h|{}{}{}'.format(ma,ma,mb): 30,
                    'hcp|tc-cc-h|{}{}{}'.format(ma,mb,mb): 31,
                    'hcp|tc-cc-h|{}{}{}'.format(mb,mb,mb): 32,
                    'fcc|sc-cc-h|{}{}{}'.format(ma,ma,ma): 33,
                    'fcc|sc-cc-h|{}{}{}'.format(ma,ma,mb): 34,
                    'fcc|sc-cc-h|{}{}{}'.format(ma,mb,mb): 35,
                    'fcc|sc-cc-h|{}{}{}'.format(mb,mb,mb): 36,
                    'hcp|sc-cc-h|{}{}{}'.format(ma,ma,ma): 37,
                    'hcp|sc-cc-h|{}{}{}'.format(ma,ma,mb): 38,
                    'hcp|sc-cc-h|{}{}{}'.format(ma,mb,mb): 39,
                    'hcp|sc-cc-h|{}{}{}'.format(mb,mb,mb): 40,
                    '4fold|corner|{}{}-{}{}'.format(ma,ma,ma,ma): 41,
                    '4fold|corner|{}{}-{}{}'.format(ma,ma,ma,mb): 42,
                    '4fold|corner|{}{}-{}{}'.format(ma,ma,mb,mb): 43,
                    '4fold|corner|{}{}-{}{}'.format(ma,mb,ma,ma): 44,
                    '4fold|corner|{}{}-{}{}'.format(ma,mb,ma,mb): 45,
                    '4fold|corner|{}{}-{}{}'.format(ma,mb,mb,mb): 46,
                    '4fold|corner|{}{}-{}{}'.format(mb,mb,ma,ma): 47,
                    '4fold|corner|{}{}-{}{}'.format(mb,mb,ma,mb): 48,
                    '4fold|corner|{}{}-{}{}'.format(mb,mb,mb,mb): 49,
                    # neighbor elements count clockwise from shorter bond ma
                    '5fold|corner|{}-{}{}{}{}'.format(ma,ma,ma,ma,ma): 50,
                    '5fold|corner|{}-{}{}{}{}'.format(ma,ma,ma,ma,mb): 51,
                    '5fold|corner|{}-{}{}{}{}'.format(ma,ma,ma,mb,mb): 52,
                    '5fold|corner|{}-{}{}{}{}'.format(ma,ma,mb,ma,mb): 53,
                    '5fold|corner|{}-{}{}{}{}'.format(ma,ma,mb,mb,ma): 54,
                    '5fold|corner|{}-{}{}{}{}'.format(ma,ma,mb,mb,mb): 55,
                    '5fold|corner|{}-{}{}{}{}'.format(ma,mb,mb,mb,mb): 56,
                    '5fold|corner|{}-{}{}{}{}'.format(mb,ma,ma,ma,ma): 57,
                    '5fold|corner|{}-{}{}{}{}'.format(mb,ma,ma,ma,mb): 58,
                    '5fold|corner|{}-{}{}{}{}'.format(mb,ma,ma,mb,mb): 59,
                    '5fold|corner|{}-{}{}{}{}'.format(mb,ma,mb,ma,mb): 60,
                    '5fold|corner|{}-{}{}{}{}'.format(mb,ma,mb,mb,ma): 61,
                    '5fold|corner|{}-{}{}{}{}'.format(mb,ma,mb,mb,mb): 62,
                    '5fold|corner|{}-{}{}{}{}'.format(mb,mb,mb,mb,mb): 63,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,ma): 64,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,mb): 65,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,ma,ma): 66,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,mb,mb): 67,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,ma): 68,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,mb): 69,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,ma): 70,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,mb): 71,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,ma,ma): 72,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,mb,mb): 73,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,ma): 74,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,mb): 75, 
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,ma): 76,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,mb): 77,         
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,ma,ma): 78,          
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,mb,mb): 79,         
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,ma): 80,          
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,mb): 81,         
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,ma): 82,         
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,mb): 83,         
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,ma,ma): 84,         
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,mb,mb): 85,         
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,ma): 86,         
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,mb): 87} 

        elif self.surface == 'bcc110':
            return {'ontop|terrace|{}'.format(ma): 1, 
                    'ontop|terrace|{}'.format(mb): 2,
                    'short-bridge|terrace|{}{}'.format(ma,ma): 3,
                    'short-bridge|terrace|{}{}'.format(ma,mb): 4,
                    'short-bridge|terrace|{}{}'.format(mb,mb): 5,
                    'long-bridge|terrace|{}{}'.format(ma,ma): 6, 
                    'long-bridge|terrace|{}{}'.format(ma,mb): 7,
                    'long-bridge|terrace|{}{}'.format(mb,mb): 8, 
                    '3fold|terrace|{}{}{}'.format(ma,ma,ma): 9,
                    '3fold|terrace|{}{}{}'.format(ma,ma,mb): 10, 
                    '3fold|terrace|{}{}{}'.format(ma,mb,mb): 11,
                    '3fold|terrace|{}{}{}'.format(mb,mb,mb): 12}

        elif self.surface == 'bcc111':                         
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'ontop|terrace|{}'.format(ma): 3,
                    'ontop|terrace|{}'.format(mb): 4,
                    'ontop|corner|{}'.format(mb): 5,
                    'ontop|corner|{}'.format(mb): 6,
                    'short-bridge|sc-tc-o|{}{}'.format(ma,ma): 7, 
                    'short-bridge|sc-tc-o|{}{}'.format(ma,mb): 8,
                    'short-bridge|sc-tc-o|{}{}'.format(mb,mb): 9,
                    'short-bridge|tc-cc-o|{}{}'.format(ma,ma): 10,
                    'short-bridge|tc-cc-o|{}{}'.format(ma,mb): 11,
                    'short-bridge|tc-cc-o|{}{}'.format(mb,mb): 12,
                    'long-bridge|sc-cc-o|{}{}'.format(ma,ma): 13,
                    'long-bridge|sc-cc-o|{}{}'.format(ma,mb): 14,
                    'long-bridge|sc-cc-o|{}{}'.format(mb,mb): 15,
                    '3fold|sc-tc-cc-o|{}{}{}'.format(ma,ma,ma): 16,
                    '3fold|sc-tc-cc-o|{}{}{}'.format(ma,ma,mb): 17, 
                    '3fold|sc-tc-cc-o|{}{}{}'.format(ma,mb,mb): 18,
                    '3fold|sc-tc-cc-o|{}{}{}'.format(mb,mb,mb): 19}

        elif self.surface == 'bcc210':
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'ontop|terrace|{}'.format(ma): 3,
                    'ontop|terrace|{}'.format(mb): 4,
                    'ontop|corner|{}'.format(mb): 5,
                    'ontop|corner|{}'.format(mb): 6,
                    'bridge|step|{}{}'.format(ma,ma): 7, 
                    'bridge|step|{}{}'.format(ma,mb): 8,
                    'bridge|step|{}{}'.format(mb,mb): 9,
                    'bridge|terrace|{}{}'.format(ma,ma): 10,
                    'bridge|terrace|{}{}'.format(ma,mb): 11,
                    'bridge|terrace|{}{}'.format(mb,mb): 12,
                    'bridge|corner|{}{}'.format(ma,ma): 13,
                    'bridge|corner|{}{}'.format(ma,mb): 14,
                    'bridge|corner|{}{}'.format(mb,mb): 15,
                    'bridge|sc-tc-o|{}{}'.format(ma,ma): 16,
                    'bridge|sc-tc-o|{}{}'.format(ma,mb): 17,
                    'bridge|sc-tc-o|{}{}'.format(mb,mb): 18,
                    'bridge|tc-cc-o|{}{}'.format(ma,ma): 19,
                    'bridge|tc-cc-o|{}{}'.format(ma,mb): 20,
                    'bridge|tc-cc-o|{}{}'.format(mb,mb): 21,
                    'bridge|sc-cc-t|{}{}'.format(ma,ma): 22,
                    'bridge|sc-cc-t|{}{}'.format(ma,mb): 23,
                    'bridge|sc-cc-t|{}{}'.format(mb,mb): 24,
                    '3fold|sc-tc-o|{}{}{}'.format(ma,ma,ma): 25,
                    '3fold|sc-tc-o|{}{}{}'.format(ma,ma,mb): 26, 
                    '3fold|sc-tc-o|{}{}{}'.format(ma,mb,mb): 27,
                    '3fold|sc-tc-o|{}{}{}'.format(mb,mb,mb): 28,
                    '3fold|tc-cc-o|{}{}{}'.format(ma,ma,ma): 29,
                    '3fold|tc-cc-o|{}{}{}'.format(ma,ma,mb): 30,
                    '3fold|tc-cc-o|{}{}{}'.format(ma,mb,mb): 31,
                    '3fold|tc-cc-o|{}{}{}'.format(mb,mb,mb): 32,
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,ma,ma,ma): 33,
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,ma,ma,mb): 34, 
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,ma,mb,mb): 35,
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,mb,ma,mb): 36, 
                    '4fold|sc-cc-t|{}{}{}{}'.format(ma,mb,mb,mb): 37,
                    '4fold|sc-cc-t|{}{}{}{}'.format(mb,mb,mb,mb): 38}

        elif self.surface == 'bcc211':
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'bridge|step|{}{}'.format(ma,ma): 3,
                    'bridge|step|{}{}'.format(ma,mb): 4,
                    'bridge|step|{}{}'.format(mb,mb): 5,
                    'bridge|sc-tc-o|{}{}'.format(ma,ma): 6,
                    'bridge|sc-tc-o|{}{}'.format(ma,mb): 7,
                    'bridge|sc-tc-o|{}{}'.format(mb,mb): 8,
                    '3fold|sc-tc-o|{}{}{}'.format(ma,ma,ma): 9,
                    '3fold|sc-tc-o|{}{}{}'.format(ma,ma,mb): 10, 
                    '3fold|sc-tc-o|{}{}{}'.format(ma,mb,mb): 11,
                    '3fold|sc-tc-o|{}{}{}'.format(mb,mb,mb): 12,
                    '4fold|terrace|{}{}-{}{}'.format(ma,ma,ma,ma): 13,
                    '4fold|terrace|{}{}-{}{}'.format(ma,ma,ma,mb): 14,
                    '4fold|terrace|{}{}-{}{}'.format(ma,ma,mb,mb): 15,
                    '4fold|terrace|{}{}-{}{}'.format(ma,mb,ma,ma): 16,
                    '4fold|terrace|{}{}-{}{}'.format(ma,mb,ma,mb): 17,
                    '4fold|terrace|{}{}-{}{}'.format(ma,mb,mb,mb): 18,
                    '4fold|terrace|{}{}-{}{}'.format(mb,mb,ma,ma): 19,
                    '4fold|terrace|{}{}-{}{}'.format(mb,mb,ma,mb): 20,
                    '4fold|terrace|{}{}-{}{}'.format(mb,mb,mb,mb): 21,
                    # neighbor elements count clockwise from shorter bond ma
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,ma,ma,ma,ma): 22,
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,ma,ma,ma,mb): 23,
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,ma,ma,mb,mb): 24,
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,ma,mb,ma,mb): 25,
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,ma,mb,mb,ma): 26,
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,ma,mb,mb,mb): 27,
                    '5fold|terrace|{}-{}{}{}{}'.format(ma,mb,mb,mb,mb): 28,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,ma,ma,ma,ma): 29,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,ma,ma,ma,mb): 30,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,ma,ma,mb,mb): 31,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,ma,mb,ma,mb): 32,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,ma,mb,mb,ma): 33,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,ma,mb,mb,mb): 34,
                    '5fold|terrace|{}-{}{}{}{}'.format(mb,mb,mb,mb,mb): 35}

        elif self.surface == 'bcc310':
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'ontop|terrace|{}'.format(ma): 3,
                    'ontop|terrace|{}'.format(mb): 4,
                    'bridge|step|{}{}'.format(ma,ma): 5,
                    'bridge|step|{}{}'.format(ma,mb): 6,
                    'bridge|step|{}{}'.format(mb,mb): 7,
                    'bridge|terrace|{}{}'.format(ma,ma): 8,
                    'bridge|terrace|{}{}'.format(ma,mb): 9,
                    'bridge|terrace|{}{}'.format(mb,mb): 10,
                    'bridge|sc-tc-o|{}{}'.format(ma,ma): 11,
                    'bridge|sc-tc-o|{}{}'.format(ma,mb): 12,
                    'bridge|sc-tc-o|{}{}'.format(mb,mb): 13,
                    'bridge|sc-tc-t|{}{}'.format(ma,ma): 14,
                    'bridge|sc-tc-t|{}{}'.format(ma,mb): 15,
                    'bridge|sc-tc-t|{}{}'.format(mb,mb): 16,
                    '3fold|sc-tc-o|{}{}{}'.format(ma,ma,ma): 17,
                    '3fold|sc-tc-o|{}{}{}'.format(ma,ma,mb): 18,
                    '3fold|sc-tc-o|{}{}{}'.format(ma,mb,mb): 19,
                    '3fold|sc-tc-o|{}{}{}'.format(mb,mb,mb): 20,                    
                    '4fold|sc-tc-t|{}{}{}{}'.format(ma,ma,ma,ma): 21,
                    '4fold|sc-tc-t|{}{}{}{}'.format(ma,ma,ma,mb): 22, 
                    '4fold|sc-tc-t|{}{}{}{}'.format(ma,ma,mb,mb): 23,
                    '4fold|sc-tc-t|{}{}{}{}'.format(ma,mb,ma,mb): 24, 
                    '4fold|sc-tc-t|{}{}{}{}'.format(ma,mb,mb,mb): 25,
                    '4fold|sc-tc-t|{}{}{}{}'.format(mb,mb,mb,mb): 26}

        elif self.surface == 'hcp10m10-t':
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'ontop|terrace|{}'.format(ma): 3,
                    'ontop|terrace|{}'.format(mb): 4,                     
                    'bridge|step|{}{}'.format(ma,ma): 5,    
                    'bridge|step|{}{}'.format(ma,mb): 6,
                    'bridge|step|{}{}'.format(mb,mb): 7,
                    'bridge|terrace|{}{}'.format(ma,ma): 8,
                    'bridge|terrace|{}{}'.format(ma,mb): 9,
                    'bridge|terrace|{}{}'.format(mb,mb): 10,
                    'bridge|sc-tc-t|{}{}'.format(ma,ma): 11,
                    'bridge|sc-tc-t|{}{}'.format(ma,mb): 12,
                    'bridge|sc-tc-t|{}{}'.format(mb,mb): 13,
                    # neighbor elements count clockwise from shorter bond ma
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,ma,ma,ma,ma): 14,
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,ma,ma,ma,mb): 15,
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,ma,ma,mb,mb): 16,
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,ma,mb,ma,mb): 17,
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,ma,mb,mb,mb): 18, 
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,ma,mb,mb,mb): 19,
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,mb,mb,mb,mb): 20,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,ma,ma,ma,ma): 21,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,ma,ma,ma,mb): 22,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,ma,ma,mb,mb): 23,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,ma,mb,ma,mb): 24,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,ma,mb,mb,ma): 25,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,ma,mb,mb,mb): 26,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,mb,mb,mb,mb): 27}
 
        elif self.surface == 'hcp10m11':
            return {'ontop|step|{}'.format(ma): 1, 
                    'ontop|step|{}'.format(mb): 2,
                    'ontop|terrace|{}'.format(ma): 3, 
                    'ontop|terrace|{}'.format(mb): 4,                    
                    'bridge|step|{}{}'.format(ma,ma): 5, 
                    'bridge|step|{}{}'.format(ma,mb): 6,
                    'bridge|step|{}{}'.format(mb,mb): 7, 
                    'bridge|terrace|{}{}'.format(ma,ma): 8, 
                    'bridge|terrace|{}{}'.format(ma,mb): 9,
                    'bridge|terrace|{}{}'.format(mb,mb): 10, 
                    'bridge|sc-tc-h|{}{}'.format(ma,ma): 11, 
                    'bridge|sc-tc-h|{}{}'.format(ma,mb): 12,
                    'bridge|sc-tc-h|{}{}'.format(mb,mb): 13, 
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,ma): 14,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,mb): 15, 
                    'fcc|sc-tc-h|{}{}{}'.format(ma,mb,mb): 16,
                    'fcc|sc-tc-h|{}{}{}'.format(mb,mb,mb): 17,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,ma): 18,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,mb): 19, 
                    'hcp|sc-tc-h|{}{}{}'.format(ma,mb,mb): 20,
                    'hcp|sc-tc-h|{}{}{}'.format(mb,mb,mb): 21,
                    '4fold|subsurf|{}{}-{}{}'.format(ma,ma,ma,ma): 22,
                    '4fold|subsurf|{}{}-{}{}'.format(ma,ma,ma,mb): 23,
                    '4fold|subsurf|{}{}-{}{}'.format(ma,ma,mb,mb): 24,
                    '4fold|subsurf|{}{}-{}{}'.format(ma,mb,ma,ma): 25,
                    '4fold|subsurf|{}{}-{}{}'.format(ma,mb,ma,mb): 26,
                    '4fold|subsurf|{}{}-{}{}'.format(ma,mb,mb,mb): 27,
                    '4fold|subsurf|{}{}-{}{}'.format(mb,mb,ma,ma): 28,
                    '4fold|subsurf|{}{}-{}{}'.format(mb,mb,ma,mb): 29,
                    '4fold|subsurf|{}{}-{}{}'.format(mb,mb,mb,mb): 30,
                    # neighbor elements count clockwise from shorter bond ma
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,ma,ma,ma,ma): 31,
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,ma,ma,ma,mb): 32,
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,ma,ma,mb,mb): 33,
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,ma,mb,ma,mb): 34,
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,ma,mb,mb,mb): 35, 
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,ma,mb,mb,mb): 36,
                    '5fold|subsurf|{}-{}{}{}{}'.format(ma,mb,mb,mb,mb): 37,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,ma,ma,ma,ma): 38,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,ma,ma,ma,mb): 39,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,ma,ma,mb,mb): 40,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,ma,mb,ma,mb): 41,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,ma,mb,mb,ma): 42,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,ma,mb,mb,mb): 43,
                    '5fold|subsurf|{}-{}{}{}{}'.format(mb,mb,mb,mb,mb): 44,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,ma): 45,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,mb): 46,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,ma,ma): 47,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,mb,mb): 48,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,ma): 49,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,mb): 50,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,ma): 51,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,mb): 52,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,ma,ma): 53,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,mb,mb): 54,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,ma): 55,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,mb): 56,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,ma): 57,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,mb): 58,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,ma,ma): 59,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,mb,mb): 60,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,ma): 61,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,mb): 62,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,ma): 63,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,mb): 64,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,ma,ma): 65,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,mb,mb): 66,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,ma): 67,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,mb): 68}

        elif self.surface == 'hcp10m12':
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'ontop|terrace|{}'.format(ma): 3,
                    'ontop|terrace|{}'.format(mb): 4,
                    'ontop|corner|{}'.format(mb): 5,
                    'ontop|corner|{}'.format(mb): 6,
                    'bridge|step|{}{}'.format(ma,ma): 7, 
                    'bridge|step|{}{}'.format(ma,mb): 8,
                    'bridge|step|{}{}'.format(mb,mb): 9,
                    'bridge|terrace|{}{}'.format(ma,ma): 10,
                    'bridge|terrace|{}{}'.format(ma,mb): 11,
                    'bridge|terrace|{}{}'.format(mb,mb): 12,
                    'bridge|corner|{}{}'.format(ma,ma): 13,
                    'bridge|corner|{}{}'.format(ma,mb): 14,
                    'bridge|corner|{}{}'.format(mb,mb): 15,
                    'bridge|sc-tc-h|{}{}'.format(ma,ma): 16,
                    'bridge|sc-tc-h|{}{}'.format(ma,mb): 17,
                    'bridge|sc-tc-h|{}{}'.format(mb,mb): 18,
                    'bridge|tc-cc-t|{}{}'.format(ma,ma): 19,
                    'bridge|tc-cc-t|{}{}'.format(ma,mb): 20,
                    'bridge|tc-cc-t|{}{}'.format(mb,mb): 21,
                    'bridge|sc-cc-h|{}{}'.format(ma,ma): 22,
                    'bridge|sc-cc-h|{}{}'.format(ma,mb): 23,
                    'bridge|sc-cc-h|{}{}'.format(mb,mb): 24,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,ma): 25,
                    'fcc|sc-tc-h|{}{}{}'.format(ma,ma,mb): 26, 
                    'fcc|sc-tc-h|{}{}{}'.format(ma,mb,mb): 27,
                    'fcc|sc-tc-h|{}{}{}'.format(mb,mb,mb): 28,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,ma): 29,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,ma,mb): 30,
                    'hcp|sc-tc-h|{}{}{}'.format(ma,mb,mb): 31,
                    'hcp|sc-tc-h|{}{}{}'.format(mb,mb,mb): 32,
                    'fcc|sc-cc-h|{}{}{}'.format(ma,ma,ma): 33,
                    'fcc|sc-cc-h|{}{}{}'.format(ma,ma,mb): 34,
                    'fcc|sc-cc-h|{}{}{}'.format(ma,mb,mb): 35,
                    'fcc|sc-cc-h|{}{}{}'.format(mb,mb,mb): 36,
                    'hcp|sc-cc-h|{}{}{}'.format(ma,ma,ma): 37,
                    'hcp|sc-cc-h|{}{}{}'.format(ma,ma,mb): 38,
                    'hcp|sc-cc-h|{}{}{}'.format(ma,mb,mb): 39,
                    'hcp|sc-cc-h|{}{}{}'.format(mb,mb,mb): 40,
                    '4fold|tc-cc-t|{}{}{}{}'.format(ma,ma,ma,ma): 41,
                    '4fold|tc-cc-t|{}{}{}{}'.format(ma,ma,ma,mb): 42, 
                    '4fold|tc-cc-t|{}{}{}{}'.format(ma,ma,mb,mb): 43,
                    '4fold|tc-cc-t|{}{}{}{}'.format(ma,mb,ma,mb): 44, 
                    '4fold|tc-cc-t|{}{}{}{}'.format(ma,mb,mb,mb): 45,
                    '4fold|tc-cc-t|{}{}{}{}'.format(mb,mb,mb,mb): 46,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,ma): 47,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,ma,mb): 48,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,ma,ma): 49,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,ma,mb,mb): 50,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,ma): 51,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,ma,mb,mb,mb): 52,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,ma): 53,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,ma,mb): 54,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,ma,ma): 55,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,ma,mb,mb): 56,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,ma): 57,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,ma,mb,mb,mb,mb): 58,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,ma): 59,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,ma,mb): 60,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,ma,ma): 61,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,ma,mb,mb): 62,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,ma): 63,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(ma,mb,mb,mb,mb,mb): 64,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,ma): 65,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,ma,mb): 66,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,ma,ma): 67,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,ma,mb,mb): 68,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,ma): 69,
                    '6fold|subsurf|{}{}{}{}{}{}'.format(mb,mb,mb,mb,mb,mb): 70}


def add_adsorbate_to_site(atoms, adsorbate, site, height=None, 
                          orientation=None):            
    '''orientation: vector that the adsorbate is algined to'''
    
    if height is None:
        height = heights_dict[site['site']]

    # Make the correct position
    normal = site['normal']
    pos = site['position'] + normal * height

    # Convert the adsorbate to an Atoms object
    if isinstance(adsorbate, Atoms):
        ads = adsorbate
    elif isinstance(adsorbate, Atom):
        ads = Atoms([adsorbate])

    # Or assume it is a string representing a molecule
    else:
        ads = adsorbate_molecule(adsorbate) 
        if len(ads) == 2 or adsorbate == 'COH':
            ads.rotate(ads[1].position - ads[0].position, normal)
            #pvec = np.cross(np.random.rand(3) - ads[0].position, normal)
            #ads.rotate(-45, pvec, center=ads[0].position)

    if adsorbate not in adsorbate_list:
        # Always sort the indices the same order as the input symbol.
        # This is a naive sorting which might cause H in wrong order.
        # Please sort your own adsorbate atoms by reindexing as has
        # been done in adsorbate_molecule function.
        symout = list(Formula(adsorbate))
        symin = list(ads.symbols)
        newids = []
        for elt in symout:
            idx = symin.index(elt)
            newids.append(idx)
            symin[idx] = None
        ads = ads[newids]

    bondpos = ads[0].position
    if orientation is not None:
        oripos = next((a.position for a in ads[1:] if 
                       a.symbol != 'H'), ads[1].position)
        v1 = get_rejection_between(oripos - bondpos, normal)
        v2 = get_rejection_between(orientation, normal)
        radian = get_angle_between(v1, v2)

        # Flip the sign of the angle if the result is not the closest
        rm_p = get_rotation_matrix(axis=normal, angle=radian)
        rm_n = get_rotation_matrix(axis=normal, angle=-radian)        
        npos_p, npos_n = rm_p @ oripos, rm_n @ oripos
        nbpos_p = npos_p + pos - bondpos
        nbpos_n = npos_n + pos - bondpos
        d_p = np.linalg.norm(nbpos_p - pos - orientation)
        d_n = np.linalg.norm(nbpos_n - pos - orientation)
        if d_p <= d_n:
            for a in ads:
                a.position = rm_p @ a.position
        else:
            for a in ads:
                a.position = rm_n @ a.position

    ads.translate(pos - bondpos)
    atoms += ads


def remove_adosorbate_from_site(atoms, site, remove_fragment=False):

    if not remove_fragment:
        si = list(site['adsorbate_indices'])
    else:
        si = list(site['fragment_indices'])
    del atoms[si]


def add_adsorbate(atoms, adsorbate, site, surface=None, geometry=None, 
                  indices=None, height=None, composition=None, 
                  subsurf_element=None, site_list=None):
    """
    A function for adding adsorbate to a specific adsorption site on a 
    monometalic nanoparticle in icosahedron / cuboctahedron / decahedron / 
    truncated-octahedron shapes, or a 100/111 surface slab.

    Parameters
    ----------
    atoms: The nanoparticle or surface slab onto which the adsorbate should be added.
        
    adsorbate: The adsorbate. Must be one of the following three types:
        A string containing the chemical symbol for a single atom.
        An atom object.
        An atoms object (for a molecular adsorbate).

    site: Support 5 typical adsorption sites: 
        1-fold site 'ontop', 
        2-fold site 'bridge', 
        3-fold hollow sites 'fcc' and 'hcp', 
        4-fold hollow site '4fold'.

    surface: Support 4 typical surfaces (positions) for fcc crystal where the 
    adsorbate is attached: 
        'vertex', 
        'edge', 
        'fcc100', 
        'fcc111'.

    height: The height from the adsorbate to the surface.
        Default is {'ontop': 2.0, 'bridge': 1.8, 'fcc': 1.8, 'hcp': 1.8, 
        '4fold': 1.7} for nanoparticles and 2.0 for all sites on surface slabs.

    Example
    ------- 
    add_adsorbate(atoms,adsorbate='CO',site='4fold',surface='fcc100')
    """

    if height is None:
        height = heights_dict[site]
    composition_effect = False if composition is None else True
    subsurf_effect = False if subsurf_element is None else True

    if composition:
        if '-' in composition or site == '6fold':
            scomp = composition
        else:
            comp = re.findall('[A-Z][^A-Z]*', composition)
            if len(comp) != 4:
                scomp = ''.join(sorted(comp, key=lambda x: 
                                       Atom(x).number))
            else:
                if comp[0] != comp[2]:
                    scomp = ''.join(sorted(comp, key=lambda x: 
                                           Atom(x).number))
                else:
                    if Atom(comp[0]).number > Atom(comp[1]).number:
                        scomp = comp[1]+comp[0]+comp[3]+comp[2]
                    else:
                        scomp = ''.join(comp)
    else:
        scomp = None

    if site_list:
        all_sites = site_list.copy()
    else:
        all_sites = enumerate_adsorption_sites(atoms, surface, 
                                               geometry, True, 
                                               composition_effect, 
                                               subsurf_effect)

    if indices:
        if not isinstance(indices, Iterable):
            indices = [indices]
        indices = tuple(sorted(indices))
        st = next((s for s in all_sites if 
                   s['indices'] == indices), None)
    else:
        st = next((s for s in all_sites if 
                   s['site'] == site and
                   s['composition'] == scomp and 
                   s['subsurf_element'] 
                   == subsurf_element), None)

    if not st:
        print('No such site can be found')            
    else:
        add_adsorbate_to_site(atoms, adsorbate, st, height)


def group_sites_by_surface(atoms, sites, site_list=None):                            
    """A function that uses networkx to group one type of sites 
    by geometrical facets of the nanoparticle"""

    # Find all indices of vertex and edge sites
    if not site_list:
        nas = NanoparticleAdsorptionSites(atoms)
        site_list = nas.site_list
    ve_indices = [s['indices'] for s in site_list if 
                  s['site'] == 'ontop' and 
                  s['surface'] in ['vertex', 'edge']]
    unique_ve_indices = set(list(sum(ve_indices, ())))
     
    G=nx.Graph()
    for site in sites:
        indices = site['indices']
        reduced_indices = tuple(i for i in indices if i 
                                not in unique_ve_indices)
        site['reduced_indices'] = reduced_indices
        nx.add_path(G, reduced_indices)
    components = list(nx.connected_components(G))
    groups = []
    for component in components:
        group = []
        for path in [s['reduced_indices'] for s in sites]:
            if component.issuperset(path):
                group.append(path)
        groups.append(group)
    grouped_sites = defaultdict(list)
    for site in sites:
        for group in groups:
            if site['reduced_indices'] in group:
                grouped_sites[groups.index(group)] += [site]

    return grouped_sites


def symmetric_pattern_generator(atoms, adsorbate, surface=None, 
                                coverage=1., height=None, 
                                min_adsorbate_distance=0.):
    """A function for generating certain well-defined symmetric adsorbate 
       coverage patterns.

       Parameters
       ----------
       atoms: The nanoparticle or surface slab onto which the adsorbate 
              should be added.
           
       adsorbate: The adsorbate. Must be one of the following three types:
           A string containing the chemical symbol for a single atom.
           An atom object.
           An atoms object (for a molecular adsorbate).                                                                                                         
       surface: Support 2 typical surfaces for fcc crystal where the 
           adsorbate is attached:  
           'fcc100', 
           'fcc111'.
           Can either specify a string or a list of strings

       coverage: The coverage (ML) of the adsorbate.
           Note that for small nanoparticles, the function might give results 
           that do not correspond to the coverage. This is normal since the 
           surface area can be too small to encompass the coverage pattern 
           properly. We expect this function to work especially well on large 
           nanoparticles and low-index extended surfaces.                                                                                              

       height: The height from the adsorbate to the surface.
           Default is {'ontop': 2.0, 'bridge': 1.8, 'fcc': 1.8, 'hcp': 1.8, 
           '4fold': 1.7} for nanoparticles and 2.0 for all sites on surface 
           slabs.

       min_adsorbate_distance: The minimum distance between two adsorbate 
           atoms. Default value 0.2 is good for adsorbate coverage patterns. 
           Play around to find the best value.
       
       Example
       ------- 
       pattern_generator(atoms, adsorbate='CO', surface='fcc111', coverage=3/4)
    """

    ads_indices = [a.index for a in atoms if 
                   a.symbol in adsorbate_elements]
    ads_atoms = None
    if ads_indices:
        ads_atoms = atoms[ads_indices]
        atoms = atoms[[a.index for a in atoms if 
                       a.symbol not in adsorbate_elements]]
    ads = adsorbate_molecule(adsorbate)

    if True not in atoms.pbc:                            
        if surface is None:
            surface = ['fcc100', 'fcc111']        
        sas = NanoparticleAdsorptionSites(atoms)
        site_list = sas.site_list
    else:
        sas = SlabAdsorptionSites(atoms, surface=surface)
        if surface is None:
            surface = sas.surface
        site_list = sas.site_list
    if not isinstance(surface, list):
        surface = [surface] 

    final_sites = []
    if 'fcc111' in surface: 
        if coverage == 1:
            fcc_sites = [s for s in site_list 
                         if s['site'] == 'fcc']
            if fcc_sites:
                final_sites += fcc_sites

        elif coverage == 3/4:
            # Kagome pattern
            fcc_sites = [s for s in site_list  
                         if s['site'] == 'fcc']
            if True not in atoms.pbc:                                
                grouped_sites = group_sites_by_surface(
                                atoms, fcc_sites, site_list)
            else:
                grouped_sites = {'pbc_sites': fcc_sites}

            for sites in grouped_sites.values():
                if sites:
                    sites_to_delete = [sites[0]]
                    for sitei in sites_to_delete:
                        common_site_indices = []
                        non_common_sites = []
                        for sitej in sites:
                            if sitej['indices'] == sitei['indices']:
                                pass
                            elif set(sitej['indices']) & set(sitei['indices']):
                                common_site_indices += list(sitej['indices'])
                            else:
                                non_common_sites.append(sitej)
                        for sitej in non_common_sites:
                            overlap = sum([common_site_indices.count(i) 
                                          for i in sitej['indices']])
                            if overlap == 1 and sitej['indices'] \
                            not in [s['indices'] for s in sites_to_delete]:
                                sites_to_delete.append(sitej)                
                    for s in sites:
                        if s['indices'] not in [st['indices'] 
                        for st in sites_to_delete]:
                            final_sites.append(s)

        elif coverage == 2/4:
            # Honeycomb pattern
            fcc_sites = [s for s in site_list if s['site'] == 'fcc']
            hcp_sites = [s for s in site_list if s['site'] == 'hcp']
            all_sites = fcc_sites + hcp_sites
            if True not in atoms.pbc:    
                grouped_sites = group_sites_by_surface(
                                atoms, all_sites, site_list)
            else:
                grouped_sites = {'pbc_sites': all_sites}
            for sites in grouped_sites.values():
                if sites:                    
                    sites_to_remain = [sites[0]]
                    for sitei in sites_to_remain:
                        for sitej in sites:
                            if sitej['indices'] == sitei['indices']:
                                pass
                            elif len(set(sitej['indices']) & \
                            set(sitei['indices'])) == 1 \
                            and sitej['site'] != sitei['site'] \
                            and sitej['indices'] not in [s['indices'] 
                            for s in sites_to_remain]:
                                sites_to_remain.append(sitej)
                    final_sites += sites_to_remain                                         

            if True not in atoms.pbc:                                                                       
                bad_sites = []
                for sti in final_sites:
                    if sti['site'] == 'hcp':
                        count = 0
                        for stj in final_sites:
                            if stj['site'] == 'fcc':
                                if len(set(stj['indices']) & \
                                set(sti['indices'])) == 2:
                                    count += 1
                        if count != 0:
                            bad_sites.append(sti)
                final_sites = [s for s in final_sites if s['indices'] \
                               not in [st['indices'] for st in bad_sites]]

        elif coverage == 1/4:
            # Kagome pattern
            fcc_sites = [s for s in site_list 
                         if s['site'] == 'fcc']                                                                 
            if True not in atoms.pbc:                                
                grouped_sites = group_sites_by_surface(
                                atoms, fcc_sites, site_list)
            else:
                grouped_sites = {'pbc_sites': fcc_sites}

            for sites in grouped_sites.values():
                if sites:
                    sites_to_remain = [sites[0]]
                    for sitei in sites_to_remain:
                        common_site_indices = []
                        non_common_sites = []
                        for sitej in sites:
                            if sitej['indices'] == sitei['indices']:
                                pass
                            elif set(sitej['indices']) & set(sitei['indices']):
                                common_site_indices += list(sitej['indices'])
                            else:
                                non_common_sites.append(sitej)
                        for sitej in non_common_sites:
                            overlap = sum([common_site_indices.count(i) 
                                          for i in sitej['indices']])
                            if overlap == 1 and sitej['indices'] \
                            not in [s['indices'] for s in sites_to_remain]:
                                sites_to_remain.append(sitej)               
                    final_sites += sites_to_remain

    if 'fcc100' in surface:
        if coverage == 1:
            fold4_sites = [s for s in site_list if s['site'] == '4fold']
            if fold4_sites:
                final_sites += fold4_sites

        elif coverage == 3/4:
            fold4_sites = [s for s in site_list if s['site'] == '4fold']
            if True not in atoms.pbc:                                           
                grouped_sites = group_sites_by_surface(
                                atoms, fold4_sites, site_list)
            else:
                grouped_sites = {'pbc_sites': fold4_sites}
            for sites in grouped_sites.values():
                if sites:
                    sites_to_delete = [sites[0]]
                    for sitei in sites_to_delete:
                        common_site_indices = []
                        non_common_sites = []
                        for sitej in sites:
                            if sitej['indices'] == sitei['indices']:
                                pass
                            elif set(sitej['indices']) & set(sitei['indices']):
                                common_site_indices += list(sitej['indices'])
                            else:
                                non_common_sites.append(sitej)                        
                        for sitej in non_common_sites:                        
                            overlap = sum([common_site_indices.count(i) 
                                          for i in sitej['indices']])                        
                            if overlap in [1, 4] and sitej['indices'] not in \
                            [s['indices'] for s in sites_to_delete]:  
                                sites_to_delete.append(sitej)
                    for s in sites:
                        if s['indices'] not in [st['indices'] 
                                   for st in sites_to_delete]:
                            final_sites.append(s)

        elif coverage == 2/4:
            #c(2x2) pattern
            fold4_sites = [s for s in site_list if s['site'] == '4fold']
            original_sites = copy.deepcopy(fold4_sites)
            if True not in atoms.pbc:
                grouped_sites = group_sites_by_surface(
                                atoms, fold4_sites, site_list)
            else:
                grouped_sites = {'pbc_sites': fold4_sites}
            for sites in grouped_sites.values():
                if sites:
                    sites_to_remain = [sites[0]]
                    for sitei in sites_to_remain:
                        for sitej in sites:
                            if (len(set(sitej['indices']) & \
                            set(sitei['indices'])) == 1) and \
                            (sitej['indices'] not in [s['indices'] 
                            for s in sites_to_remain]):
                                sites_to_remain.append(sitej)
                    for s in original_sites:
                        if s['indices'] in [st['indices'] 
                        for st in sites_to_remain]:
                            final_sites.append(s)

        elif coverage == 1/4:
            #p(2x2) pattern
            fold4_sites = [s for s in site_list if s['site'] == '4fold']
            if True not in atoms.pbc:                                           
                grouped_sites = group_sites_by_surface(
                                atoms, fold4_sites, site_list)
            else:
                grouped_sites = {'pbc_sites': fold4_sites}
            for sites in grouped_sites.values():
                if sites:
                    sites_to_remain = [sites[0]]
                    for sitei in sites_to_remain:
                        common_site_indices = []
                        non_common_sites = []
                        for idx, sitej in enumerate(sites):
                            if sitej['indices'] == sitei['indices']:
                                pass
                            elif set(sitej['indices']) & set(sitei['indices']):
                                common_site_indices += list(sitej['indices'])
                            else:
                                non_common_sites.append(sitej)
                        for sitej in non_common_sites:
                            overlap = sum([common_site_indices.count(i) 
                                          for i in sitej['indices']])
                            if overlap in [1, 4] and sitej['indices'] not in \
                            [s['indices'] for s in sites_to_remain]:  
                                sites_to_remain.append(sitej)
                    final_sites += sites_to_remain

    # Add edge coverage for nanoparticles
    if True not in atoms.pbc:
        if coverage == 1:
            edge_sites = [s for s in site_list if 
                          s['site'] == 'bridge' and 
                          s['surface'] == 'edge']
            vertex_indices = [s['indices'][0] for 
                              s in site_list if 
                              s['site'] == 'ontop' and 
                              s['surface'] == 'vertex']
            ve_common_indices = set()
            for esite in edge_sites:
                if set(esite['indices']) & set(vertex_indices):
                    for i in esite['indices']:
                        if i not in vertex_indices:
                            ve_common_indices.add(i)
            for esite in edge_sites:
                if not set(esite['indices']).issubset(
                ve_common_indices):
                    final_sites.append(esite)

        if coverage == 3/4:
            occupied_sites = final_sites.copy()
            hcp_sites = [s for s in site_list if 
                         s['site'] == 'hcp' and
                         s['surface'] == 'fcc111']
            edge_sites = [s for s in site_list if 
                          s['site'] == 'bridge' and
                          s['surface'] == 'edge']
            vertex_indices = [s['indices'][0] for 
                              s in site_list if
                              s['site'] == 'ontop' and 
                              s['surface'] == 'vertex']
            ve_common_indices = set()
            for esite in edge_sites:
                if set(esite['indices']) & set(vertex_indices):
                    for i in esite['indices']:
                        if i not in vertex_indices:
                            ve_common_indices.add(i)                
            for esite in edge_sites:
                if not set(esite['indices']).issubset(
                ve_common_indices):
                    intermediate_indices = []
                    for hsite in hcp_sites:
                        if len(set(esite['indices']) & \
                               set(hsite['indices'])) == 2:
                            intermediate_indices.append(min(
                            set(esite['indices']) ^ \
                            set(hsite['indices'])))
                    too_close = 0
                    for s in occupied_sites:
                        if len(set(esite['indices']) & \
                        set(s['indices'])) == 2:
                            too_close += 1
                    share = [0]
                    for interi in intermediate_indices:
                        share.append(len([s for s in occupied_sites if \
                                          interi in s['indices']]))
                    if max(share) <= 2 and too_close == 0:
                        final_sites.append(esite)

        if coverage == 2/4:            
            occupied_sites = final_sites.copy()
            edge_sites = [s for s in site_list if 
                          s['site'] == 'bridge' and
                          s['surface'] == 'edge']
            vertex_indices = [s['indices'][0] for 
                              s in site_list if
                              s['site'] == 'ontop' and 
                              s['surface'] == 'vertex']
            ve_common_indices = set()
            for esite in edge_sites:
                if set(esite['indices']) & set(vertex_indices):
                    for i in esite['indices']:
                        if i not in vertex_indices:
                            ve_common_indices.add(i)                
            for esite in edge_sites:
                if not set(esite['indices']).issubset(
                ve_common_indices):
                    intermediate_indices = []
                    for hsite in hcp_sites:
                        if len(set(esite['indices']) & \
                               set(hsite['indices'])) == 2:
                            intermediate_indices.append(min(
                            set(esite['indices']) ^ \
                            set(hsite['indices'])))
                    share = [0]
                    for interi in intermediate_indices:
                        share.append(len([s for s in occupied_sites if \
                                          interi in s['indices']]))
                    too_close = 0
                    for s in occupied_sites:
                        if len(set(esite['indices']) & \
                        set(s['indices'])) == 2:
                            too_close += 1
                    if max(share) <= 1 and too_close == 0:
                        final_sites.append(esite)

        if coverage == 1/4:
            occupied_sites = final_sites.copy()
            hcp_sites = [s for s in site_list if 
                         s['site'] == 'hcp' and
                         s['surface'] == 'fcc111']
            edge_sites = [s for s in site_list if 
                          s['site'] == 'bridge' and
                          s['surface'] == 'edge']
            vertex_indices = [s['indices'][0] for 
                              s in site_list if
                              s['site'] == 'ontop' and 
                              s['surface'] == 'vertex'] 
            ve_common_indices = set()
            for esite in edge_sites:
                if set(esite['indices']) & set(vertex_indices):
                    for i in esite['indices']:
                        if i not in vertex_indices:
                            ve_common_indices.add(i)                
            for esite in edge_sites:
                if not set(esite['indices']).issubset(
                ve_common_indices):
                    intermediate_indices = []
                    for hsite in hcp_sites:
                        if len(set(esite['indices']) & \
                        set(hsite['indices'])) == 2:
                            intermediate_indices.append(min(
                             set(esite['indices']) ^ \
                             set(hsite['indices'])))
                    share = [0]
                    for interi in intermediate_indices:
                        share.append(len([s for s in occupied_sites if \
                                          interi in s['indices']]))
                    too_close = 0
                    for s in occupied_sites:
                        if len(set(esite['indices']) & \
                        set(s['indices'])) > 0:
                            too_close += 1
                    if max(share) == 0 and too_close == 0:
                        final_sites.append(esite)

    for site in final_sites:
        add_adsorbate_to_site(atoms, adsorbate, site, height)

    if min_adsorbate_distance > 0.:
        if True not in atoms.pbc:
            sac = NanoparticleAdsorbateCoverage(atoms, sas)
        else:
            sac = SlabAdsorbateCoverage(atoms, sas)        
        remove_adsorbates_too_close(atoms, sac, min_adsorbate_distance)

    return atoms


def remove_adsorbates_too_close(atoms, adsorbate_coverage,
                                min_adsorbate_distance=0.5):

    dups = get_duplicate_atoms(atoms, cutoff=min_adsorbate_distance)
    if dups.size == 0:
        return

    del_ids = set(dups[:,0])
    rm_ids = []
    hsl = adsorbate_coverage.hetero_site_list
    for st in hsl:
        if st['occupied'] == 1:
            ads_ids = st['adsorbate_indices']
            if del_ids.intersection(set(ads_ids)):
                rm_ids += list(ads_ids)
    rm_ids = list(set(rm_ids))

    del atoms[rm_ids]


def full_coverage_pattern_generator(atoms, adsorbate, site, height=None, 
                                    min_adsorbate_distance=0.6):
    '''A function to generate different 1ML coverage patterns'''

    ads_indices = [a.index for a in atoms if a.symbol in adsorbate_elements]
    ads_atoms = None
    if ads_indices:
        ads_atoms = atoms[ads_indices]
        atoms = atoms[[a.index for a in atoms if a.symbol not in adsorbate_elements]]
    ads = molecule(adsorbate)[::-1]
    if str(ads.symbols) != 'CO':
        ads.set_chemical_symbols(ads.get_chemical_symbols()[::-1])
    final_sites = []
    positions = []
    if site == 'fcc':
        return symmetric_pattern_generator(atoms, adsorbate, coverage=1, height=height, 
                                           min_adsorbate_distance=min_adsorbate_distance)
    elif site == 'ontop':
        sites = get_monometallic_sites(atoms, site='ontop', surface='fcc100') +\
                get_monometallic_sites(atoms, site='ontop', surface='fcc111') +\
                get_monometallic_sites(atoms, site='ontop', surface='edge') +\
                get_monometallic_sites(atoms, site='ontop', surface='vertex')
        if sites:
            final_sites += sites
            positions += [s['adsorbate_position'] for s in sites]
    elif site in ['hcp', '4fold']:
        if True not in atoms.pbc:
            sites = get_monometallic_sites(atoms, site='hcp', surface='fcc111', height=height) +\
                    get_monometallic_sites(atoms, site='4fold', surface='fcc100', height=height)
        else:
            sites = get_monometallic_sites(atoms, site='hcp', surface='fcc111', height=height)
        if sites:
            final_sites += sites
            positions += [s['adsorbate_position'] for s in sites]

    if True not in atoms.pbc:
        if adsorbate == 'CO':
            for site in final_sites:
                add_adsorbate(atoms, molecule(adsorbate)[::-1], site)
            nl = FullNeighborList(rCut=min_adsorbate_distance, atoms=atoms)     
            nl.update(atoms)            
            atom_indices = [a.index for a in atoms if a.symbol == 'O']            
            n_ads_atoms = 2
            overlap_atoms_indices = []
            for idx in atom_indices:   
                neighbor_indices, _ = nl.get_neighbors(idx)
                overlap = 0
                for i in neighbor_indices:
                    if (atoms[i].symbol in adsorbate_elements) and (i not in overlap_atoms_indices):
                        overlap += 1
                if overlap > 0:
                    overlap_atoms_indices += list(set([idx-n_ads_atoms+1, idx]))
            del atoms[overlap_atoms_indices]

        else:
            for site in final_sites:
                add_adsorbate(atoms, molecule(adsorbate), site)
            nl = FullNeighborList(rCut=min_adsorbate_distance, atoms=atoms)   
            nl.update(atoms)            
            atom_indices = [a.index for a in atoms if a.symbol == adsorbate[-1]]
            ads_symbols = molecule(adsorbate).get_chemical_symbols()
            n_ads_atoms = len(ads_symbols)
            overlap_atoms_indices = []
            for idx in atom_indices:   
                neighbor_indices, _ = nl.get_neighbors(idx)
                overlap = 0
                for i in neighbor_indices:                                                                
                    if (atoms[i].symbol in adsorbate_elements) and (i not in overlap_atoms_indices):                       
                        overlap += 1                                                                      
                if overlap > 0:                                                                           
                    overlap_atoms_indices += list(set([idx-n_ads_atoms+1, idx]))                                
            del atoms[overlap_atoms_indices]                                                                    

    else:
        for pos in positions:
            ads.translate(pos - ads[0].position)
            atoms += ads
        if ads_indices:
            atoms += ads_atoms

    return atoms


def random_pattern_generator(atoms, adsorbate, surface=None, 
                             min_adsorbate_distance=2., 
                             heights=heights_dict):
    '''A function for generating random coverage patterns with constraint.
       Parameters
       ----------
       atoms: The nanoparticle or surface slab onto which the adsorbate should be added.
           
       adsorbate: The adsorbate. Must be one of the following three types:
           A string containing the chemical symbol for a single atom.
           An atom object.
           An atoms object (for a molecular adsorbate).                                                                                                       
       min_adsorbate_distance: The minimum distance constraint between any two adsorbates.

       heights: A dictionary that contains the adsorbate height for each site type.'''
 
    ads_indices = [a.index for a in atoms if a.symbol in adsorbate_elements]
    ads_atoms = None
    if ads_indices:
        ads_atoms = atoms[ads_indices]
        atoms = atoms[[a.index for a in atoms if a.symbol not in adsorbate_elements]]
    all_sites = enumerate_monometallic_sites(atoms, surface=surface, 
                                             heights=heights, subsurf_effect=False)
    random.shuffle(all_sites)    
 
    if True not in atoms.pbc:
        for site in all_sites:
            add_adsorbate(atoms, molecule(adsorbate), site)
    else:
        ads = molecule(adsorbate)[::-1]
        if str(ads.symbols) != 'CO':
            ads.set_chemical_symbols(ads.get_chemical_symbols()[::-1])
        positions = [s['adsorbate_position'] for s in all_sites]
        for pos in positions:
            ads.translate(pos - ads[0].position)
            atoms += ads
        if ads_indices:
            atoms += ads_atoms

    nl = FullNeighborList(rCut=min_adsorbate_distance, atoms=atoms)   
    nl.update(atoms)            
    atom_indices = [a.index for a in atoms if a.symbol == adsorbate[-1]]
    random.shuffle(atom_indices)
    ads_symbols = molecule(adsorbate).get_chemical_symbols()
    n_ads_atoms = len(ads_symbols)
    overlap_atoms_indices = []
    
    for idx in atom_indices:   
        neighbor_indices, _ = nl.get_neighbors(idx)
        overlap = 0
        for i in neighbor_indices:                                                                
            if (atoms[i].symbol in adsorbate_elements) and (i not in overlap_atoms_indices):                     
                overlap += 1                                                                      
        if overlap > 0:                                                                           
            overlap_atoms_indices += list(set([idx-n_ads_atoms+1, idx]))                                
    del atoms[overlap_atoms_indices]

    return atoms
