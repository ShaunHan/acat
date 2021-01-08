from .settings import adsorbate_elements, adsorbate_formulas
from .adsorption_sites import * 
from .utilities import *
from ase.io import read, write
from ase.build import molecule
from ase.data import covalent_radii, atomic_numbers
from ase.geometry import find_mic, get_duplicate_atoms
from ase.formula import Formula
from ase.visualize import view
from ase import Atom, Atoms
from collections import defaultdict 
from collections import Counter
from operator import itemgetter
from copy import deepcopy
import networkx as nx
import numpy as np
import random
import copy
import re


class ClusterAdsorbateCoverage(object):
    """dmax: maximum bond length [Ã] that should be considered as an adsorbate"""       

    def __init__(self, atoms, 
                 adsorption_sites=None, 
                 dmax=2.5):

        self.atoms = atoms.copy()
        self.positions = atoms.positions
        self.symbols = atoms.symbols
        self.ads_ids = [a.index for a in atoms if 
                        a.symbol in adsorbate_elements]
        assert self.ads_ids 
        self.ads_atoms = atoms[self.ads_ids]
        self.cell = atoms.cell
        self.pbc = atoms.pbc
        self.dmax = dmax

        self.make_ads_neighbor_list()
        self.ads_connectivity_matrix = self.get_ads_connectivity() 
        self.identify_adsorbates()

        if adsorption_sites:
            cas = adsorption_sites
        else:
            cas = ClusterAdsorptionSites(atoms, allow_6fold=True,
                                         composition_effect=True,
                                         subsurf_effect=False)    
        self.cas = cas
        self.slab = cas.atoms
        self.allow_6fold = cas.allow_6fold
        self.composition_effect = cas.composition_effect
        if cas.subsurf_effect:
            raise NotImplementedError

        self.metals = cas.metals
        self.surf_ids = cas.surf_ids
        self.hetero_site_list = deepcopy(cas.site_list)
        self.unique_sites = cas.get_unique_sites(unique_composition=
                                                 self.composition_effect) 
        self.label_dict = self.get_bimetallic_label_dict() \
                          if self.composition_effect else \
                          self.get_monometallic_label_dict()

        self.label_list = ['0'] * len(self.hetero_site_list)
        self.label_occupied_sites()
        self.labels = self.get_labels()

    def identify_adsorbates(self):
        G = nx.Graph()
        adscm = self.ads_connectivity_matrix

        # Cut off all intermolecular H-H bonds except intramolecular               
        # H-H bonds in e.g. H2
        hids = [a.index for a in self.ads_atoms if a.symbol == 'H']
        for hi in hids:
            conns = np.where(adscm[hi] == 1)[0]
            hconns = [i for i in conns if self.ads_atoms.symbols[i] == 'H']
            if hconns and len(conns) > 1:
                adscm[hi,hconns] = 0
      
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

    def get_hetero_connectivity(self):
        """Generate a connection matrix of slab + adsorbates."""
        nbslist = neighbor_shell_list(self.atoms, 0.3, neighbor_number=1)
        return get_connectivity_matrix(nbslist)                          

    def get_ads_connectivity(self):
        """Generate a connection matrix for adsorbate atoms."""
        return get_connectivity_matrix(self.ads_nblist) 

    def get_site_connectivity(self):
        """Generate a connection matrix for adsorption sites."""
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
            adssym = next((k for k, v in adsorbate_formulas.items() 
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
                    fsym = next((f for f in adsorbate_fragments(adssym) 
                                 if f[0] == bondsym), None)
                    st['fragment'] = fsym
                    flen = len(list(Formula(fsym)))
                    adsids = st['adsorbate_indices']
                    ibond = adsids.index(bondid)
                    fsi = adsids[ibond:ibond+flen]
                    st['fragment_indices'] = fsi
                else:
                    st['fragment_indices'] = st['adsorbate_indices'] 
                signature = [st['site'], st['surface']]                     
                if self.composition_effect:
                    signature.append(st['composition'])
                stlab = self.label_dict['|'.join(signature)]
                label = str(stlab) + st['fragment']
                st['label'] = label
                ll[j] = label

    def make_ads_neighbor_list(self, dx=.2, neighbor_number=1):
        """Generate a periodic neighbor list (defaultdict).""" 
        self.ads_nblist = neighbor_shell_list(self.ads_atoms, dx, 
                                              neighbor_number, mic=False)

    def get_labels(self):
        ll = self.label_list
        labs = [lab for lab in ll if lab != '0']
        return sorted(labs)

    def get_graph(self):                                         
        hsl = self.hetero_site_list
        hcm = self.connectivity_matrix.copy()
        surfhcm = hcm[self.surf_ids]
        symbols = self.symbols[self.surf_ids]
        nrows, ncols = surfhcm.shape[0], surfhcm.shape[1]        
        newrows, frag_list = [], []
        for st in hsl:
            if st['occupied'] == 1:
                si = st['indices']
                newrow = np.zeros(ncols)
                newrow[list(si)] = 1
                newrows.append(newrow)
                frag_list.append(st['fragment'])
        if newrows:
            surfhcm = np.vstack((surfhcm, np.asarray(newrows)))

        G = nx.Graph()               
        # Add nodes from label list
        G.add_nodes_from([(i, {'symbol': symbols[i]}) 
                           for i in range(nrows)] + 
                         [(j + nrows, {'symbol': frag_list[j]})
                           for j in range(len(frag_list))])
        # Add edges from surface connectivity matrix
        shcm = surfhcm[:,self.surf_ids]
        shcm *= np.tri(*shcm.shape, k=-1)
        rows, cols = np.where(shcm == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G.add_edges_from(edges)

        return G

    def get_site_graph(self):                                         
        ll = self.label_list
        scm = self.get_site_connectivity()

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
        nsubsurf = len(self.cas.get_subsurface())
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

 
class SlabAdsorbateCoverage(object):

    """dmax: maximum bond length [Ã] that should be considered as an adsorbate"""        

    def __init__(self, atoms, 
                 adsorption_sites=None, 
                 surface=None, 
                 dmax=2.5):

        self.atoms = atoms.copy()
        self.positions = atoms.positions
        self.symbols = atoms.symbols
        self.ads_ids = [a.index for a in atoms if 
                        a.symbol in adsorbate_elements]
        assert self.ads_ids
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
        self.hetero_site_list = deepcopy(sas.site_list)
        self.unique_sites = sas.get_unique_sites(unique_composition=
                                                 self.composition_effect) 
        self.label_dict = self.get_bimetallic_label_dict() \
                          if self.composition_effect else \
                          self.get_monometallic_label_dict()

        self.label_list = ['0'] * len(self.hetero_site_list)
        self.label_occupied_sites()
        self.labels = self.get_labels()

    def identify_adsorbates(self):
        G = nx.Graph()
        adscm = self.ads_connectivity_matrix

        # Cut off all intermolecular H-H bonds except intramolecular        
        # H-H bonds in e.g. H2
        hids = [a.index for a in self.ads_atoms if a.symbol == 'H']
        for hi in hids:
            conns = np.where(adscm[hi] == 1)[0]
            hconns = [i for i in conns if self.ads_atoms.symbols[i] == 'H']
            if hconns and len(conns) > 1:
                adscm[hi,hconns] = 0

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

    def get_hetero_connectivity(self):
        """Generate a connection matrix of slab + adsorbates."""
        nbslist = neighbor_shell_list(self.atoms, 0.3, neighbor_number=1)
        return get_connectivity_matrix(nbslist)                           

    def get_ads_connectivity(self):
        """Generate a connection matrix for adsorbate atoms."""
        return get_connectivity_matrix(self.ads_nblist) 

    def get_site_connectivity(self):
        """Generate a connection matrix for adsorption sites."""
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
            adssym = next((k for k, v in adsorbate_formulas.items() 
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
                    fsym = next((f for f in adsorbate_fragments(adssym) 
                                 if f[0] == bondsym), None)
                    st['fragment'] = fsym
                    flen = len(list(Formula(fsym)))
                    adsids = st['adsorbate_indices']
                    ibond = adsids.index(bondid)
                    fsi = adsids[ibond:ibond+flen]
                    st['fragment_indices'] = fsi
                else:
                    st['fragment_indices'] = st['adsorbate_indices'] 
                signature = [st['site'], st['geometry']]                     
                if self.composition_effect:
                    signature.append(st['composition'])
                stlab = self.label_dict['|'.join(signature)]
                label = str(stlab) + st['fragment']
                st['label'] = label
                ll[j] = label

    def make_ads_neighbor_list(self, dx=.2, neighbor_number=1):
        """Generate a periodic neighbor list (defaultdict).""" 
        self.ads_nblist = neighbor_shell_list(self.ads_atoms, dx, 
                                              neighbor_number, mic=True)

    def get_labels(self):
        ll = self.label_list
        labs = [lab for lab in ll if lab != '0']
        return sorted(labs)

    def get_graph(self):                                         
        hsl = self.hetero_site_list
        hcm = self.connectivity_matrix.copy()
        surfhcm = hcm[self.surf_ids]
        symbols = self.symbols[self.surf_ids]
        nrows, ncols = surfhcm.shape[0], surfhcm.shape[1]        
        newrows, frag_list = [], []
        for st in hsl:
            if st['occupied'] == 1:
                si = st['indices']
                newrow = np.zeros(ncols)
                newrow[list(si)] = 1
                newrows.append(newrow)
                frag_list.append(st['fragment'])
        if newrows:
            surfhcm = np.vstack((surfhcm, np.asarray(newrows)))

        G = nx.Graph()               
        # Add nodes from label list
        G.add_nodes_from([(i, {'symbol': symbols[i]}) 
                           for i in range(nrows)] + 
                         [(j + nrows, {'symbol': frag_list[j]})
                           for j in range(len(frag_list))])
        # Add edges from surface connectivity matrix
        shcm = surfhcm[:,self.surf_ids]
        shcm *= np.tri(*shcm.shape, k=-1)
        rows, cols = np.where(shcm == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G.add_edges_from(edges)

        return G

    def get_site_graph(self):                                         
        ll = self.label_list
        scm = self.get_site_connectivity()

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

