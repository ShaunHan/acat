from .adsorption_sites import * 
from .utilities import * 
from ase.io import read, write
from ase.build import molecule
from ase.collections import g2
from ase.data import covalent_radii, atomic_numbers
from ase.neighborlist import NeighborList
from ase.visualize import view
from ase import Atom, Atoms
from collections import defaultdict, Iterable, Counter
from operator import itemgetter
import networkx as nx
import numpy as np
import random
import copy
import re


# Set global variables
ads_elements = 'SCHON'
heights_dict = {'ontop': 2.0, 
                'bridge': 1.8, 
                'fcc': 1.8, 
                'hcp': 1.8, 
                '4fold': 1.7}
# Make your own adsorbate label dict
adsorbate_label_dict = {'H': 1,
                        'C': 2,
                        'O': 3,
                        'CH': 4,
                        'OH': 5,
                        'CO': 6,
                        'CH2': 7, # Not supplied by ASE
                        'H2O': 8,
                        'COH': 9, # Vertical, Not supplied by ASE
                        'CH3': 10,
                        'H2CO': 11, # Vertical
                        # Bidentate adsorbates
                        'HCO': 12, 
                        'CHOH': 13, # Not supplied by ASE
                        'CH2O': 14, # Not supplied by ASE
                        'CH3O': 15, 
                        'H2COH': 16, 
                        'CH3OH': 17} 


def get_label_dict(surface):

    if surface == 'fcc111':
        return {'ontop|111': 1,
                'bridge|111': 2,
                'fcc|111': 3,
                'hcp|111': 4}

    elif surface == 'fcc100':
        return {'ontop|100': 1,
                'bridge|100': 2,
                '4fold|100': 3}

    elif surface == 'fcc110':
        return {'ontop|step': 1,
                'ontop|terrace': 2,
                'bridge|step': 3, 
                'bridge|terrace': 4, 
                'bridge|111': 5,
                'fcc|111': 6,
                'hcp|111': 7}

    elif surface == 'fcc211':
        return {'ontop|step': 1,
                'ontop|terrace': 2,
                'ontop|lowerstep': 3, 
                'bridge|step': 4,
                'bridge|upper111': 5,
                'bridge|lower111': 6,
                'bridge|100': 7,
                'fcc|111': 8,
                'hcp|111': 9,
                '4fold|100': 10}

    elif surface == 'fcc311':
        return {'ontop|step': 1,
                'ontop|terrace': 2,
                'bridge|step': 3,
                'bridge|terrace': 4,
                'bridge|111': 5,
                'bridge|100': 6,
                'fcc|111': 7,
                'hcp|111': 8,
                '4fold|100': 9}


class NanoparticleAdsorbateCoverage(NanoparticleAdsorptionSites):
    None


class SlabAdsorbateCoverage(SlabAdsorptionSites):

    """dmax: maximum bond length (Å) that should be considered as an adsorbate"""

    def __init__(self, atoms, adsorption_sites, surface=None, dmax=2.2):
 
        self.atoms = atoms.copy()
        self.ads_ids = [a.index for a in atoms if a.symbol in ads_elements]
        assert len(self.ads_ids) > 0 
        self.ads_atoms = atoms[self.ads_ids]
        self.cell = atoms.cell
        self.pbc = atoms.pbc
        self.dmax = dmax

        self.ads_names = g2.names
        self.ads_symbols = [sorted(molecule(n).get_chemical_symbols())
                            for n in self.ads_names]
        self.make_ads_neighbor_list()
        self.ads_connectivity_matrix = self.get_ads_connectivity() 
        self.identify_adsorbates()

        self.slab = adsorption_sites.atoms
        self.surface = adsorption_sites.surface
        self.show_composition = adsorption_sites.show_composition
        self.show_subsurface = adsorption_sites.show_subsurface
        self.surf_ids = adsorption_sites.surf_ids
        self.connectivity_matrix = adsorption_sites.connectivity_matrix
        self.full_site_list = adsorption_sites.site_list.copy()
        self.clean_list()

        self.unique_sites = adsorption_sites.get_unique_sites(
                            unique_composition=self.show_composition,
                            unique_subsurface=self.show_subsurface) 

        self.label_dict = self.get_bimetallic_label_dict() \
                          if self.show_composition else \
                          get_label_dict(self.surface)
        self.site_matrix = np.zeros((len(self.full_site_list), 
                                     len(self.label_dict)))
#        self.chem_env_matrix = np.zeros((len(self.slab), 
#                                         len(self.unique_sites)))

        self.label_occupied_sites()
        self.surf_connectivity_matrix = self.get_surf_connectivity()
        self.site_connectivity_matrix = self.get_site_connectivity()
#        self.surf_chem_env_matrix = self.chem_env_matrix[self.surf_ids]
        self.chem_env = sorted(self.site_matrix.tolist())

    def identify_adsorbates(self):
        G = nx.Graph()
        adscm = self.ads_connectivity_matrix
      
        if adscm.size != 0:
            np.fill_diagonal(adscm, 1)
            print(adscm)
            rows, cols = np.where(adscm == 1)

            edges = zip([self.ads_ids[row] for row in rows.tolist()], 
                        [self.ads_ids[col] for col in cols.tolist()])
            G.add_edges_from(edges)                        
            SG = (G.subgraph(c) for c in nx.connected_components(G))
 
            adsorbates = []
            for sg in SG:
                nodes = list(sg.nodes)
                adsorbates += [nodes]
        else:
            adsorbates = [self.ads_ids]
        self.ads_list = adsorbates

    def clean_list(self):
        sl = self.full_site_list
        entries = ['occupied', 'adsorbate', 'adsorbate_indices', 
                   'bonded_index', 'bond_length', 'label', 'dentate']
        for d in sl:
            for k in entries:
                if k in d:
                    del d[k]

    def get_ads_connectivity(self):
        """Generate a connections matrix for adsorbate atoms."""
        return get_connectivity_matrix(self.ads_nblist) 

    def get_surf_connectivity(self):
        """Generate a connections matrix for surface atoms."""
        cm = self.connectivity_matrix
        return cm[self.surf_ids][:,self.surf_ids]

    def get_site_connectivity(self):
        """Generate a connections matrix for adsorption sites."""
        sl = self.full_site_list
        conn_mat = []
        for i, sti in enumerate(sl):
            conn_x = []
            for j, stj in enumerate(sl):
                if i == j:
                    conn_x.append(0.)
                elif bool(set(sti['indices']).intersection(
                stj['indices'])):
                    conn_x.append(1.)                     
                else:
                    conn_x.append(0.)
            conn_mat.append(conn_x)   

        return np.asarray(conn_mat) 

    def label_occupied_sites(self):
        fsl = self.full_site_list
        sm = self.site_matrix
#        cem = self.chem_env_matrix
        ads_list = self.ads_list
        ndentate_dict = {}
 
        for adsid in self.ads_ids:
            tag = self.atoms[adsid].tag
            if self.atoms[adsid].symbol == 'H' and tag > 1:
                continue

            def get_bond_length(site):
                return get_mic_distance(self.atoms[adsid].position, 
                                        site['position'], 
                                        self.cell, self.pbc)
            st, bl = min(((s, get_bond_length(s)) for s in fsl), 
                           key=itemgetter(1))
            if bl > self.dmax:
                continue
            adsids = next((l for l in ads_list if adsid in l), None)

            print(ads_list)
            print(adsid)


            if not adsids:
                view(self.atoms)




            adsi = tuple(sorted(adsids))
            if 'occupied' in st:
                if bl >= st['bond_length']:
                    continue
                else:
                    ndentate_dict[adsi] -= 1

            if tag != 0:
                for k, v in adsorbate_label_dict.items():
                    if v == tag:
                        adssym = k
            else:
                adssyms = [self.atoms[i].symbol for i in adsids]
                try:
                    adssym = self.ads_names[self.ads_symbols.index(
                                            sorted(adssyms))] 
                except:
                    adssym = ''.join(adssyms)

            st['adsorbate'] = adssym
            st['adsorbate_group'] = adssym
            st['adsorbate_indices'] = adsi
            st['bonded_index'] = adsid
            st['bond_length'] = bl
            if adsi in ndentate_dict:
                ndentate_dict[adsi] += 1
            else:
                ndentate_dict[adsi] = 1
            st['occupied'] = 1            
#            for i in st['indices']:
#                cem[i, label-1] += 1

        # Get bidentate fragments, site matrix, label and dentate numbers   
        for j, st in enumerate(fsl):
            if 'occupied' not in st:
                st['adsorbate'] = st['adsorbate_group'] = \
                st['adsorbate_indices'] = st['bonded_index'] = \
                st['bond_length'] = None
                st['occupied'] = st['label'] = 0
                continue

            if st['adsorbate_group'] == 'HCO':
                bondsym = self.atoms[st['bonded_index']].symbol
                if bondsym == 'C':
                    st['adsorbate'] = 'CH'
                elif bondsym == 'O':
                    st['adsorbate'] = 'O'
            elif st['adsorbate_group'] == 'CHOH':
                bondsym = self.atoms[st['bonded_index']].symbol
                if bondsym == 'C':
                    st['adsorbate'] = 'CH'
                elif bondsym == 'O':
                    st['adsorbate'] = 'OH'
            elif st['adsorbate_group'] == 'CH2O':
                bondsym = self.atoms[st['bonded_index']].symbol
                if bondsym == 'C':
                    st['adsorbate'] = 'CH2' 
                elif bondsym == 'O':
                    st['adsorbate'] = 'O'
            elif st['adsorbate_group'] == 'CH3O':
                bondsym = self.atoms[st['bonded_index']].symbol
                if bondsym == 'C':
                    st['adsorbate'] = 'CH3'
                elif bondsym == 'O':
                    st['adsorbate'] = 'O'
            elif st['adsorbate_group'] == 'H2COH':
                bondsym = self.atoms[st['bonded_index']].symbol
                if bondsym == 'C':
                    st['adsorbate'] = 'CH2'
                elif bondsym == 'O':
                    st['adsorbate'] = 'OH'
            elif st['adsorbate_group'] == 'CH3OH':
                bondsym = self.atoms[st['bonded_index']].symbol
                if bondsym == 'C':
                    st['adsorbate'] = 'CH3'
                elif bondsym == 'O':
                    st['adsorbate'] = 'OH'

            adsi = st['adsorbate_indices']
            ads = st['adsorbate']

            signature = [st['site'], st['geometry']]                        
            if self.show_composition:
                signature.append(st['composition'])
                if self.show_subsurface:
                    signature.append(st['subsurface_element'])
            else:
                if self.show_subsurface:
                    raise ValueError('To include the subsurface element, ',
                                     'show_composition also need to be ',
                                     'set to True in adsorption_sites')    
            label = self.label_dict['|'.join(signature)]
            st['label'] = label
            if ads in adsorbate_label_dict:
                sm[j,label-1] = adsorbate_label_dict[ads]
            if adsi in ndentate_dict:
                st['dentate'] = ndentate_dict[adsi]
            else:
                st['dentate'] = 0

    def make_ads_neighbor_list(self, dx=.3, neighbor_number=1):
        """Generate a periodic neighbor list (defaultdict).""" 
        self.ads_nblist = neighbor_shell_list(self.ads_atoms, dx, 
                                              neighbor_number, mic=True)

#    def point_in_cylinder(self, point, site, hmax):
#        """Check if a point is in the cylinder space on a site"""
#        projection = point_projection_on_line(point, site['position'], 
#                     site['normal'], hmax)
#        maxpt = site['normal'] * hmax
#        nums = [self.atoms[i].number for i in site['indices']]
#        rmax = np.average([covalent_radii[n] for n in nums], 0) / 2
#        within_rmax = get_mic_distance(point, projection, self.cell, 
#                      self.pbc) < rmax 
#        within_hmax = np.dot(point - maxpt, 
#                             maxpt - site['position']) <= 0
#
#        return (within_rmax and within_hmax)

    def get_site_graph(self):                                         
        sm = self.site_matrix
        scm = self.site_connectivity_matrix
        sm_list = [''.join(r.astype(int).astype(str)) for r in sm]

        G = nx.Graph()                                                  
        # Add nodes from surface chemical environment matrix
        G.add_nodes_from([(i, {'chem_env': np.array(sm_list)[i]}) 
                                  for i in range(scm.shape[0])])
        # Add edges from surface connectivity matrix
        rows, cols = np.where(scm == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G.add_edges_from(edges)

        return G

    def get_surface_graph(self):                                        
        surfcem = self.surf_chem_env_matrix
        surfcm = self.site_connectivity_matrix
        cem_list = [''.join(r.astype(int).astype(str)) for r in surfcem]

        G = nx.Graph()                                                  
        # Add nodes from surface chemical environment matrix
        G.add_nodes_from([(i, {'chem_env': np.array(cem_list)[i]}) 
                                  for i in range(surfcm.shape[0])])
        # Add edges from surface connectivity matrix
        rows, cols = np.where(surfcm == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G.add_edges_from(edges)

        return G

    def draw_graph(self, G):
        import matplotlib.pyplot as plt

        nx.draw(G, with_labels=True)
#        plt.savefig("graph.png")
        plt.show() 


    def get_bimetallic_label_dict(self):
    
        metals = list(set(self.slab.symbols))      
        ma, mb = metals[0], metals[1]
        if atomic_numbers[ma] > atomic_numbers[mb]:
            ma, mb = metals[1], metals[0]
 
        if self.surface == 'fcc111':
            return {'ontop|111|{}'.format(ma): 1, 
                    'ontop|111|{}'.format(mb): 2,
                    'bridge|111|{}{}'.format(ma,ma): 3, 
                    'bridge|111|{}{}'.format(ma,mb): 4,
                    'bridge|111|{}{}'.format(mb,mb): 5, 
                    'fcc|111|{}{}{}'.format(ma,ma,ma): 6,
                    'fcc|111|{}{}{}'.format(ma,ma,mb): 7, 
                    'fcc|111|{}{}{}'.format(ma,mb,mb): 8,
                    'fcc|111|{}{}{}'.format(mb,mb,mb): 9,
                    'hcp|111|{}{}{}'.format(ma,ma,ma): 10,
                    'hcp|111|{}{}{}'.format(ma,ma,mb): 11,
                    'hcp|111|{}{}{}'.format(ma,mb,mb): 12,
                    'hcp|111|{}{}{}'.format(mb,mb,mb): 13}
    
        elif self.surface == 'fcc100':
            return {'ontop|100|{}'.format(ma): 1, 
                    'ontop|100|{}'.format(mb): 2,
                    'bridge|100|{}{}'.format(ma,ma): 3, 
                    'bridge|100|{}{}'.format(ma,mb): 4,
                    'bridge|100|{}{}'.format(mb,mb): 5, 
                    '4fold|100|{}{}{}{}'.format(ma,ma,ma,ma): 6,
                    '4fold|100|{}{}{}{}'.format(ma,ma,ma,mb): 7, 
                    '4fold|100|{}{}{}{}'.format(ma,ma,mb,mb): 8,
                    '4fold|100|{}{}{}{}'.format(ma,mb,ma,mb): 9, 
                    '4fold|100|{}{}{}{}'.format(ma,mb,mb,mb): 10,
                    '4fold|100|{}{}{}{}'.format(mb,mb,mb,mb): 11}
    
        elif self.surface == 'fcc110':
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                   # neightbor elements count clockwise from shorter bond ma
                    'ontop|terrace|{}-{}{}{}{}'.format(ma,ma,ma,ma,ma): 3,
                    'ontop|terrace|{}-{}{}{}{}'.format(ma,ma,ma,ma,mb): 4,
                    'ontop|terrace|{}-{}{}{}{}'.format(ma,ma,ma,mb,mb): 5,
                    'ontop|terrace|{}-{}{}{}{}'.format(ma,ma,mb,ma,mb): 6,
                    'ontop|terrace|{}-{}{}{}{}'.format(ma,ma,mb,mb,ma): 7,
                    'ontop|terrace|{}-{}{}{}{}'.format(ma,ma,mb,mb,mb): 8,
                    'ontop|terrace|{}-{}{}{}{}'.format(ma,mb,mb,mb,mb): 9,
                    'ontop|terrace|{}-{}{}{}{}'.format(mb,ma,ma,ma,ma): 10,
                    'ontop|terrace|{}-{}{}{}{}'.format(mb,ma,ma,ma,mb): 11,
                    'ontop|terrace|{}-{}{}{}{}'.format(mb,ma,ma,mb,mb): 12,
                    'ontop|terrace|{}-{}{}{}{}'.format(mb,ma,mb,ma,mb): 13,
                    'ontop|terrace|{}-{}{}{}{}'.format(mb,ma,mb,mb,ma): 14,
                    'ontop|terrace|{}-{}{}{}{}'.format(mb,ma,mb,mb,mb): 15,
                    'ontop|terrace|{}-{}{}{}{}'.format(mb,mb,mb,mb,mb): 16,
                    'bridge|step|{}{}'.format(ma,ma): 17,
                    'bridge|step|{}{}'.format(ma,mb): 18,
                    'bridge|step|{}{}'.format(mb,mb): 19, 
                    'bridge|terrace|{}{}-{}{}'.format(ma,ma,ma,ma): 20,
                    'bridge|terrace|{}{}-{}{}'.format(ma,ma,ma,mb): 21,
                    'bridge|terrace|{}{}-{}{}'.format(ma,ma,mb,mb): 22,
                    'bridge|terrace|{}{}-{}{}'.format(ma,mb,ma,ma): 23,
                    'bridge|terrace|{}{}-{}{}'.format(ma,mb,ma,mb): 24,
                    'bridge|terrace|{}{}-{}{}'.format(ma,mb,mb,mb): 25,
                    'bridge|terrace|{}{}-{}{}'.format(mb,mb,ma,ma): 26,
                    'bridge|terrace|{}{}-{}{}'.format(mb,mb,ma,mb): 27,
                    'bridge|terrace|{}{}-{}{}'.format(mb,mb,mb,mb): 28,
                    'bridge|111|{}{}'.format(ma,ma): 29,
                    'bridge|111|{}{}'.format(ma,mb): 30,
                    'bridge|111|{}{}'.format(mb,mb): 31,
                    'fcc|111|{}{}{}'.format(ma,ma,ma): 32,
                    'fcc|111|{}{}{}'.format(ma,ma,mb): 33, 
                    'fcc|111|{}{}{}'.format(ma,mb,mb): 34,
                    'fcc|111|{}{}{}'.format(mb,mb,mb): 35,
                    'hcp|111|{}{}{}'.format(ma,ma,ma): 36,
                    'hcp|111|{}{}{}'.format(ma,ma,mb): 37,
                    'hcp|111|{}{}{}'.format(ma,mb,mb): 38,
                    'hcp|111|{}{}{}'.format(mb,mb,mb): 39}
    
        elif self.surface == 'fcc211':
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'ontop|terrace|{}'.format(ma): 3,
                    'ontop|terrace|{}'.format(mb): 4,
                    'ontop|lowerstep|{}'.format(mb): 5,
                    'ontop|lowerstep|{}'.format(mb): 6,
                    'bridge|step|{}{}'.format(ma,ma): 7, 
                    'bridge|step|{}{}'.format(ma,mb): 8,
                    'bridge|step|{}{}'.format(mb,mb): 9,
                    'bridge|lowerstep|{}{}'.format(ma,ma): 10,
                    'bridge|lowerstep|{}{}'.format(ma,mb): 11,
                    'bridge|lowerstep|{}{}'.format(mb,mb): 12,
                    'bridge|upper111|{}{}'.format(ma,ma): 13,
                    'bridge|upper111|{}{}'.format(ma,mb): 14,
                    'bridge|upper111|{}{}'.format(mb,mb): 15,
                    # terrace bridge is equivalent to lower111 bridge
                    'bridge|lower111|{}{}'.format(ma,ma): 16,
                    'bridge|lower111|{}{}'.format(ma,mb): 17,
                    'bridge|lower111|{}{}'.format(mb,mb): 18,
                    'bridge|100|{}{}'.format(ma,ma): 19,
                    'bridge|100|{}{}'.format(ma,mb): 20,
                    'bridge|100|{}{}'.format(mb,mb): 21,
                    'fcc|upper111|{}{}{}'.format(ma,ma,ma): 22,
                    'fcc|upper111|{}{}{}'.format(ma,ma,mb): 23, 
                    'fcc|upper111|{}{}{}'.format(ma,mb,mb): 24,
                    'fcc|upper111|{}{}{}'.format(mb,mb,mb): 25,
                    'hcp|upper111|{}{}{}'.format(ma,ma,ma): 26,
                    'hcp|upper111|{}{}{}'.format(ma,ma,mb): 27,
                    'hcp|upper111|{}{}{}'.format(ma,mb,mb): 28,
                    'hcp|upper111|{}{}{}'.format(mb,mb,mb): 29,
                    'fcc|lower111|{}{}{}'.format(ma,ma,ma): 30,
                    'fcc|lower111|{}{}{}'.format(ma,ma,mb): 31,
                    'fcc|lower111|{}{}{}'.format(ma,mb,mb): 32,
                    'fcc|lower111|{}{}{}'.format(mb,mb,mb): 33,
                    'hcp|lower111|{}{}{}'.format(ma,ma,ma): 34,
                    'hcp|lower111|{}{}{}'.format(ma,ma,mb): 35,
                    'hcp|lower111|{}{}{}'.format(ma,mb,mb): 36,
                    'hcp|lower111|{}{}{}'.format(mb,mb,mb): 37,
                    '4fold|100|{}{}{}{}'.format(ma,ma,ma,ma): 38,
                    '4fold|100|{}{}{}{}'.format(ma,ma,ma,mb): 39, 
                    '4fold|100|{}{}{}{}'.format(ma,ma,mb,mb): 40,
                    '4fold|100|{}{}{}{}'.format(ma,mb,ma,mb): 41, 
                    '4fold|100|{}{}{}{}'.format(ma,mb,mb,mb): 42,
                    '4fold|100|{}{}{}{}'.format(mb,mb,mb,mb): 43}
    
        elif self.surface == 'fcc311':
            return {'ontop|step|{}'.format(ma): 1,
                    'ontop|step|{}'.format(mb): 2,
                    'ontop|terrace|{}'.format(ma): 3 ,
                    'ontop|terrace|{}'.format(mb): 4,
                    'bridge|step|{}{}'.format(ma,ma): 5,
                    'bridge|step|{}{}'.format(ma,mb): 6,
                    'bridge|step|{}{}'.format(mb,mb): 7,
                    'bridge|terrace|{}{}'.format(ma,ma): 8,
                    'bridge|terrace|{}{}'.format(ma,mb): 9,
                    'bridge|terrace|{}{}'.format(mb,mb): 10,
                    'bridge|111|{}{}'.format(ma,ma): 11,
                    'bridge|111|{}{}'.format(ma,mb): 12,
                    'bridge|111|{}{}'.format(mb,mb): 13,
                    'bridge|100|{}{}'.format(ma,ma): 14,
                    'bridge|100|{}{}'.format(ma,mb): 15,
                    'bridge|100|{}{}'.format(mb,mb): 16,
                    'fcc|111|{}{}{}'.format(ma,ma,ma): 17,
                    'fcc|111|{}{}{}'.format(ma,ma,mb): 18,
                    'fcc|111|{}{}{}'.format(ma,mb,mb): 19,
                    'fcc|111|{}{}{}'.format(mb,mb,mb): 20,
                    'hcp|111|{}{}{}'.format(ma,ma,ma): 21,
                    'hcp|111|{}{}{}'.format(ma,ma,mb): 22,
                    'hcp|111|{}{}{}'.format(ma,mb,mb): 23,
                    'hcp|111|{}{}{}'.format(mb,mb,mb): 24,
                    '4fold|100|{}{}{}{}'.format(ma,ma,ma,ma): 25,
                    '4fold|100|{}{}{}{}'.format(ma,ma,ma,mb): 26, 
                    '4fold|100|{}{}{}{}'.format(ma,ma,mb,mb): 27,
                    '4fold|100|{}{}{}{}'.format(ma,mb,ma,mb): 28, 
                    '4fold|100|{}{}{}{}'.format(ma,mb,mb,mb): 29,
                    '4fold|100|{}{}{}{}'.format(mb,mb,mb,mb): 30}                    
 

def add_adsorbate_to_site(atoms, adsorbate, site, height=None, 
                          rotation=None, tag=0):            

    '''rotation: vector that the adsorbate is rotated into'''

    
    if height is None:
        height = heights_dict[site['site']]

    # Make the correct position
    normal = site['normal']
    pos = site['position'] + normal * height

    # Convert the adsorbate to an Atoms object
    if isinstance(adsorbate, Atoms):
        ads = adsorbate
        if tag != 0:
            for a in ads:
                a.tag = tag
    elif isinstance(adsorbate, Atom):
        ads = Atoms([adsorbate])
        if tag != 0:
            for a in ads:
                a.tag = tag
    else:
        # Assume it is a string representing a molecule
        if adsorbate in ['CO', 'OCS']:
            ads = molecule(adsorbate)[::-1]
        elif adsorbate in ['H2O', 'NH2']:
            ads = molecule(adsorbate)
            ads.rotate(180, 'y')
        elif adsorbate == 'CH2':
            ads = molecule('NH2')
            ads[0].symbol = 'C'
            ads.rotate(180, 'y')
        elif adsorbate == 'COH':
            ads = molecule('H2COH')
            del ads[-2:]
            ads.rotate(90, 'y') 
        elif adsorbate == 'H2CO':
            ads = molecule(adsorbate)[::-1]
            newpos = ads.positions[:2]
            del ads[:2]
            ads.extend(Atoms('H2', newpos))
        elif adsorbate == 'CH2O':
            ads = molecule('H2CO')[::-1]
            ads.rotate(-90, '-y')
            newpos = ads.positions[:2]
            del ads[:2]
            ads.extend(Atoms('H2', newpos))
        elif adsorbate == 'CHOH':
            ads = molecule('H2COH')
            del ads[-1]
        else:
            ads = molecule(adsorbate)
 
        if len(ads) == 2 or (len(ads) > 2 and adsorbate in ['COH','H2CO']):
            if len(ads) == 2 or adsorbate == 'COH':
                adsi = [1]
            else:
                adsi = [a.index+1 for a in ads[1:] if a.symbol != 'H']
            avg_pos = np.average(ads[adsi].positions, 0)
            ads.rotate(avg_pos - ads[0].position, normal)
            #pvec = np.cross(np.random.rand(3) - ads[0].position, normal)
            #ads.rotate(-45, pvec, center=ads[0].position)

        # Set tags
        if (tag == 0) and (adsorbate in adsorbate_label_dict):
            for a in ads:
                a.tag = adsorbate_label_dict[adsorbate]

    if rotation is not None:
        ads.rotate(np.average(ads.positions[1:],0) - ads[0].position, rotation)
    ads.translate(pos - ads[0].position)

    atoms.extend(ads)


def add_adsorbate(atoms, adsorbate, site, surface=None, geometry=None, 
                  indices=None, height=None, composition=None, 
                  subsurface_element=None, site_list=None):
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
    show_composition = False if composition is None else True
    show_subsurface = False if subsurface_element is None else True
    if composition:
        if '-' in composition:
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
                    geometry, show_composition, show_subsurface)

    if indices:
        if not isinstance(indices, Iterable):
            indices = [indices]
        indices = tuple(sorted(indices))
        si = next((s for s in all_sites if 
                   s['indices'] == indices), None)
    else:
        si = next((s for s in all_sites if 
                   s['site'] == site and
                   s['composition'] == scomp and 
                   s['subsurface_element'] 
                   == subsurface_element), None)

    if not si:
        print('No such site can be found')            
    else:
        add_adsorbate_to_site(atoms, adsorbate, si, height)


def add_cluster(atoms, cluster, site, surface=None, geometry=None, 
                indices=None, height=None, composition=None, 
                subsurface_element=None, site_list=None):

    None


def get_coverage(atoms, adsorbate, surface=None, nfullsite=None):
    """Get the coverage of a nanoparticle / surface with adsorbates.
       Provide the number of adsorption sites under 1 ML coverage will
       significantly accelerate the calculation."""

    sites = enumerate_monometallic_sites(atoms, surface=surface, 
            show_subsurface=False)
    if nfullsite is None:
        pattern = pattern_generator(atoms, 'O', ['fcc111', 'fcc100'], 
                                    coverage=1)
        nfullsite = len([a for a in pattern if a.symbol == 'O'])
    symbol = list(adsorbate)[0]
    nadsorbate = 0
    for a in atoms:
        if a.symbol == symbol:
            nadsorbate += 1

    return nadsorbate / nfullsite


def group_sites_by_surface(atoms, sites):                            
    """A function that uses networkx to group one type of sites 
    by geometrical facets of the nanoparticle"""

    # Find all indices of vertex and edge sites
    ve_indices = [s['indices'] for s in sites if s['site'] == 'ontop'
                  and s['surface'] in ['vertex', 'edge']]

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
                   a.symbol in ads_elements]
    ads_atoms = None
    if ads_indices:
        ads_atoms = atoms[ads_indices]
        atoms = atoms[[a.index for a in atoms if 
                       a.symbol not in ads_elements]]
    ads = molecule(adsorbate)[::-1]
    if str(ads.symbols) != 'CO':
        ads.set_chemical_symbols(
        ads.get_chemical_symbols()[::-1])

    if True not in atoms.pbc:
        if surface is None:
            surface = ['fcc100', 'fcc111']        
        nas = NanoparticleAdsorptionSites(atoms)
        site_list = nas.site_list
    else:
        sas = SlabAdsorptionSites(atoms, surface=surface)
        if surface is None:
            surface = sas.surface
        site_list = sas.site_list
    if not isinstance(surface, list):
        surface = [surface] 

    final_sites = []
    if not set(surface).isdisjoint(['fcc111','fcc110',
                                    'fcc211','fcc311']): 
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
                grouped_sites = group_sites_by_surface(atoms, 
                                                       fcc_sites)
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
                grouped_sites = group_sites_by_surface(atoms, all_sites)
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
                grouped_sites = group_sites_by_surface(atoms, 
                                                       fcc_sites)
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

    if not set(surface).isdisjoint(['fcc100','fcc211','fcc311']):
        if coverage == 1:
            fold4_sites = [s for s in site_list if s['site'] == '4fold']
            if fold4_sites:
                final_sites += fold4_sites

        elif coverage == 3/4:
            fold4_sites = [s for s in site_list if s['site'] == '4fold']
            if True not in atoms.pbc:                                           
                grouped_sites = group_sites_by_surface(atoms, fold4_sites)
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
                grouped_sites = group_sites_by_surface(atoms, fold4_sites)
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
                grouped_sites = group_sites_by_surface(atoms, fold4_sites)
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
        remove_adsorbates_too_close(atoms, min_adsorbate_distance)

    return atoms


def remove_adsorbates_too_close(atoms, min_adsorbate_distance=0.6):
 
    rmin = min_adsorbate_distance/2.9
    nl = NeighborList([rmin for a in atoms], self_interaction=False, bothways=True)   
    nl.update(atoms)            
    atom_indices = [a.index for a in atoms if a.symbol == adsorbate[-1]]
    ads_symbols = molecule(adsorbate).get_chemical_symbols()
    n_ads_atoms = len(ads_symbols)
    overlap_atoms_indices = []
    for idx in atom_indices:   
        neighbor_indices, _ = nl.get_neighbors(idx)
        overlap = 0
        for i in neighbor_indices:                                                    
            if (atoms[i].symbol in ads_elements) and (i not in overlap_atoms_indices):  
                overlap += 1                                                          
        if overlap > 0:                                                               
            overlap_atoms_indices += list(set([idx-n_ads_atoms+1, idx]))              
    del atoms[overlap_atoms_indices]                                      


def full_coverage_pattern_generator(atoms, adsorbate, site, height=None, 
                                    min_adsorbate_distance=0.6):
    '''A function to generate different 1ML coverage patterns'''

    rmin = min_adsorbate_distance/2.9
    ads_indices = [a.index for a in atoms if a.symbol in ads_elements]
    ads_atoms = None
    if ads_indices:
        ads_atoms = atoms[ads_indices]
        atoms = atoms[[a.index for a in atoms if a.symbol not in ads_elements]]
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
            nl = NeighborList([rmin for a in atoms], self_interaction=False, bothways=True)     
            nl.update(atoms)            
            atom_indices = [a.index for a in atoms if a.symbol == 'O']            
            n_ads_atoms = 2
            overlap_atoms_indices = []
            for idx in atom_indices:   
                neighbor_indices, _ = nl.get_neighbors(idx)
                overlap = 0
                for i in neighbor_indices:
                    if (atoms[i].symbol in ads_elements) and (i not in overlap_atoms_indices):
                        overlap += 1
                if overlap > 0:
                    overlap_atoms_indices += list(set([idx-n_ads_atoms+1, idx]))
            del atoms[overlap_atoms_indices]

        else:
            for site in final_sites:
                add_adsorbate(atoms, molecule(adsorbate), site)
            nl = NeighborList([rmin for a in atoms], self_interaction=False, bothways=True)   
            nl.update(atoms)            
            atom_indices = [a.index for a in atoms if a.symbol == adsorbate[-1]]
            ads_symbols = molecule(adsorbate).get_chemical_symbols()
            n_ads_atoms = len(ads_symbols)
            overlap_atoms_indices = []
            for idx in atom_indices:   
                neighbor_indices, _ = nl.get_neighbors(idx)
                overlap = 0
                for i in neighbor_indices:                                                                
                    if (atoms[i].symbol in ads_elements) and (i not in overlap_atoms_indices):                       
                        overlap += 1                                                                      
                if overlap > 0:                                                                           
                    overlap_atoms_indices += list(set([idx-n_ads_atoms+1, idx]))                                
            del atoms[overlap_atoms_indices]                                                                    

    else:
        for pos in positions:
            ads.translate(pos - ads[0].position)
            atoms.extend(ads)
        if ads_indices:
            atoms.extend(ads_atoms)

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

    rmin = min_adsorbate_distance/2.9
    ads_indices = [a.index for a in atoms if a.symbol in ads_elements]
    ads_atoms = None
    if ads_indices:
        ads_atoms = atoms[ads_indices]
        atoms = atoms[[a.index for a in atoms if a.symbol not in ads_elements]]
    all_sites = enumerate_monometallic_sites(atoms, surface=surface, 
                                             heights=heights, show_subsurface=False)
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
            atoms.extend(ads)
        if ads_indices:
            atoms.extend(ads_atoms)

    nl = NeighborList([rmin for a in atoms], self_interaction=False, bothways=True)   
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
            if (atoms[i].symbol in ads_elements) and (i not in overlap_atoms_indices):                     
                overlap += 1                                                                      
        if overlap > 0:                                                                           
            overlap_atoms_indices += list(set([idx-n_ads_atoms+1, idx]))                                
    del atoms[overlap_atoms_indices]

    return atoms


def is_site_occupied(atoms, site, min_adsorbate_distance=0.5):
    """Returns True if the site on the atoms object is occupied by
    creating a sphere of radius min_adsorbate_distance and checking
    that no other adsorbate is inside the sphere.
    """

    # if site['occupied']:
    #     return True
    if True not in atoms.pbc:
        height = site['height']
        normal = np.array(site['normal'])
        pos = np.array(site['position']) + normal * height
        dists = [np.linalg.norm(pos - a.position)
                 for a in atoms if a.symbol in ads_elements]
        for d in dists:
            if d < min_adsorbate_distance:
                # print('under min d', d, pos)
                # site['occupied'] = 1
                return True
        return False
    else:
        cell = atoms.get_cell()
        pbc = np.array([cell[0][0], cell[1][1], 0])
        pos = np.array(site['position'])
        dists = [get_mic_distance(pos, a.position, atoms.cell, atoms.pbc) 
                 for a in atoms if a.symbol in ads_elements]
        for d in dists:
            if d < min_adsorbate_distance:
                # print('under min d', d, pos)
                # site['occupied'] = 1
                return True
        return False
                                                                                   
                                                                                   
def is_site_occupied_by(atoms, adsorbate, site, 
                        min_adsorbate_distance=0.5):
    """Returns True if the site on the atoms object is occupied 
    by a specific species.
    """
    # if site['occupied']:
    #     return True
    if True not in atoms.pbc:
        ads_symbols = molecule(adsorbate).get_chemical_symbols()
        n_ads_atoms = len(ads_symbols)
        # play aruond with the cutoff
        height = site['height']
        normal = np.array(site['normal'])
        pos = np.array(site['position']) + normal * height
        dists = []
        for a in atoms:
            if a.symbol in set(ads_symbols):
                dists.append((a.index, np.linalg.norm(pos - a.position)))
        for (i, d) in dists:
            if d < min_adsorbate_distance:
                site_ads_symbols = []
                if n_ads_atoms > 1:
                    for k in range(i,i+n_ads_atoms):
                        site_ads_symbols.append(atoms[k].symbol)
                else:
                    site_ads_symbols.append(atoms[i].symbol)
                if sorted(site_ads_symbols) == sorted(ads_symbols):               
                # print('under min d', d, pos)
                # site['occupied'] = 1
                    return True
        return False
    else:
        ads_symbols = molecule(adsorbate).get_chemical_symbols()
        n_ads_atoms = len(ads_symbols)
        cell = atoms.get_cell()
        pbc = np.array([cell[0][0], cell[1][1], 0])
        pos = np.array(site['position'])
        dists = []
        for a in atoms:
            if a.symbol in set(ads_symbols):
                dists.append((a.index, get_mic_distance(pos, a.position, 
                                                        atoms.cell, atoms.pbc)))
        for (i, d) in dists:
            if d < min_adsorbate_distance:
                site_ads_symbols = []
                if n_ads_atoms > 1:
                    for k in range(i,i+n_ads_atoms):
                        site_ads_symbols.append(atoms[k].symbol)
                else:
                    site_ads_symbols.append(atoms[i].symbol)
                if sorted(site_ads_symbols) == sorted(ads_symbols):               
                # print('under min d', d, pos)
                # site['occupied'] = 1
                    return True
        return False


def label_occupied_sites(atoms, adsorbate, show_subsurface=False):                        
    '''Assign labels to all occupied sites. Different labels represent 
    different sites.
    
    The label is defined as the number of atoms being labeled at that site 
    (considering second shell).
    
    Change the 2 metal elements to 2 pseudo elements for sites occupied by a 
    certain species. If multiple species are present, the 2 metal elements 
    are assigned to multiple pseudo elements. Atoms that are occupied by 
    multiple species also need to be changed to new pseudo elements. Currently 
    only a maximum of 2 species is supported.
    
    Note: Please provide atoms including adsorbate(s), with adsorbate being a 
    string or a list of strings.
    
    Set show_subsurface=True if you also want to label the second shell atoms.'''

    species_pseudo_mapping = [('As','Sb'),('Se','Te'),('Br','I')]  
    elements = list(set(atoms.symbols))
    metals = [element for element in elements if element not in ads_elements]
    mA = metals[0]
    mB = metals[1]
    if Atom(metals[0]).number > Atom(metals[1]).number:
        mA = metals[1]
        mB = metals[0]
    sites = enumerate_adsorption_sites(atoms, show_subsurface=show_subsurface)
    n_occupied_sites = 0
    atoms.set_tags(0)
    if isinstance(adsorbate, list):               
        if len(adsorbate) == 2:
            for site in sites:            
                for ads in adsorbate:
                    k = adsorbate.index(ads)
                    if is_site_occupied_by(atoms, ads, site, 
                                            min_adsorbate_distance=0.5):
                        site['occupied'] = 1
                        site['adsorbate'] = ads
                        indices = site['indices']
                        label = site['label']
                        for idx in indices:                
                            if atoms[idx].tag == 0:
                                atoms[idx].tag = label
                            else:
                                atoms[idx].tag = str(atoms[idx].tag) + label
                            if atoms[idx].symbol not in \
                            species_pseudo_mapping[0]+species_pseudo_mapping[1]:
                                if atoms[idx].symbol == mA:
                                    atoms[idx].symbol = \
                                    species_pseudo_mapping[k][0]
                                elif atoms[idx].symbol == mB:
                                    atoms[idx].symbol = \
                                    species_pseudo_mapping[k][1]
                            else:
                                if atoms[idx].symbol == \
                                   species_pseudo_mapping[k-1][0]:
                                    atoms[idx].symbol = \
                                    species_pseudo_mapping[2][0]
                                elif atoms[idx].symbol == \
                                     species_pseudo_mapping[k-1][1]:\
                                    atoms[idx].symbol = \
                                    species_pseudo_mapping[2][1]
                        n_occupied_sites += 1 
        else:
            raise NotImplementedError
    else:
        for site in sites:
            if is_site_occupied(atoms, site, min_adsorbate_distance=0.5):
                site['occupied'] = 1
                indices = site['indices']
                label = site['label']
                for idx in indices:                
                    if atoms[idx].tag == 0:
                        atoms[idx].tag = label
                    else:
                        atoms[idx].tag = str(atoms[idx].tag) + label
                    # Map to pseudo elements even when there is only one 
                    # adsorbate species (unnecessary)
                    #if atoms[idx].symbol == mA:
                    #    atoms[idx].symbol = species_pseudo_mapping[0][0]
                    #elif atoms[idx].symbol == mB:
                    #    atoms[idx].symbol = species_pseudo_mapping[0][1]
                n_occupied_sites += 1
    tag_set = set([a.tag for a in atoms])
    print('{0} sites labeled with tags including \
           {1}'.format(n_occupied_sites, tag_set))

    return atoms


def multi_label_counter(atoms, adsorbate, show_subsurface=False):
    '''Encoding the labels into 5d numpy arrays. 
    This can be further used as a fingerprint.

    Atoms that constitute an occupied adsorption site will be labeled as 1.
    If an atom contributes to multiple sites of same type, the number wil 
    increase. One atom can encompass multiple non-zero values if it 
    contributes to multiple types of sites.

    Note: Please provide atoms including adsorbate(s), with adsorbate being a 
    string or a list of strings.

    Set show_subsurface=True if you also want to label the second shell atoms.'''

    labeled_atoms = label_occupied_sites(atoms, adsorbate, show_subsurface)
    np_indices = [a.index for a in labeled_atoms if a.symbol not in ads_elements]
    np_atoms = labeled_atoms[np_indices]
    
    counter_lst = []
    for atom in np_atoms:
        if atom.symbol not in ads_elements:
            if atom.tag == 0:
                counter_lst.append(np.zeros(5).astype(int).tolist())
            else:
                line = str(atom.tag)
                cns = [int(s) for s in line]
                lst = np.zeros(5).astype(int).tolist()
                for idx in cns:
                    lst[idx-1] += int(1)
                counter_lst.append(lst)

    return counter_lst
