from .utilities import * 
from ase.data import reference_states, atomic_numbers, covalent_radii
from ase.optimize import BFGS, FIRE
from ase.io import read, write
from ase import Atom, Atoms
from asap3.analysis.rdf import RadialDistributionFunction
from asap3 import FullNeighborList
from asap3.analysis import FullCNA, PTM
from asap3 import EMT as asapEMT
from collections import defaultdict
from collections import Iterable, Counter
from itertools import combinations, groupby
import numpy as np
import networkx as nx
import warnings
import random
import scipy
import math
import re


icosa_dict = {
    # Triangle sites on outermost shell -- Icosa, Cubocta, Deca, Tocta
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': [str({(3, 1, 1): 6, (4, 2, 1): 3})],
    # Edge sites on outermost shell -- Icosa
    str({(3, 1, 1): 4, (3, 2, 2): 2, (4, 2, 2): 2}): 'edge',
    'edge': [str({(3, 1, 1): 4, (3, 2, 2): 2, (4, 2, 2): 2})],
    # Vertice sites on outermost shell -- Icosa, Deca
    str({(3, 2, 2): 5, (5, 5, 5): 1}): 'vertex',
    'vertex': [str({(3, 2, 2): 5, (5, 5, 5): 1})],
}

ticosa_dict = {
    # Triangle sites on outermost shell -- Icosa, Cubocta, Deca, Tocta
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': [str({(3, 1, 1): 6, (4, 2, 1): 3})],
    # Edge sites on outermost shell -- Icosa
    str({(3, 1, 1): 4, (3, 2, 2): 2, (4, 2, 2): 2}): 'edge',
    'edge': [str({(3, 1, 1): 4, (3, 2, 2): 2, (4, 2, 2): 2})],
    # Vertice sites on outermost shell -- Icosa, Deca
    str({(3, 2, 2): 5, (5, 5, 5): 1}): 'vertex',
    'vertex': [str({(3, 2, 2): 5, (5, 5, 5): 1}), 
               str({(2, 0, 0): 2, (3, 0, 0): 1, (3, 1, 1): 2, (3, 2, 2): 1, (4, 2, 2): 1})],
    # Vertice sites B on outermost shell -- Ticosa, Deca
    str({(2, 0, 0): 2, (3, 0, 0): 1, (3, 1, 1): 2, (3, 2, 2): 1, (4, 2, 2): 1}): 'vertex',
}

cubocta_dict = {
    # Edge sites on outermost shell -- Cubocta, Tocta
    str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}): 'edge',
    'edge': [str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2})],
    # Square sites on outermost shell -- Cubocta, Deca, Tocta, (Surface)
    str({(2, 1, 1): 4, (4, 2, 1): 4}): 'fcc100',
    'fcc100': [str({(2, 1, 1): 4, (4, 2, 1): 4})],
    # Vertice sites on outermost shell -- Cubocta
    str({(2, 1, 1): 4, (4, 2, 1): 1}): 'vertex',
    'vertex': [str({(2, 1, 1): 4, (4, 2, 1): 1})],
    # Triangle sites on outermost shell -- Icosa, Cubocta, Deca, Tocta, (Surface)
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': [str({(3, 1, 1): 6, (4, 2, 1): 3})],
}

mdeca_dict = {
    # Edge sites (111)-(111) on outermost shell -- Deca
    str({(3, 1, 1): 4, (3, 2, 2): 2, (4, 2, 2): 2}): 'edge',
    'edge': [str({(3, 1, 1): 4, (3, 2, 2): 2, (4, 2, 2): 2}), 
             str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}), 
             str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1})],
    # Edge sites (111)-(100) on outermost shell -- Deca
    str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}): 'edge',
    # Edge sites (111)-(111)notch on outermost shell -- Deca
    str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}): 'edge',
    # Square sites on outermost shell -- Cubocta, Deca, Tocta
    str({(2, 1, 1): 4, (4, 2, 1): 4}): 'fcc100',
    'fcc100': [str({(2, 1, 1): 4, (4, 2, 1): 4})],
    # Vertice sites on outermost shell -- Icosa, Deca
    str({(3, 2, 2): 5, (5, 5, 5): 1}): 'vertex',
    'vertex': [str({(3, 2, 2): 5, (5, 5, 5): 1}), 
               str({(2, 0, 0): 1, (2, 1, 1): 2, (3, 1, 1): 2, (4, 2, 1): 1}), 
               str({(2, 0, 0): 2, (3, 0, 0): 1, (3, 1, 1): 2, (3, 2, 2): 1, (4, 2, 2): 1})],
    # Vertice sites A on outermost shell -- Mdeca
    str({(2, 0, 0): 1, (2, 1, 1): 2, (3, 1, 1): 2, (4, 2, 1): 1}): 'vertex',
    # Vertice sites B on outermost shell -- Mdeca
    str({(2, 0, 0): 2, (3, 0, 0): 1, (3, 1, 1): 2, (3, 2, 2): 1, (4, 2, 2): 1}): 'vertex',
    # Triangle (pentagon) sites on outermost shell -- Icosa, Cubocta, Deca, Tocta
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': [str({(3, 1, 1): 6, (4, 2, 1): 3}), 
               str({(3, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 2, (4, 2, 2): 2})],
    # Triangle (pentagon) notch sites on outermost shell -- Deca
    str({(3, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 2, (4, 2, 2): 2}): 'fcc111',
}

octa_dict = {
    # Edge sites (111)-(111) on outermost shell -- Octa, Tocta
    str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}): 'edge',
    'edge': [str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1})],
    # Vertice sites on outermost shell -- Octa
    str({(2, 0, 0): 4}): 'vertex',
    'vertex': [str({(2, 0, 0): 4})],
    # Triangle (pentagon) sites on outermost shell -- Icosa, Cubocta, Deca, Octa
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': [str({(3, 1, 1): 6, (4, 2, 1): 3})],
}

tocta_dict = {
    # Edge sites on outermost shell -- Cubocta, Tocta
    str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}): 'edge',
    'edge': [str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}), 
             str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1})],
    # Edge sites (111)-(111) on outermost shell -- Octa
    str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}): 'edge',
    # Square sites on outermost shell -- Cubocta, Deca, Tocta
    str({(2, 1, 1): 4, (4, 2, 1): 4}): 'fcc100',
    'fcc100': [str({(2, 1, 1): 4, (4, 2, 1): 4})],
    # Vertice sites on outermost shell -- Tocta
    str({(2, 0, 0): 1, (2, 1, 1): 2, (3, 1, 1): 2, (4, 2, 1): 1}): 'vertex',
    'vertex': [str({(2, 0, 0): 1, (2, 1, 1): 2, (3, 1, 1): 2, (4, 2, 1): 1})],
    # Triangle (pentagon) sites on outermost shell -- Icosa, Cubocta, Deca, Octa
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': [str({(3, 1, 1): 6, (4, 2, 1): 3})],
}

surf_dict = {
    # Square sites
    str({(2, 1, 1): 4, (4, 2, 1): 4}): 'fcc100',
    'fcc100': [str({(2, 1, 1): 4, (4, 2, 1): 4})],
    # Triangle sites
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': [str({(3, 1, 1): 6, (4, 2, 1): 3}),
    str({(2, 0, 0): 2, (3, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}),
    str({(2, 0, 0): 2, (3, 1, 1): 6})],
    # Triangle sites
    str({(2, 0, 0): 2, (3, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}): 'fcc111',
    # Triangle sites
    str({(2, 0, 0): 2, (3, 1, 1): 6}): 'fcc111',
    # Triangle-Triangle sites
    str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}): 'fcc110',
    'fcc110': [str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}), 
               str({(3, 1, 1): 4, (4, 2, 1): 7})],
    str({(3, 1, 1): 4, (4, 2, 1): 7}): 'fcc110',
    # Triangle-Triangle-Square sites
    str({(3, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 2, (4, 2, 2): 2}): 'fcc211',
    'fcc211': [str({(3, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 2, (4, 2, 2): 2}),
               str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}),
               str({(2, 1, 1): 1, (3, 1, 1): 2, (3, 2, 2): 2, (4, 2, 1): 2, (4, 2, 2): 2}),
               str({(3, 1, 1): 6, (4, 2, 1): 3})],
    str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}): 'fcc211',
    str({(2, 1, 1): 1, (3, 1, 1): 2, (3, 2, 2): 2, (4, 2, 1): 2, (4, 2, 2): 2}): 'fcc211',
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc211',
    # Triangle-Square sites
    str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}): 'fcc311',
    'fcc311': [str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}), 
               str({(2, 1, 1): 1, (3, 1, 1): 4, (4, 2, 1): 5})],
    str({(2, 1, 1): 1, (3, 1, 1): 4, (4, 2, 1): 5}): 'fcc311',
}


# Set global variables
adsorbate_elements = 'SCHON'
heights_dict = {'ontop': 2., 
                'bridge': 2.,
                'shortbridge': 2.,
                'longbridge': 2.,
                'fcc': 2., 
                'hcp': 2., 
                '3fold': 2.,
                '4fold': 2.,
                '5fold': 2.,
                '6fold': 0.}

warnings.filterwarnings("ignore")


class NanoparticleAdsorptionSites(object):

    def __init__(self, atoms, allow_6fold=False, 
                 composition_effect=False, subsurf_effect=False):

        assert True not in atoms.pbc
        atoms = atoms.copy()
        del atoms[[a.index for a in atoms if 'a' not in reference_states[a.number]]]
        del atoms[[a.index for a in atoms if a.symbol in adsorbate_elements]]
        self.atoms = atoms
        self.positions = atoms.positions
        self.symbols = atoms.symbols
        self.numbers = atoms.numbers
        self.allow_6fold = allow_6fold
        self.composition_effect = composition_effect
        self.subsurf_effect = subsurf_effect
        self.cell = atoms.cell
        self.pbc = atoms.pbc
        self.metals = sorted(list(set(atoms.symbols)), 
                             key=lambda x: atomic_numbers[x])

        self.fullCNA = {}
        self.make_fullCNA()
        self.set_first_neighbor_distance_from_rdf()
        self.site_dict = self.get_site_dict()
        self.make_neighbor_list()
        self.surf_ids, self.surf_sites = self.get_surface_sites()

        self.site_list = []
        self.populate_site_list()
        
    def populate_site_list(self):
        """Find all ontop, bridge and hollow sites (3-fold and 4-fold) 
           given an input nanoparticle and collect in a site list. 
        """

        ss = self.surf_sites
        ssall = set(ss['all'])
        fcna = self.get_fullCNA()
        sl = self.site_list
        usi = set()  # used_site_indices
        normals_for_site = dict(list(zip(ssall, [[] for _ in ssall])))
        for surface, sites in ss.items():
            if surface == 'all':
                continue
            for s in sites:
                neighbors, _, dist2 = self.nblist.get_neighbors(s, self.r + 0.2)
                for n in neighbors:
                    si = tuple(sorted([s, n]))  # site_indices
                    if n in ssall and si not in usi:
                        # bridge sites
                        pos = np.average(self.positions[[n, s]], 0)
                        site_surf = self.get_surface_designation([s, n])
                        site = self.new_site()
                        site.update({'site': 'bridge',
                                     'surface': site_surf,
                                     'position': pos,
                                     'indices': si})
                        if self.composition_effect:                         
                            symbols = [(self.symbols[i], self.numbers[i]) 
                                       for i in si]
                            comp = sorted(symbols, key=lambda x: x[1])
                            composition = ''.join([c[0] for c in comp])
                            site.update({'composition': composition})
                        sl.append(site)
                        usi.add(si)
                # Find normal
                for n, m in combinations(neighbors, 2):
                    si = tuple(sorted([s, n, m]))
                    if n in ssall and m in ssall and si not in usi:
                        angle = self.get_angle([s, n, m])
                        if self.is_eq(angle, np.pi/3.):
                            # 3-fold (fcc or hcp) site
                            normal = self.get_surface_normal([s, n, m])
                            for i in [s, n, m]:
                                normals_for_site[i].append(normal)
                            pos = np.average(self.positions[[n, m, s]], 0)
                            new_pos = pos - normal * self.r * (2./3)**(.5)
                            if self.no_atom_too_close_to_pos(new_pos, 0.5):
                                this_site = 'fcc'
                            else:
                                this_site = 'hcp'
                            site_surf = 'fcc111'

                            site = self.new_site()
                            site.update({'site': this_site,
                                         'surface': site_surf,
                                         'position': pos,
                                         'normal': normal,
                                         'indices': si})
                            if self.composition_effect:                       
                                metals = self.metals
                                if len(metals) == 1:
                                    composition = 3*metals[0]
                                else:
                                    ma, mb = metals[0], metals[1]
                                    symbols = [self.symbols[i] for i in si]
                                    nma = symbols.count(ma)
                                    if nma == 0:
                                        composition = 3*mb
                                    elif nma == 1:
                                        composition = ma + 2*mb
                                    elif nma == 2:
                                        composition = 2*ma + mb
                                    elif nma == 3:
                                        composition = 3*ma
                                site.update({'composition': composition})   

                            if this_site == 'hcp' and self.subsurf_effect:
                                hcp_nbrs = []
                                for i in si:
                                    hcp_nbrs += list(self.nblist.get_neighbors(i, 
                                                     self.r + 1.)[0])
                                isub = [key for key, count in Counter(
                                        hcp_nbrs).items() if count == 3][0]
                                site.update({'subsurf_index': isub})
                                if self.composition_effect:
                                    site.update({'subsurf_element': 
                                                 self.symbols[isub]})
                            sl.append(site)
                            usi.add(si)

                        elif self.is_eq(angle, np.pi/2.):
                            # 4-fold hollow site
                            site_surf = 'fcc100'
                            l2 = self.r * math.sqrt(2) + 0.2
                            nebs2, _, _ = self.nblist.get_neighbors(s, l2)
                            nebs2 = [k for k in nebs2 if k not in neighbors]
                            for k in nebs2:
                                si = tuple(sorted([s, n, m, k]))
                                if k in ssall and si not in usi:
                                    d1 = self.atoms.get_distance(n, k)
                                    if self.is_eq(d1, self.r, 0.2):
                                        d2 = self.atoms.get_distance(m, k)
                                        if self.is_eq(d2, self.r, 0.2):
                                            # 4-fold hollow site found
                                            normal = self.get_surface_normal([s, n, m])
                                            # Save the normals now and add them to the site later
                                            for i in [s, n, m, k]:
                                                normals_for_site[i].append(normal)
                                            ps = self.positions[[n, m, s, k]]
                                            pos = np.average(ps, 0)
 
                                            site = self.new_site()
                                            site.update({'site': '4fold',
                                                         'surface': site_surf,
                                                         'position': pos,
                                                         'normal': normal,
                                                         'indices': si})
                                            if self.composition_effect:
                                                metals = self.metals 
                                                if len(metals) == 1:
                                                    composition = 4*metals[0]
                                                else:
                                                    ma, mb = metals[0], metals[1]
                                                    symbols = [self.symbols[i] for i in si]
                                                    nma = symbols.count(ma)
                                                    if nma == 0:
                                                        composition = 4*mb
                                                    elif nma == 1:
                                                        composition = ma + 3*mb
                                                    elif nma == 2:
                                                        opp = max(list(si[1:]), key=lambda x: 
                                                              np.linalg.norm(self.positions[x]
                                                              - self.positions[si[0]])) 
                                                        if self.symbols[opp] == self.symbols[si[0]]:
                                                            composition = ma + mb + ma + mb 
                                                        else:
                                                            composition = 2*ma + 2*mb 
                                                    elif nma == 3:
                                                        composition = 3*ma + mb
                                                    elif nma == 4:
                                                        composition = 4*ma
                                                site.update({'composition': composition})    
                                            if self.subsurf_effect:                        
                                                fold4_nbrs = []
                                                for i in si:
                                                    fold4_nbrs += list(self.nblist.get_neighbors(
                                                                       i, self.r + 1.)[0])
                                                isub = [key for key, count in Counter(
                                                        fold4_nbrs).items() if count == 4][0]
                                                site.update({'subsurf_index': isub})
                                                if self.composition_effect:
                                                    site.update({'subsurf_element': 
                                                                 self.symbols[isub]})
                                            sl.append(site)
                                            usi.add(si)

                # ontop sites
                site = self.new_site()
                site.update({'site': 'ontop', 
                             'surface': surface,
                             'position': self.positions[s],
                             'indices': (s,)})
                if self.composition_effect:
                    site.update({'composition': self.symbols[s]})
                sl.append(site)
                usi.add((s))

        # Add 6-fold sites if allowed
        if self.allow_6fold:
            dh = 5/6 * self.r
            subsurf_ids = self.get_subsurface()
        for t in sl:
            # Add normals to ontop sites
            if t['site'] == 'ontop':
                n = np.average(normals_for_site[t['indices'][0]], 0)
                t['normal'] = n / np.linalg.norm(n)
            # Add normals to bridge sites
            elif t['site'] == 'bridge':
                normals = []
                for i in t['indices']:
                    normals.extend(normals_for_site[i])
                n = np.average(normals, 0)
                t['normal'] = n / np.linalg.norm(n)
            # Add subsurf sites
            if self.allow_6fold:
                if t['site'] == 'fcc':  
                    site = t.copy()
                    subpos = t['position'] - t['normal'] * dh                                   
                    def get_squared_distance(x):
                        return np.sum((self.positions[x] - subpos)**2)
                    subsi = sorted(sorted(subsurf_ids, key=get_squared_distance)[:3])     
                    si = site['indices']
                    site.update({'site': '6fold',      
                                 'position': subpos,
                                 'indices': tuple(sorted(si+tuple(subsi)))})      
                    if self.composition_effect:
                        metals = self.metals
                        comp = site['composition']
                        if len(metals) == 1:
                            composition = ''.join([comp, 3*metals[0]])
                        else: 
                            ma, mb = metals[0], metals[1]
                            subsyms = [self.symbols[subi] for subi in subsi]
                            nma = subsyms.count(ma)
                            if nma == 0:
                                composition = ''.join([comp, 3*mb])
                            elif nma == 1:
                                ia = subsi[subsyms.index(ma)]
                                subpos = self.positions[ia]
                                if self.symbols[max(si, key=get_squared_distance)] == ma:
                                    composition = ''.join([comp, 2*mb + ma])
                                else:
                                    composition = ''.join([comp, ma + 2*mb])
                            elif nma == 2:
                                ib = subsi[subsyms.index(mb)]
                                subpos = self.positions[ib]
                                if self.symbols[max(si, key=get_squared_distance)] == mb:
                                    composition = ''.join([comp, 2*ma + mb])
                                else:
                                    composition = ''.join([comp, mb + 2*ma])
                            elif nma == 3:
                                composition = ''.join([comp, 3*ma])
                        site.update({'composition': composition})
                    sl.append(site)
                    usi.add(si) 

    def new_site(self):
        return {'site': None, 'surface': None, 'position': None, 
                'normal': None, 'indices': None, 'composition': None,
                'subsurf_index': None, 'subsurf_element': None}

    def get_site(self, indices):
        if not isinstance(indices, Iterable):
            indices = [indices]
        indices = tuple(sorted(indices))
        st = next((s for s in self.site_list if 
                   s['indices'] == indices), None)
        return st 

    def get_all_sites_of_type(self, type):            
        return [i for i in self.site_list
                if i['site'] == type]

    def get_all_fcc_sites(self):
        return self.get_all_sites_of_type('fcc')

    def get_all_from_surface(self, surface):
        return [i for i in self.site_list
                if i['surface'] == surface]

    def get_sites_from_surface(self, site, surface):
        surf = self.get_all_from_surface(surface)
        return [i for i in surf if i['site'] == site]

    def get_two_vectors(self, sites):
        p1 = self.positions[sites[1]]
        p2 = self.positions[sites[2]]
        vec1 = p1 - self.positions[sites[0]]
        vec2 = p2 - self.positions[sites[0]]
        return vec1, vec2

    def is_eq(self, v1, v2, eps=0.1):
        if abs(v1 - v2) < eps:
            return True
        else:
            return False

    def get_surface_normal(self, sites):
        vec1, vec2 = self.get_two_vectors(sites)
        n = np.cross(vec1, vec2)
        l = math.sqrt(n @ n.conj())
        new_pos = self.positions[sites[0]] + self.r * n / l
        # Add support for having adsorbates on the particles already
        # by putting in elements to check for in the function below
        j = 2 * int(self.no_atom_too_close_to_pos(new_pos, (5./6)*self.r)) - 1
        return j * n / l

    def get_angle(self, sites):
        vec1, vec2 = self.get_two_vectors(sites)
        p = (vec1 @ vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec1))
        return np.arccos(np.clip(p, -1,1))

    def no_atom_too_close_to_pos(self, pos, mindist):              
        """Returns True if no atoms are closer than mindist to pos,
        otherwise False."""
        dists = [np.linalg.norm(atom.position - pos) > mindist
                 for atom in self.atoms]
        return all(dists)                                                       

    def get_surface_sites(self): 
        """
        Returns a dictionary with all the surface designations
        """
        surface_sites = {'all': [],
                         'fcc111': [],
                         'fcc100': [],
                         'edge': [],
                         'vertex': [], }
        fcna = self.get_fullCNA()
        site_dict = self.site_dict
        ptmdata = PTM(self.atoms, rmsd_max=0.25)
        surface_ids = list(np.where(ptmdata['structure'] == 0)[0])

        for i in surface_ids:
#            if i in [284, 310]:
#                print(fcna[i])
            surface_sites['all'].append(i)
            if str(fcna[i]) not in site_dict:
                # The structure is distorted from the original, giving
                # a larger cutoff should overcome this problem
                r = self.r + 0.6
                fcna = self.get_fullCNA(rCut=r)
            if str(fcna[i]) not in site_dict:
                # If this does not solve the problem we probably have a
                # reconstruction of the surface and will leave this
                # atom unused this time
                continue
            surface_sites[site_dict[str(fcna[i])]].append(i)
        return surface_ids, surface_sites  

    def get_subsurface(self):
        notsurf = [a.index for a in self.atoms if a.index not in self.surf_ids]
        ptmdata = PTM(self.atoms[notsurf], rmsd_max=0.25)
        return list(np.where(ptmdata['structure'] == 0)[0])

    def make_fullCNA(self, rCut=None):                  
        if rCut not in self.fullCNA:
            self.fullCNA[rCut] = FullCNA(self.atoms, 
                                 rCut=rCut).get_normal_cna()

    def get_fullCNA(self, rCut=None):
        if rCut not in self.fullCNA:
            self.make_fullCNA(rCut=rCut)
        return self.fullCNA[rCut]

    def make_neighbor_list(self, rMax=10.):
        self.nblist = FullNeighborList(rCut=rMax, atoms=self.atoms)

    def get_connectivity(self):                                      
        """Generate a connections matrix from neighbor_shell_list."""
        nbslist = neighbor_shell_list(self.atoms, 0.3, neighbor_number=1)
        return get_connectivity_matrix(nbslist)                  

    def get_site_dict(self):
        fcna = self.get_fullCNA()
        icosa_weight = cubocta_weight = mdeca_weight = tocta_weight = 0
        for s in fcna:
            if str(s) in icosa_dict:
                icosa_weight += 1
            if str(s) in cubocta_dict:
                cubocta_weight += 1
            if str(s) in mdeca_dict:
                mdeca_weight += 1
            if str(s) in tocta_dict:
                tocta_weight += 1
        full_weights = [icosa_weight, cubocta_weight, 
                        mdeca_weight, tocta_weight]
        if icosa_weight == max(full_weights):
            return icosa_dict
        elif cubocta_weight == max(full_weights):
            return cubocta_dict
        elif tocta_weight == max(full_weights):
            return tocta_dict
        else:
            return mdeca_dict

    def set_first_neighbor_distance_from_rdf(self, rMax=10, nBins=200):
        atoms = self.atoms.copy()
        for j, L in enumerate(list(atoms.cell.diagonal())):
            if L <= 10:
                atoms.cell[j][j] = 12 
        rdf = RadialDistributionFunction(atoms, rMax, nBins).get_rdf()
        x = (np.arange(nBins) + 0.5) * rMax / nBins
        rdf *= x**2
        diff_rdf = np.gradient(rdf)

        i = 0
        while diff_rdf[i] >= 0:
            i += 1
        self.r = x[i]

    def get_surface_designation(self, sites):                                 
        fcna = self.get_fullCNA()
        sd = self.site_dict
        if len(sites) == 1:
            s = sites[0]
            return sd[str(fcna[s])]
        elif len(sites) == 2:
            if str(fcna[sites[0]]) not in sd or str(fcna[sites[1]]) not in sd:
                return 'unknown'
            s0 = sd[str(fcna[sites[0]])]
            s1 = sd[str(fcna[sites[1]])]
            ss01 = tuple(sorted([s0, s1]))
            if ss01 in [('edge', 'edge'), 
                        ('edge', 'vertex'),
                        ('vertex', 'vertex')]:
                return 'edge'
            elif ss01 in [('fcc111', 'vertex'), 
                          ('edge', 'fcc111'), 
                          ('fcc111', 'fcc111')]:
                return 'fcc111'
            elif ss01 in [('fcc100', 'vertex'), 
                          ('edge', 'fcc100'), 
                          ('fcc100', 'fcc100')]:
                return 'fcc100'
        elif len(sites) == 3:
            return 'fcc111'

    def get_unique_sites(self, unique_composition=False,         
                         unique_subsurf=False):
        sl = self.site_list
        key_list = ['site', 'surface']
        if unique_composition:
            if not self.composition_effect:
                raise ValueError('The site list does not include ',
                                 'information of composition')
            key_list.append('composition')
            if unique_subsurf:
                if not self.subsurf_effect:
                    raise ValueError('The site list does not include ',
                                     'information of subsurface')
                key_list.append('subsurf_element') 
        else:
            if unique_subsurf:
                raise ValueError('To include the subsurface element, ',
                                 'unique_composition also need to ',
                                 'be set to True')    
        sklist = sorted([[s[k] for k in key_list] for s in sl])
 
        return sorted(list(sklist for sklist, _ in groupby(sklist)))

    def get_graph(self):                             
        cm = self.get_connectivity()
        G = nx.Graph()                               
        # Add edges from surface connectivity matrix
        rows, cols = np.where(cm == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G.add_edges_from(edges)
        return G

    def get_neighbor_site_list(self, neighbor_number=1, span=True):           
        """Returns the site_list index of all neighbor 
        shell sites for each site
        """

        sl = self.site_list
        adsposs = np.asarray([s['position'] + s['normal'] * \
                             heights_dict[s['site']] for s in sl])
        statoms = Atoms('X{}'.format(len(sl)), 
                        positions=adsposs, 
                        cell=self.cell, 
                        pbc=self.pbc)
        cr = 0.55 
        if neighbor_number == 1:
            cr += 0.1
                                                                   
        return neighbor_shell_list(statoms, 0.1, neighbor_number,
                                   mic=False, radius=cr, span=span)

    def update_positions(self, new_atoms):
        sl = self.site_list
        for st in sl:
            si = list(st['indices'])
            newpos = np.average(new_atoms.positions[si], 0) 
            st['position'] = newpos


class SlabAdsorptionSites(object):

    def __init__(self, atoms, surface, allow_6fold=False, 
                 composition_effect=False, subsurf_effect=False, dx=.5):
        """
        allow_6fold: allow 6-fold subsurf site

        subsurf_effect: include the effect of subsurf elements 
                        for hcp(h) and 4fold(t) sites
        """
        assert True in atoms.pbc    
        atoms = atoms.copy() 
        del atoms[[a.index for a in atoms if 'a' not in reference_states[a.number]]]
        del atoms[[a.index for a in atoms if a.symbol in adsorbate_elements]]
        self.atoms = atoms
        self.positions = atoms.positions 
        self.symbols = atoms.symbols
        self.numbers = atoms.numbers
        self.indices = [a.index for a in self.atoms] 
        self.surface = surface
        ref_atoms = self.atoms.copy() 
        for a in ref_atoms:
            a.symbol = 'Au'       
        ref_atoms.set_constraint()
        ref_atoms.calc = asapEMT()
        opt = FIRE(ref_atoms, logfile=None)
        opt.run(fmax=0.05)
        ref_atoms.calc = None

        self.ref_atoms = ref_atoms
        self.delta_positions = atoms.positions - ref_atoms.positions
        self.site_dict = surf_dict 
        self.cell = atoms.cell
        self.pbc = atoms.pbc
        self.metals = sorted(list(set(atoms.symbols)), 
                             key=lambda x: atomic_numbers[x])
        self.allow_6fold = allow_6fold
        self.composition_effect = composition_effect
        self.subsurf_effect = subsurf_effect
        self.dx = dx
        self.tol = 1e-5

        self.make_neighbor_list(dx=self.dx) 
        self.connectivity_matrix = self.get_connectivity()         
        self.surf_ids, self.subsurf_ids = self.get_termination()        

        #TODO: an efficient and consistent way to identify surfaces.
       # if surface is None: 
       #     self.fullCNA, self.surfCNA = self.get_CNA(rCut=3.)
       #     self.surface = self.identify_surface()
       #     print('The surface is identified as {}. '.format(self.surface),
       #           'Please specify the surface if it is incorrect')
       # else:
       #     self.surface = surface

        self.site_list = []
        self.populate_site_list()
        
    def populate_site_list(self, allow_obtuse=True, cutoff=5.):        
        """Find all ontop, bridge and hollow sites (3-fold and 4-fold) 
           given an input slab based on Delaunay triangulation of 
           surface atoms of a super-cell and collect in a site list. 
           (Adaption from Catkit)
        """
 
        top_indices = self.surf_ids
        sl = self.site_list
        normals_for_site = dict(list(zip(top_indices, 
                            [[] for _ in top_indices])))
        usi = set() # used_site_indices
        cm = self.connectivity_matrix 

        for s in top_indices:
            occurence = cm[s]
            sumo = np.sum(occurence, axis=0)
            if self.surface in ['fcc111','hcp0001']:
                geometry = 'h'
            elif self.surface in ['fcc100','bcc100']:
                geometry = 't'
            elif self.surface in ['bcc110']:
                geometry = 'o'
            else:
                if sumo in [6, 7, 8]:
                    geometry = 'step'
                elif sumo in [9, 11]:
                    geometry = 'terrace'
                elif sumo == 10:
                    if self.surface in ['fcc211','bcc111']:
                        geometry = 'lowerstep'
                    elif self.surface in ['fcc110','fcc311','hcp10m10']:
                        geometry = 'terrace'
                else:
                    print('Cannot identify site {}'.format(s))
                    continue
            si = (s,)
            site = self.new_site()
            site.update({'site': 'ontop',
                         'surface': self.surface,
                         'geometry': geometry,
                         'position': self.positions[s],
                         'indices': si})
            if self.surface == 'fcc110' and geometry == 'terrace':
                site.update({'extra': np.where(occurence==1)[0]})
            if self.composition_effect:
                site.update({'composition': self.symbols[s]})
            sl.append(site)
            usi.add(si)

        stepids = set([s['indices'][0] for s in sl if s['geometry'] == 'step'])
        terraceids = set([s['indices'][0] for s in sl if s['geometry'] == 'terrace']) 
        if self.surface in ['fcc211','bcc111']:
            lowerids = set([s['indices'][0] for s in sl if s['geometry'] == 'lowerstep'])
        if self.surface == 'bcc111':
            sorted_steps = sorted([i for i in stepids], key=lambda x: 
                                   self.ref_atoms.positions[x,2])
            for j, stpi in enumerate(sorted_steps):
                if j < len(sorted_steps) / 2:
                    stepids.remove(stpi)
                    terraceids.add(stpi) 

            for st in sl:
                if st['geometry'] == 'step' and st['indices'][0] in terraceids:
                    st['geometry'] = 'terrace'
        if self.surface == 'fcc110':   
            for st in sl:
                if 'extra' in st:
                    extra = st['extra']
                    extraids = [e for e in extra if e in stepids]
                    if len(extraids) != 4:
                        print('Cannot identify other 4 atoms for this 5-fold site')
                    if self.composition_effect:
                        metals = self.metals
                        if len(metals) == 1:
                            composition = 4*metals[0]
                        else: 
                            ma, mb = metals[0], metals[1]                       
                            symbols = [self.symbols[i] for i in extraids]
                            nma = symbols.count(ma)
                            if nma == 0:
                                composition = 4*mb
                            elif nma == 1:
                                composition = ma + 3*mb
                            elif nma == 2:
                                idd = sorted(extraids[1:], key=lambda x:    
                                      get_mic(self.positions[x], self.positions[extraids[0]],
                                              self.cell, return_squared_distance=True))
                                opp, close = idd[-1], idd[0]
                                if self.symbols[opp] == self.symbols[extraids[0]]:
                                    composition = ma + mb + ma + mb 
                                else:
                                    if self.symbols[close] == self.symbols[extraids[0]]:
                                        composition = 2*ma + 2*mb 
                                    else:
                                        composition = ma + 2*mb + ma
                            elif nma == 3:
                                composition = 3*ma + mb
                            elif nma == 4:
                                composition = 4*ma
                        st['composition'] += '-{}'.format(composition) 
                    st['indices'] = list(si)+ extraids
                    del st['extra']

        ext_index, ext_coords, _ = expand_cell(self.ref_atoms, cutoff)
        extended_top = np.where(np.in1d(ext_index, top_indices))[0]
        meansurfz = np.average(self.atoms.positions[self.surf_ids][:,2], 0)
        ext_all_coords = ext_coords[extended_top]
        surf_screen = np.where(abs(ext_all_coords[:,2] - meansurfz) < cutoff)
        ext_surf_coords = ext_all_coords[surf_screen]
        dt = scipy.spatial.Delaunay(ext_surf_coords[:,:2])
        neighbors = dt.neighbors
        simplices = dt.simplices

        bridge_positions, fold3_positions, fold4_positions = [], [], []
        bridge_points, fold3_points, fold4_points = [], [], []

        for i, corners in enumerate(simplices):
            cir = scipy.linalg.circulant(corners)
            edges = cir[:,1:]

            # Inner angle of each triangle corner
            vec = ext_surf_coords[edges.T] - ext_surf_coords[corners]
            uvec = vec.T / np.linalg.norm(vec, axis=2).T
            angles = np.sum(uvec.T[0] * uvec.T[1], axis=1)

            # Angle types
            right = np.isclose(angles, 0)
            obtuse = (angles < self.tol)
            rh_corner = corners[right]
            edge_neighbors = neighbors[i]

            if obtuse.any() and not allow_obtuse:
                # Assumption: All simplices with obtuse angles
                # are irrelevant boundaries.
                continue
            bridge = np.sum(ext_surf_coords[edges], axis=1) / 2.0

            # Looping through corners allows for elimination of
            # redundant points, identification of 4-fold hollows,
            # and collection of bridge neighbors.            
            for j, c in enumerate(corners):
                edge = sorted(edges[j])
                if edge in bridge_points:
                    continue

                # Get the bridge neighbors (for adsorption vector)
                neighbor_simplex = simplices[edge_neighbors[j]]
                oc = list(set(neighbor_simplex) - set(edge))[0]

                # Right angles potentially indicate 4-fold hollow
                potential_hollow = edge + sorted([c, oc])
                if c in rh_corner:
                    if potential_hollow in fold4_points:
                        continue

                    # Assumption: If not 4-fold, this suggests
                    # no hollow OR bridge site is present.
                    ovec = ext_surf_coords[edge] - ext_surf_coords[oc]
                    ouvec = ovec / np.linalg.norm(ovec)
                    oangle = np.dot(*ouvec)
                    oright = np.isclose(oangle, 0)
                    if oright:
                        fold4_points.append(potential_hollow)
                        fold4_positions.append(bridge[j])
                else:
                    bridge_points.append(edge)
                    bridge_positions.append(bridge[j])

            if not right.any() and not obtuse.any():
                fold3_position = np.average(ext_surf_coords[corners], axis=0)
                fold3_points += corners.tolist()
                fold3_positions.append(fold3_position)
        self.bridge_positions, self.fold3_positions, self.fold4_positions = bridge_positions, fold3_positions, fold4_positions
        # Complete information of each site
        for n, poss in enumerate([bridge_positions,fold3_positions,fold4_positions]):
            if not poss:
                continue
            fracs = np.stack(poss, axis=0) @ np.linalg.pinv(self.cell)
            xfracs, yfracs = fracs[:,0], fracs[:,1]

            # Take only the positions within the periodic boundary
            screen = np.where((xfracs > 0-self.tol) & (xfracs < 1-self.tol) & \
                              (yfracs > 0-self.tol) & (yfracs < 1-self.tol))[0]
            reduced_poss = np.asarray(poss)[screen]

            # Sort the index list of surface and subsurface atoms 
            # so that we can retrive original indices later
            sorted_indices = sorted(self.surf_ids + self.subsurf_ids)
            top2atoms = self.ref_atoms[sorted_indices]

            # Make a new neighborlist including sites as dummy atoms
            dummies = Atoms('X{}'.format(reduced_poss.shape[0]), 
                            positions=reduced_poss, 
                            cell=self.cell, 
                            pbc=self.pbc)
            ntop2 = len(top2atoms)
            testatoms = top2atoms + dummies
            nblist = neighbor_shell_list(testatoms, dx=self.dx, neighbor_number=1, 
                                         different_species=True, mic=True) 
            # Make bridge sites  
            if n == 0:
                fold4_poss = []
                for i, refpos in enumerate(reduced_poss):
                    bridge_indices = nblist[ntop2+i]                     
                    bridgeids = [sorted_indices[j] for j in bridge_indices]
                    if len(bridgeids) != 2: 
                        if self.surface in ['fcc100','fcc211','fcc311',
                        'bcc100','hcp10m11']:
                            fold4_poss.append(refpos)
                        else:
                            'Cannot identify site {}'.format(bridgeids)
                        continue                    
                    si = tuple(sorted(bridgeids))
                    pos = refpos + np.average(self.delta_positions[bridgeids], 0) 
                    occurence = np.sum(cm[bridgeids], axis=0)
                    siset = set(si)
                    nstep = len(stepids.intersection(siset))
                    nterrace = len(terraceids.intersection(siset))
                    if self.surface == 'fcc110':
                        sitetype = 'bridge'              
                        if nstep == 2:
                            geometry = 'step'
                        elif nstep == 1 and nterrace == 1:
                            geometry = 'h'
                        elif nterrace == 2:                            
                            geometry = 'terrace'
                        else:
                            print('Cannot identify site {}'.format(si)) 
                            continue 
                    elif self.surface == 'fcc311':
                        sitetype = 'bridge'
                        if nstep == 2:
                            geometry = 'step'
                        elif nstep == 1 and nterrace == 1:
                            cto2 = list(occurence).count(2)
                            if cto2 == 2:
                                geometry = 't'
                            elif cto2 == 3:
                                geometry = 'h'
                            else:
                                print('Cannot identify site {}'.format(si))
                        elif nterrace == 2:
                            geometry = 'terrace'
                        else:
                            print('Cannot identify site {}'.format(si)) 
                            continue          
                    elif self.surface == 'fcc211':
                        sitetype = 'bridge'
                        nlower = len(lowerids.intersection(siset))
                        if nstep == 2:
                            geometry = 'step'
                        elif nstep == 1 and nterrace == 1:
                            geometry = 'upperh'
                        elif nstep == 1 and nlower == 1:
                            geometry = 't'
                        # nterrace == 2 is actually terrace bridge, 
                        # but equivalent to lower111 for fcc211
                        elif nterrace == 2 or (nterrace == 1 and nlower == 1):
                            geometry = 'lowerh'
                        elif nlower == 2:
                            geometry = 'lowerstep'
                        else:
                            print('Cannot identify site {}'.format(si)) 
                            continue         
                    elif self.surface == 'bcc110':
                        geometry = 'o'
                        cto2 = list(occurence[self.surf_ids]).count(2)
                        if cto2 == 2:
                            sitetype = 'longbridge'
                        elif cto2 == 3:
                            sitetype = 'shortbridge'
                        else:
                            print('Cannot identify site {}'.format(si))
                            continue
                    elif self.surface == 'bcc111':
                        nlower = len(lowerids.intersection(siset))
                        if nstep == 1 and nlower == 1:
                            sitetype, geometry = 'longbridge', 'o'
                        elif nstep == 1 and nterrace == 1:
                            sitetype, geometry = 'shortbridge', 'uppero'
                        elif nterrace == 1 and nlower == 1:
                            sitetype, geometry = 'shortbridge', 'lowero'
                        else:
                            print('Cannot identify site {}'.format(si))
                            continue
                    elif self.surface == 'hcp10m10':
                        sitetype = 'bridge'
                        if nstep == 2:
                            geometry = 'step'
                        elif nstep == 1 and nterrace == 1:
                            geometry = 'o'
                        elif nterrace == 2:
                            geometry = 'terrace'
                        else:
                            print('Cannot identify site {}'.format(si))
                            continue
                    elif self.surface == 'hcp10m11':
                        sitetype = 'bridge'
                        if nstep == 2:
                            geometry = 'step'
                        elif nstep == 1 and nterrace == 1:
                            cto2 = list(occurence).count(2) 
                            if cto2 == 2:
                                geometry = 'subsurf'
                                isubs = [self.subsurf_ids[i] for i in np.where(
                                         occurence[self.subsurf_ids] == 2)[0]]
                                subpos = self.positions[isubs[0]] + .5 * get_mic(
                                         self.positions[isubs[0]], self.positions[
                                         isubs[1]], self.cell)
                            elif cto2 == 3:
                                geometry = 'h'
                            else:
                                print('Cannot identify site {}'.format(si))
                                continue
                        elif nterrace == 2:
                            geometry = 'terrace'
                        else:
                            print('Cannot identify site {}'.format(si))
                            continue  
                    elif self.surface in ['fcc111','hcp0001']:
                        sitetype, geometry = 'bridge', 'h'
                    elif self.surface in ['fcc100','bcc100']:
                        sitetype, geometry = 'bridge', 't'
                    
                    site = self.new_site()
                    special = False
                    if self.surface == 'fcc110' and geometry == 'terrace':
                        special = True
                        extraids = [xi for xi in np.where(occurence==2)[0]
                                        if xi in self.surf_ids]
                        site.update({'site': sitetype,
                                     'surface': self.surface,
                                     'geometry': geometry,
                                     'position': pos,
                                     'indices': bridgeids + extraids})
                    elif self.surface == 'hcp10m11' and geometry == 'subsurf': 
                        special = True
                        extraids = si
                        site.update({'site': sitetype,
                                     'surface': self.surface,
                                     'geometry': geometry,
                                     'position': subpos,
                                     'indices': bridgeids + isubs})
                    else:                         
                        site.update({'site': sitetype,               
                                     'surface': self.surface,
                                     'geometry': geometry,
                                     'position': pos,
                                     'indices': si})           
                    if self.composition_effect:                          
                        symbols = [(self.symbols[j], self.numbers[j]) for j in si]
                        comp = sorted(symbols, key=lambda x: x[1])
                        composition = ''.join([c[0] for c in comp])
                        if special:
                            extrasymbols = [(self.symbols[xj], self.numbers[xj]) 
                                             for xj in extraids]
                            extra = sorted(extrasymbols, key=lambda x: x[1])
                            extracomp = ''.join([e[0] for e in extra])
                            composition += '-{}'.format(extracomp)
                        site.update({'composition': composition})
                    sl.append(site)
                    usi.add(si)

                if self.surface in ['fcc100','fcc211','fcc311','bcc100',
                'hcp10m10','hcp10m11'] and fold4_poss:
                    fold4atoms = Atoms('X{}'.format(len(fold4_poss)), 
                                       positions=np.asarray(fold4_poss),
                                       cell=self.cell, pbc=self.pbc)
                    sorted_top = self.surf_ids
                    ntop = len(sorted_top)
                    topatoms = self.ref_atoms[sorted_top] 
                    newatoms = topatoms + fold4atoms
                    newnblist = neighbor_shell_list(newatoms, dx=.1, 
                                                    neighbor_number=2,
                                                    different_species=True, 
                                                    mic=True) 
                    
                    # Make 4-fold hollow sites
                    for i, refpos in enumerate(fold4_poss):
                        fold4_indices = newnblist[ntop+i]                     
                        fold4ids = [sorted_top[j] for j in fold4_indices]
                        occurence = np.sum(cm[fold4ids], axis=0)
                        isub = np.where(occurence >= 4)[0][0] 
                        si = tuple(sorted(fold4ids)) 
                        pos = refpos + np.average(
                              self.delta_positions[fold4ids], 0)
                        normal = self.get_surface_normal(
                                 [si[0], si[1], si[2]])
                        for idx in si:
                            normals_for_site[idx].append(normal)

                        site = self.new_site() 
                        if self.surface in ['hcp10m10','hcp10m11']:
                            sitetype = '5fold'
                            site.update({'site': sitetype,
                                         'surface': self.surface,
                                         'geometry': 'subsurf',
                                         'position': self.positions[isub],
                                         'normal': normal,
                                         'indices': tuple(sorted(fold4ids+[isub]))})
                        else:
                            sitetype = '4fold'
                            site.update({'site': sitetype,               
                                         'surface': self.surface,
                                         'geometry': 't',
                                         'position': pos,
                                         'normal': normal,
                                         'indices': si})                     
                        if self.composition_effect:                        
                            metals = self.metals
                            if len(metals) == 1:
                                composition = 4*metals[0]
                            else:
                                ma, mb = metals[0], metals[1]
                                symbols = [self.symbols[i] for i in fold4ids]
                                nma = symbols.count(ma)
                                if nma == 0:
                                    composition = 4*mb
                                elif nma == 1:
                                    composition = ma + 3*mb
                                elif nma == 2:
                                    if sitetype == '5fold':
                                        idd = sorted(fold4ids[1:], key=lambda x:        
                                              get_mic(self.positions[x], self.positions[fold4ids[0]],
                                                      self.cell, return_squared_distance=True))
                                        opp, close = idd[-1], idd[0]
                                        if self.symbols[opp] == self.symbols[fold4ids[0]]:
                                            composition = ma + mb + ma + mb 
                                        else:
                                            if self.symbols[close] == self.symbols[fold4ids[0]]:
                                                composition = 2*ma + 2*mb 
                                            else:
                                                composition = ma + 2*mb + ma           
                                    else:
                                        opposite = np.where(
                                                   cm[si[1:],si[0]]==0)[0]
                                        opp = si[1+opposite[0]]         
                                        if self.symbols[opp] == self.symbols[fold4ids[0]]:
                                            composition = ma + mb + ma + mb 
                                        else:
                                            composition = 2*ma + 2*mb 
                                elif nma == 3:
                                    composition = 3*ma + mb
                                elif nma == 4:
                                    composition = 4*ma
                            site.update({'composition': composition})
                            if sitetype == '5fold':                   
                                site['composition'] = '{}-'.format(
                                self.symbols[isub]) + site['composition']
                        if self.subsurf_effect and sitetype != '5fold':
                            site.update({'subsurf_index': isub})
                            if self.composition_effect:
                                site.update({'subsurf_element': 
                                             self.symbols[isub]})
                        sl.append(site)
                        usi.add(si)
             
            # Make 3-fold hollow sites (differentiate fcc / hcp)
            if n == 1 and self.surface not in ['fcc100','bcc100','hcp10m10']:
                for i, refpos in enumerate(reduced_poss):
                    fold3_indices = nblist[ntop2+i]
                    fold3ids = [sorted_indices[j] for j in fold3_indices]
                    if len(fold3ids) != 3:
                        if self.surface != 'hcp10m11':
                            print('Cannot find correct indices of this 3-fold site')
                        continue
                    si = tuple(sorted(fold3ids))
                    pos = refpos + np.average(
                          self.delta_positions[fold3ids], 0)
                    normal = self.get_surface_normal(
                             [si[0], si[1], si[2]])
                    for idx in si:
                        normals_for_site[idx].append(normal)
                    occurence = np.sum(cm[fold3ids], axis=0)
                    if self.surface == 'fcc211':
                        siset = set(si)
                        if stepids.intersection(siset):
                            geometry = 'upperh'
                            if np.max(occurence) == 3:
                                sitetype = 'hcp'
                            else:
                                sitetype = 'fcc'
                        elif lowerids.intersection(siset):
                            geometry = 'lowerh' 
                            if np.max(occurence) == 3:
                                sitetype = 'hcp'
                            else:
                                sitetype = 'fcc'
                        else:
                            print('Cannot identify site {}'.format(si))
                            continue
                    elif self.surface in ['hcp10m11']:
                        sitetype, geometry = '3fold', 'h'
                    elif self.surface in ['bcc110','bcc111']:
                        sitetype, geometry = '3fold', 'o'
                    elif self.surface in ['fcc111','fcc110','fcc311','hcp0001']:
                        geometry = 'h'
                        if np.max(occurence) == 3:
                            sitetype = 'hcp'
                        else:
                            sitetype = 'fcc'
                    site = self.new_site()               
                    site.update({'site': sitetype,
                                 'surface': self.surface,
                                 'geometry': geometry,
                                 'position': pos,
                                 'normal': normal,
                                 'indices': si})
                    if self.composition_effect:                       
                        metals = self.metals                            
                        if len(metals) == 1:
                            composition = 3*metals[0]
                        else:
                            ma, mb = metals[0], metals[1]
                            symbols = [self.symbols[i] for i in si]
                            nma = symbols.count(ma)
                            if nma == 0:
                                composition = 3*mb
                            elif nma == 1:
                                composition = ma + 2*mb
                            elif nma == 2:
                                composition = 2*ma + mb
                            elif nma == 3:
                                composition = 3*ma
                        site.update({'composition': composition})   

                    if sitetype == 'hcp' and self.subsurf_effect:
                        isub = np.where(occurence == 3)[0][0]
                        site.update({'subsurf_index': isub})
                        if self.composition_effect:
                            site.update({'subsurf_element': 
                                         self.symbols[isub]})
                    sl.append(site)
                    usi.add(si)
 
            if n == 2 and self.surface in ['fcc100','bcc100',
            'hcp10m10'] and list(reduced_poss):
                fold4atoms = Atoms('X{}'.format(len(reduced_poss)), 
                                   positions=np.asarray(reduced_poss),
                                   cell=self.cell, 
                                   pbc=self.pbc)
                sorted_top = self.surf_ids
                ntop = len(sorted_top)
                topatoms = self.ref_atoms[sorted_top]
                newatoms = topatoms + fold4atoms
                newnblist = neighbor_shell_list(newatoms, dx=.1, 
                                                neighbor_number=2,
                                                different_species=True, 
                                                mic=True) 

                for i, refpos in enumerate(reduced_poss): 
                    fold4_indices = newnblist[ntop+i]                     
                    fold4ids = [sorted_top[j] for j in fold4_indices]
                    occurence = np.sum(cm[fold4ids], axis=0)
                    isub = np.where(occurence == 4)[0][0]
                    si = tuple(sorted(fold4ids))
                    pos = refpos + np.average(
                          self.delta_positions[fold4ids], 0)
                    normal = self.get_surface_normal(
                             [si[0], si[1], si[2]])
                    for idx in si:
                        normals_for_site[idx].append(normal)
 
                    site = self.new_site()
                    if self.surface in ['hcp10m10']:
                        sitetype = '5fold'
                        site.update({'site': sitetype,
                                     'surface': self.surface,
                                     'geometry': 'subsurf',
                                     'position': self.positions[isub],
                                     'normal': normal,
                                     'indices': tuple(sorted(fold4ids+[isub]))})
                    else:
                        sitetype = '4fold'
                        site.update({'site': sitetype,
                                     'surface': self.surface,
                                     'geometry': 't',
                                     'position': pos,
                                     'normal': normal,
                                     'indices': si})                    
                    if self.composition_effect:                       
                        metals = self.metals
                        if len(metals) == 1:
                            composition = 4*metals[0] 
                        else:
                            ma, mb = metals[0], metals[1]
                            symbols = [self.symbols[i] for i in si]
                            nma = symbols.count(ma)
                            if nma == 0:
                                composition = 4*mb
                            elif nma == 1:
                                composition = ma + 3*mb
                            elif nma == 2:
                                if sitetype == '5fold':
                                    idd = sorted(fold4ids[1:], key=lambda x:       
                                          get_mic(self.positions[x], self.positions[fold4ids[0]],
                                                  self.cell, return_squared_distance=True))
                                    opp, close = idd[-1], idd[0]
                                    if self.symbols[opp] == self.symbols[fold4ids[0]]:
                                        composition = ma + mb + ma + mb 
                                    else:
                                        if self.symbols[close] == self.symbols[fold4ids[0]]:
                                            composition = 2*ma + 2*mb 
                                        else:
                                            composition = ma + 2*mb + ma  
                                else:
                                    opposite = np.where(cm[si[1:],si[0]]==0)[0]
                                    opp = si[1+opposite[0]]         
                                    if self.symbols[opp] == self.symbols[fold4ids[0]]:
                                        composition = ma + mb + ma + mb 
                                    else:
                                        composition = 2*ma + 2*mb 
                            elif nma == 3:
                                composition = 3*ma + mb
                            elif nma == 4:
                                composition = 4*ma
                        site.update({'composition': composition})
                        if sitetype == '5fold':
                            site['composition'] = '{}-'.format(
                            self.symbols[isub]) + site['composition'] 
                    if self.subsurf_effect and sitetype != '5fold':
                        site.update({'subsurf_index': isub})
                        if self.composition_effect:
                            site.update({'subsurf_element': 
                                         self.symbols[isub]})
                    sl.append(site)
                    usi.add(si)

        index_list, pos_list, st_list = [], [], []
        for t in sl:
            stids = t['indices']
            sitetype = t['site']
            # Add normals to ontop sites                                
            if sitetype == 'ontop':
                n = np.average(normals_for_site[stids[0]], 0)
                t['normal'] = n / np.linalg.norm(n)
                nstids = len(stids)
                if nstids > 1:
                    t['site'] = '{}fold'.format(nstids)
                    t['indices'] = tuple(sorted(stids))

            # Add normals to bridge sites
            elif 'bridge' in sitetype:
                if t['geometry'] in ['terrace', 'step', 'lowerstep']:
                    normals = []
                    for i in stids[:2]:
                        normals.extend(normals_for_site[i])
                    n = np.average(normals, 0)
                    t['normal'] = n / np.linalg.norm(n)
                    nstids = len(stids)
                    if nstids > 2:
                        t['site'] = '{}fold'.format(nstids)
                        t['indices'] = tuple(sorted(stids))
                else:
                    if self.surface == 'hcp10m10':
                        normals = [s['normal'] for s in sl if s['geometry'] 
                                   == 'subsurf' and len(set(s['indices']
                                   ).intersection(set(t['indices']))) == 2]
                    else:
                        # Make sure upperx and lowerx geometry are the same as x
                        normals = [s['normal'] for s in sl if 
                                   s['geometry'][-1] == t['geometry'][-1] and
                                   'bridge' not in s['site']]
                    t['normal'] = np.average(normals, 0)
                    nstids = len(stids)
                    if nstids > 2:
                        t['site'] = '{}fold'.format(nstids)
                        t['indices'] = tuple(sorted(stids))

            # Take care of duplicate 3-fold indices. When unit cell is 
            # small, different sites can have exactly same indices
            elif sitetype in ['fcc', 'hcp']:
                if stids in index_list:
                    slid = next(si for si in range(len(sl)) if 
                                sl[si]['indices'] == stids)
                    previd = index_list.index(stids)
                    prevpos = pos_list[previd]
                    prevst = st_list[previd]
                    if min([np.linalg.norm(prevpos - pos) for pos 
                    in self.positions[self.subsurf_ids]]) < \
                    min([np.linalg.norm(t['position'] - pos) for pos
                    in self.positions[self.subsurf_ids]]):
                        t['site'] = 'fcc'
                        if prevst == 'fcc':
                            sl[slid]['site'] = 'hcp'
                        if self.subsurf_effect:
                            t['subsurf_index'] = None
                            if self.composition_effect:
                                t['subsurf_element'] = None 
                    else:
                        t['site'] = 'hcp'
                        if prevst == 'hcp':
                            sl[slid]['site'] = 'fcc'
                        if self.subsurf_effect:
                            sl[slid]['subsurf_index'] = None
                            if self.composition_effect:
                                sl[slid]['subsurf_element'] = None
                else:
                    index_list.append(t['indices'])
                    pos_list.append(t['position'])
                    st_list.append(t['site'])

        # Add 6-fold sites if allowed
        if self.allow_6fold:
            dh = 1/2*(meansurfz - np.average(
                      self.atoms.positions[self.subsurf_ids][:,2], 0))
            for st in sl:
                if self.surface == 'fcc110' and st['site'] == 'bridge' \
                and st['geometry'] == 'step':
                    site = st.copy()
                    sid = list(st['indices'])
                    occurence = cm[sid]
                    surfsi = sid + list(np.where(np.sum(occurence, axis=0) == 2)[0])
                    subsi = [self.subsurf_ids[i] for i in np.where(np.sum(
                             occurence[:,self.subsurf_ids], axis=0) == 1)[0]]
                    si = tuple(sorted(surfsi + subsi))
                    normal = np.asarray([0.,0.,1.])
                    subpos = st['position'] - st['normal'] * dh
                    site.update({'site': '6fold',
                                 'position': subpos,
                                 'geometry': 'subsurf',
                                 'normal': normal,
                                 'indices': si})
                elif st['site'] == 'fcc':
                    site = st.copy()
                    subpos = st['position'] - st['normal'] * dh                                
                    def get_squared_distance(x):
                        return get_mic(self.positions[x], subpos, 
                                       self.cell, return_squared_distance=True)
                    subsi = sorted(sorted(self.subsurf_ids, 
                                          key=get_squared_distance)[:3])     
                    si = site['indices']
                    site.update({'site': '6fold',      
                                 'position': subpos,
                                 'geometry': 'subsurf',
                                 'indices': tuple(sorted(si+tuple(subsi)))})
                else:
                    continue
                # Find the opposite element
                if self.composition_effect:
                    metals = self.metals
                    if self.surface == 'fcc110':
                        newsi = surfsi[:-1] 
                        metals = self.metals                                                    
                        if len(metals) == 1:
                            comp = 3*metals[0]
                        else:
                            ma, mb = metals[0], metals[1]
                            symbols = [self.symbols[i] for i in newsi]
                            nma = symbols.count(ma)
                            if nma == 0:
                                comp = 3*mb
                            elif nma == 1:
                                comp = ma + 2*mb
                            elif nma == 2:
                                comp = 2*ma + mb
                            elif nma == 3:
                                comp = 3*ma
                    else:
                        comp = site['composition']
                    if len(metals) == 1:
                        composition = ''.join([comp, 3*metals[0]])
                    else: 
                        ma, mb = metals[0], metals[1]
                        subsyms = [self.symbols[subi] for subi in subsi]
                        nma = subsyms.count(ma)
                        if nma == 0:
                            composition = ''.join([comp, 3*mb])
                        elif nma == 1:
                            ia = subsi[subsyms.index(ma)]
                            subpos = self.positions[ia]
                            if self.symbols[max(si, key=get_squared_distance)] == ma:
                                composition = ''.join([comp, 2*mb + ma])
                            else:
                                composition = ''.join([comp, ma + 2*mb])
                        elif nma == 2:
                            ib = subsi[subsyms.index(mb)]
                            subpos = self.positions[ib]
                            if self.symbols[max(si, key=get_squared_distance)] == mb:
                                composition = ''.join([comp, 2*ma + mb])
                            else:
                                composition = ''.join([comp, mb + 2*ma])
                        elif nma == 3:
                            composition = ''.join([comp, 3*ma])
                    site.update({'composition': composition})
                sl.append(site)
                usi.add(si)

    def new_site(self):
        return {'site': None, 'surface': None, 'geometry': None, 
                'position': None, 'normal': None, 'indices': None,
                'composition': None, 'subsurf_index': None,
                'subsurf_element': None}

    def get_site(self, indices):
        if not isinstance(indices, Iterable):
            indices = [indices]
        indices = tuple(sorted(indices))
        st = next((s for s in self.site_list if 
                   s['indices'] == indices), None)
        return st

    def get_all_sites_of_type(self, type):            
        return [i for i in self.site_list
                if i['site'] == type]
                                                      
    def get_all_fcc_sites(self):
        return self.get_all_sites_of_type('fcc')
                                                      
    def get_all_from_geometry(self, geometry):
        return [i for i in self.site_list
                if i['geometry'] == geometry]
                                                      
    def get_sites_from_geometry(self, site, surface):
        surf = self.get_all_from_surface(surface)
        return [i for i in surf if i['site'] == site]

    def make_neighbor_list(self, dx=.5, neighbor_number=1):
        """Generate a periodic neighbor list (defaultdict).""" 
        self.nblist = neighbor_shell_list(self.ref_atoms, dx, 
                                          neighbor_number, mic=True)

    def get_connectivity(self):                                      
        """Generate a connections matrix from neighbor_shell_list."""       
        return get_connectivity_matrix(self.nblist)                   

    def get_termination(self):
        """Return lists surf and subsurf containing atom indices belonging to
        those subsets of a surface atoms object.
        This function relies on PTM and the connectivity of the atoms.
        """
    
        xcell = self.cell[0][0]
        ycell = self.cell[1][1] 
        xmul = math.ceil(15/xcell)
        ymul = math.ceil(15/ycell) 
        atoms = self.ref_atoms*(xmul,ymul,1)
        cm = self.connectivity_matrix.copy()                               
        np.fill_diagonal(cm, 0)
        indices = self.indices 

        if self.surface in ['fcc110','bcc111']:
            rmsd = 0.1
        else: 
            rmsd = 0.25
        ptmdata = PTM(atoms, rmsd_max=rmsd)
        allbigsurf = np.where(ptmdata['structure'] == 0)[0]
        allsurf = [i for i in allbigsurf if i < len(indices)]
        bulk = [i for i in indices if i not in allsurf]
        surfcm = cm.copy()
        surfcm[bulk] = 0
        surfcm[:,bulk] = 0
        
        # Use networkx to separate top layer and bottom layer
        rows, cols = np.where(surfcm == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        G.add_edges_from(edges)
        components = nx.connected_components(G)
        surf = list(max(components, 
                    key=lambda x:np.mean(
                    self.ref_atoms.positions[list(x),2])))
        subsurf = []
        for a_b in bulk:
            for a_t in surf:
                if cm[a_t, a_b] > 0:
                    subsurf.append(a_b)
        subsurf = list(np.unique(subsurf))
        return sorted(surf), sorted(subsurf)

    def get_two_vectors(self, sites):
        p1 = self.positions[sites[1]]
        p2 = self.positions[sites[2]]
        vec1 = get_mic(p1, self.positions[sites[0]], self.cell)
        vec2 = get_mic(p2, self.positions[sites[0]], self.cell)
        return vec1, vec2

    def get_surface_normal(self, sites):
        vec1, vec2 = self.get_two_vectors(sites)
        n = np.cross(vec1, vec2)
        l = math.sqrt(n @ n.conj())
        n /= l
        if n[2] < 0:
            n *= -1
        return n

    def get_CNA(self, rCut=None):                  
        atoms = self.ref_atoms.copy()
        _, extcoords, _ = expand_cell(atoms, cutoff=5.0)
        fracs = extcoords @ np.linalg.pinv(self.cell)
        xfracs, yfracs = fracs[:,0], fracs[:,1]
        nums = list(set(atoms.numbers))
        crmax = max([covalent_radii[i] for i in nums])
        dx = 3 * crmax / np.linalg.norm(self.cell[0])            
        dy = 3 * crmax / np.linalg.norm(self.cell[1])             
                                                   
        # Extend the surface to get the correct surface CNA
        outer = np.where((xfracs > 0-dx) & (xfracs < 1+dx) & \
                         (yfracs > 0-dy) & (yfracs < 1+dy))[0]
        inner = np.where((xfracs > 0-self.tol) & (xfracs < 1-self.tol) & \
                         (yfracs > 0-self.tol) & (yfracs < 1-self.tol))[0]
        ringcoords = [cord for i, cord in enumerate(extcoords) if 
                              (i in outer) & (i not in inner)]
        ringatoms = Atoms(numbers = len(ringcoords)*[nums[0]],
                          positions = np.asarray(ringcoords),
                          cell = self.cell, pbc = self.pbc)
        fullatoms = atoms + ringatoms
        fullatoms.center(vacuum=5.)            
        fullCNA = FullCNA(fullatoms, rCut=rCut).get_normal_cna() 
        surfCNA = [fullCNA[i] for i in self.surf_ids]
        return fullCNA, surfCNA
   
    def identify_surface(self):
        # Identify surfaces based on CNA. Turns out to be very 
        # inconsistent for surfaces slabs.                 
        sd = self.site_dict
        scna = self.surfCNA
        fcc100_weight = fcc111_weight = fcc110_weight = \
        fcc211_weight = fcc311_weight = 0
        for s in scna:
            if str(s) in sd['fcc100']:
                fcc100_weight += 1
            if str(s) in sd['fcc111']:
                fcc111_weight += 1
            if str(s) in sd['fcc110']:
                fcc110_weight += 1
            if str(s) in sd['fcc211']:
                fcc211_weight += 1
            if str(s) in sd['fcc311']:
                fcc311_weight += 1
            if str(s) not in sd['fcc100'] + sd['fcc111'] + sd['fcc110'] \
                           + sd['fcc211'] + sd['fcc311']:
                fcc110_weight += 1
                fcc211_weight += 1
 
        full_weights = [fcc100_weight, fcc111_weight, fcc110_weight,
                        fcc211_weight, fcc311_weight]
         
        if full_weights.count(max(full_weights)) > 1:
            raise ValueError('Cannot identify the surface. \
                              Please specify the surface')
        elif fcc100_weight == max(full_weights):
            return 'fcc100'
        elif fcc111_weight == max(full_weights): 
            return 'fcc111'
        elif fcc110_weight == max(full_weights):
            return 'fcc110'
        elif fcc211_weight == max(full_weights):
            return 'fcc211'
        elif fcc311_weight == max(full_weights):
            return 'fcc311'

    def get_unique_sites(self, unique_composition=False,                
                         unique_subsurf=False):
        sl = self.site_list
        key_list = ['site', 'geometry']
        if unique_composition:
            if not self.composition_effect:
                raise ValueError('The site list does not include ',
                                 'information of composition')
            key_list.append('composition')
            if unique_subsurf:
                if not self.subsurf_effect:
                    raise ValueError('The site list does not include ',
                                     'information of subsurface')
                key_list.append('subsurf_element') 
        else:
            if unique_subsurf:
                raise ValueError('To include the subsurface element ',
                                 'unique_composition also need to ',
                                 'be set to True') 
        sklist = sorted([[s[k] for k in key_list] for s in sl])
 
        return sorted(list(sklist for sklist, _ in groupby(sklist)))

    def get_graph(self):                              
        cm = self.connectivity_matrix
        G = nx.Graph()                                                  
        # Add edges from surface connectivity matrix
        rows, cols = np.where(cm == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G.add_edges_from(edges)
        return G

    def get_neighbor_site_list(self, neighbor_number=1, span=True):           
        """Returns the site_list index of all neighbor 
        shell sites for each site
        """

        sl = self.site_list
        refposs = np.asarray([s['position'] - np.average(
                             self.delta_positions[list(
                             s['indices'])], 0) + s['normal'] * \
                             heights_dict[s['site']] for s in sl])
        statoms = Atoms('X{}'.format(len(sl)), 
                        positions=refposs, 
                        cell=self.cell, 
                        pbc=self.pbc)
        cr = 0.55 
        if neighbor_number == 1:
            cr += 0.1

        return neighbor_shell_list(statoms, 0.1, neighbor_number,
                                   mic=True, radius=cr, span=span)

    def update_positions(self, new_atoms):
        sl = self.site_list
        new_slab = new_atoms[[a.index for a in new_atoms if 
                              a.symbol not in adsorbate_elements]]
        shifted_positions = new_slab.positions - self.atoms.positions
        for st in sl:
            si = list(st['indices'])
            st['position'] += np.average(shifted_positions[si], 0) 


def get_adsorption_site(atoms, indices, surface=None, return_index=False):
    if not isinstance(indices, Iterable):                      
        indices = [indices]                                        
    indices = tuple(sorted(indices))

    if True not in atoms.pbc:
        sas = NanoparticleAdsorptionSites(atoms, 
                                          allow_6fold=True,
                                          composition_effect=False,
                                          subsurf_effect=False)
                                                             
    else:
        sas = SlabAdsorptionSites(atoms, surface, 
                                  allow_6fold=True, 
                                  composition_effect=False, 
                                  subsurf_effect=False)             
    site_list = sas.site_list                              
    sti, site = next(((i, s) for i, s in enumerate(site_list) if                        
                      s['indices'] == indices), None)                     
    if return_index:
        return sti, site
                    
    return site    


def enumerate_adsorption_sites(atoms, 
                               surface=None, 
                               geometry=None, 
                               allow_6fold=False,
                               composition_effect=False, 
                               subsurf_effect=False):

    if True not in atoms.pbc:
        nas = NanoparticleAdsorptionSites(atoms, 
                                          allow_6fold,
                                          composition_effect,
                                          subsurf_effect)
        all_sites = nas.site_list
        if surface:
            all_sites = [s for s in all_sites if 
                         s['surface'] == surface] 

    else:
        sas = SlabAdsorptionSites(atoms, surface,
                                  allow_6fold, 
                                  composition_effect, 
                                  subsurf_effect)            
        all_sites = sas.site_list
        if geometry:
            all_sites = [s for s in all_sites if 
                         s['geometry'] == geometry]       

    return all_sites
