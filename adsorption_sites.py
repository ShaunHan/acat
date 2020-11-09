from .utilities import PeriodicNeighborList, get_connectivity_matrix
from .utilities import get_plane_normal, get_mic_distance, expand_cell
from asap3.analysis.rdf import RadialDistributionFunction
from asap3 import FullNeighborList
from asap3.analysis import FullCNA, PTM
from asap3 import EMT as asapEMT
from ase.build import molecule
from ase.geometry import Cell, get_layers
from ase.optimize import BFGS, FIRE
from ase import Atom, Atoms
from operator import itemgetter
import numpy as np
from itertools import combinations, groupby
from ase.data import reference_states, atomic_numbers, covalent_radii
from ase.io import read, write
from collections import Counter, defaultdict
import networkx as nx
import scipy
import re
import math
import random
import warnings


#warnings.filterwarnings('ignore')


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
adsorbates = 'SCHON'


class NanoparticleAdsorptionSites(object):

    def __init__(self, atoms, show_composition=False, show_subsurface=False):

        assert True not in atoms.pbc
        atoms = atoms.copy()
        del atoms[[a.index for a in atoms if 'a' not in reference_states[a.number]]]
        del atoms[[a.index for a in atoms if a.symbol in adsorbates]]
        self.atoms = atoms
        self.show_composition = show_composition
        self.show_subsurface = show_subsurface

        self.fullCNA = {}
        self.make_fullCNA()
        self.set_first_neighbor_distance_from_rdf()
        self.site_dict = self.get_site_dict()
        self.make_neighbor_list()
        self.surf_sites = self.get_surface_sites()

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
                neighbors, _, dist2 = self.nlist.get_neighbors(s, self.r + 0.2)
                for n in neighbors:
                    si = tuple(sorted([s, n]))  # site_indices
                    if n in ssall and si not in usi:
                        # bridge sites
                        pos = np.average(self.atoms[[n, s]].positions, 0)
                        site_surf = self.get_surface_designation([s, n])
                        site = self.new_site()
                        site.update({'site': 'bridge',
                                     'surface': site_surf,
                                     'position': pos,
                                     'indices': si})
                        if self.show_composition:                         
                            symbols = [(self.atoms[i].symbol, 
                                        self.atoms[i].number) for i in si]
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
                            pos = np.average(self.atoms[[n, m,
                                                         s]].positions, 0)
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
                            if self.show_composition:                       
                                metals = list(set(self.atoms.symbols))
                                ma, mb = metals[0], metals[1]
                                if atomic_numbers[ma] > atomic_numbers[mb]:
                                    ma, mb = metals[1], metals[0]
                                symbols = [self.atoms[i].symbol for i in si]
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

                            if this_site == 'hcp' and self.show_subsurface:
                                hcp_nbrs = []
                                for i in si:
                                    hcp_nbrs += list(self.nlist.get_neighbors(i, 
                                                                self.r + 1.)[0])
                                isub = [key for key, count in Counter(
                                        hcp_nbrs).items() if count == 3][0]
                                site.update({'subsurface_id': isub})
                                if self.show_composition:
                                    site.update({'subsurface_element': 
                                                 self.atoms[isub].symbol})
                            sl.append(site)
                            usi.add(si)

                        elif self.is_eq(angle, np.pi/2.):
                            # 4-fold hollow site
                            site_surf = 'fcc100'
                            l2 = self.r * np.sqrt(2) + 0.2
                            nebs2, _, _ = self.nlist.get_neighbors(s, l2)
                            nebs2 = [k for k in nebs2 if k not in neighbors]
                            for k in nebs2:
                                si = tuple(sorted([s, n, m, k]))
                                if k in ssall and si not in usi:
                                    d1 = self.atoms.get_distance(n, k)
                                    if self.is_eq(d1, self.r, 0.2):
                                        d2 = self.atoms.get_distance(m, k)
                                        if self.is_eq(d2, self.r, 0.2):
                                            # 4-fold hollow site found
                                            normal = self.get_surface_normal(
                                                [s, n, m])
                                            # Save the normals now and add them to the site later
                                            for i in [s, n, m, k]:
                                                normals_for_site[i].append(normal)
                                            ps = self.atoms[[
                                                n, m, s, k]].positions
                                            pos = np.average(ps, 0)
 
                                            site = self.new_site()
                                            site.update({'site': '4fold',
                                                         'surface': site_surf,
                                                         'position': pos,
                                                         'normal': normal,
                                                         'indices': si})
                                            if self.show_composition:
                                                metals = list(set(self.atoms.symbols))      
                                                ma, mb = metals[0], metals[1]
                                                if atomic_numbers[ma] > atomic_numbers[mb]:
                                                    ma, mb = metals[1], metals[0]
                                                symbols = [self.atoms[i].symbol for i in si]
                                                nma = symbols.count(ma)
                                                if nma == 0:
                                                    composition = 4*mb
                                                elif nma == 1:
                                                    composition = ma + 3*mb
                                                elif nma == 2:
                                                    opp = max(list(si[1:]), key=lambda x: 
                                                          np.linalg.norm(self.atoms[x].position
                                                                 - self.atoms[si[0]].position)) 
                                                    if self.atoms[opp].symbol == \
                                                       self.atoms[si[0]].symbol:
                                                        composition = ma + mb + ma + mb 
                                                    else:
                                                        composition = 2*ma + 2*mb 
                                                elif nma == 3:
                                                    composition = 3*ma + mb
                                                elif nma == 4:
                                                    composition = 4*ma
                                                site.update({'composition': composition})    
                                            if self.show_subsurface:                        
                                                fold4_nbrs = []
                                                for i in si:
                                                    fold4_nbrs += list(self.nlist.get_neighbors(
                                                                             i, self.r + 1.)[0])
                                                isub = [key for key, count in Counter(
                                                        fold4_nbrs).items() if count == 4][0]
                                                site.update({'subsurface_id': isub})
                                                if self.show_composition:
                                                    site.update({'subsurface_element': 
                                                                 self.atoms[isub].symbol})
                                            sl.append(site)
                                            usi.add(si)

                # ontop sites
                site = self.new_site()
                site.update({'site': 'ontop', 
                             'surface': surface,
                             'position': self.atoms[s].position,
                             'indices': (s,)})
                if self.show_composition:
                    site.update({'composition': self.atoms[s].symbol})
                sl.append(site)
                usi.add((s))

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


    def new_site(self):
        return {'site': None, 'surface': None, 'position': None, 
                'normal': None, 'indices': None, 'composition': None,
                'subsurface_id': None, 'subsurface_element': None}

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
        p1 = self.atoms[int(sites[1])].position
        p2 = self.atoms[int(sites[2])].position
        vec1 = p1 - self.atoms[int(sites[0])].position
        vec2 = p2 - self.atoms[int(sites[0])].position
        return vec1, vec2

    def is_eq(self, v1, v2, eps=0.1):
        if abs(v1 - v2) < eps:
            return True
        else:
            return False

    def get_surface_normal(self, sites):
        vec1, vec2 = self.get_two_vectors(sites)
        n = np.cross(vec1, vec2)
        l = np.sqrt(np.dot(n, n.conj()))
        new_pos = self.atoms[sites[0]].position + self.r * n / l
        # Add support for having adsorbates on the particles already
        # by putting in elements to check for in the function below
        j = 2 * int(self.no_atom_too_close_to_pos(new_pos, (5./6)*self.r)) - 1
        return j * n / l

    def get_angle(self, sites):
        vec1, vec2 = self.get_two_vectors(sites)
        p = np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec1))
        return np.arccos(p)

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
        atoms = self.atoms.copy()
        fcna = self.get_fullCNA()
        site_dict = self.site_dict

        for i in range(len(atoms)):
#            if i == 185:
#                print(fcna[i])
            if sum(fcna[i].values()) < 12:
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
        return surface_sites  

    def make_fullCNA(self, rCut=None):                  
        if rCut not in self.fullCNA:
            self.fullCNA[rCut] = FullCNA(self.atoms, 
                                 rCut=rCut).get_normal_cna()

    def get_fullCNA(self, rCut=None):
        if rCut not in self.fullCNA:
            self.make_fullCNA(rCut=rCut)
        return self.fullCNA[rCut]

    def make_neighbor_list(self, rMax=10.):
        self.nlist = FullNeighborList(rCut=rMax, atoms=self.atoms)

    def get_site_dict(self):
        fcna = self.get_fullCNA()
        icosa_weight = ticosa_weight = cubocta_weight = \
        mdeca_weight = octa_weight = tocta_weight = 0
        for s in fcna:
            if str(s) in icosa_dict:
                icosa_weight += 1
            if str(s) in ticosa_dict:
                ticosa_weight += 1
            if str(s) in cubocta_dict:
                cubocta_weight += 1
            if str(s) in mdeca_dict:
                mdeca_weight += 1
            if str(s) in octa_dict:
                octa_weight += 1
            if str(s) in tocta_dict:
                tocta_weight += 1
        full_weights = [icosa_weight, ticosa_weight, cubocta_weight, 
                        mdeca_weight, octa_weight, tocta_weight]
        if icosa_weight == max(full_weights):
            return icosa_dict
        elif ticosa_weight == max(full_weights):
            return ticosa_dict
        elif cubocta_weight == max(full_weights):
            return cubocta_dict
        elif mdeca_weight == max(full_weights):
            return mdeca_dict
        elif octa_weight == max(full_weights):
            return octa_dict
        else:
            return tocta_dict

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
            if ss01 in [('edge', 'edge'), ('edge', 'vertex'),
                        ('vertex', 'vertex')]:
                return 'edge'
            elif ss01 in [('edge', 'fcc111'), ('fcc111', 'fcc111')]:
                return 'fcc111'
            elif ss01 in [('edge', 'fcc100'), ('fcc100', 'fcc100')]:
                return 'fcc100'
        elif len(sites) == 3:
            return 'fcc111'

    def get_unique_sites(self, unique_composition=False,               
                         unique_subsurface=False):
        sl = self.site_list.copy()
        key_list = ['site', 'surface']
        if unique_composition:
            if not self.show_composition:
                raise ValueError('The site list does not include \
                                  information of composition')
            key_list.append('composition')
            if unique_subsurface:
                if not self.show_subsurface:
                    raise ValueError('The site list does not include \
                                      information of subsurface')
                key_list.append('subsurface_element') 
        else:
            if unique_subsurface:
                raise ValueError('To include the subsurface element, \
                                  unique_composition also need to \
                                  be set to True')    
        name_list = [[s[k] for k in key_list] for s in sl]
        name_list.sort()
 
        return list(name_list for name_list,_ in groupby(name_list))  


class SlabAdsorptionSites(object):

    def __init__(self, atoms, surface=None, show_composition=False, 
                 show_subsurface=False, tol=1e-5):

        assert True in atoms.pbc    
        atoms = atoms.copy() 
        del atoms[[a.index for a in atoms if 'a' not in 
                               reference_states[a.number]]]
        del atoms[[a.index for a in atoms if a.symbol in adsorbates]]
        self.atoms = atoms
        self.indices = [a.index for a in self.atoms] 

        refatoms = self.atoms.copy()        
        for a in refatoms:
            a.symbol = 'Pt'
        refatoms.calc = asapEMT()
        opt = FIRE(refatoms, logfile=None)
        opt.run(fmax=0.1) 
        self.refatoms = refatoms
        self.delta_positions = atoms.positions - refatoms.positions
        self.cell = atoms.cell
        self.pbc = atoms.pbc
        self.tol = tol
        self.show_composition = show_composition
        self.show_subsurface = show_subsurface
        self.site_dict = surf_dict 
        self.r = 2.675

        self.make_neighbor_list(dx=.3) 
        self.connectivity_matrix = self.get_connectivity()         
        self.surf_ids, self.subsurf_ids = self.get_termination()        
        if surface is None: 
            self.fullCNA, self.surfCNA = self.get_CNA(rCut=self.r+.3)
            self.surface = self.identify_surface()
            print('The surface is identified as {}. Please specify \
                   the surface if it is incorrect'.format(self.surface))
        else:
            self.surface = surface

        self.site_list = []
        self.populate_site_list()

    def populate_site_list(self, allow_obtuse=True):
        """Find all ontop, bridge and hollow sites (3-fold and 4-fold) 
           given an input slab based on Delaunay triangulation of 
           surface atoms of a super-cell and collect in a site list. 
           (Adaption from Catkit)
        """
 
        top_indices = self.surf_ids
        sl = self.site_list
        usi = set()  # used_site_indices
        cm = self.connectivity_matrix 

        for s in top_indices:
            occurence = cm[s]
            sumo = np.sum(occurence, axis=0)
            if self.surface in ['fcc100', 'fcc111']:
                geometry = self.surface[-3:]
            else:
                if sumo == 7:
                    geometry = 'step'
                elif sumo in [9, 11]:
                    geometry = 'terrace'
                elif sumo == 10:
                    if self.surface == 'fcc211':
                        geometry = 'lowerstep'
                    elif self.surface == 'fcc311':
                        geometry = 'terrace'
                else:
                    print('Cannot identify site {}'.format(s))
                    continue
            si = (s,)
            site = self.new_site()
            site.update({'site': 'ontop',
                         'surface': self.surface,
                         'geometry': geometry,
                         'position': self.atoms[s].position,
                         'normal': np.array([0,0,1]),
                         'indices': si})
            if self.show_composition:
                site.update({'composition': self.atoms[s].symbol})
                if self.surface == 'fcc110' and geometry == 'terrace':
                    site.update({'extra': np.where(occurence==1)[0]})
            sl.append(site)
            usi.add(si)

        if self.surface == 'fcc110' and self.show_composition:
            stepids = [s['indices'][0] for s in sl 
                       if s['geometry'] == 'step']
            for st in sl:
                if 'extra' in st:
                    extra = st['extra']
                    extraids = [e for e in extra if e in stepids]
                    if len(extraids) != 4:
                        print('Cannot identify other 4 atoms \
                               for this 5-fold site')
                    metals = list(set(self.atoms.symbols))      
                    ma, mb = metals[0], metals[1]
                    if atomic_numbers[ma] > atomic_numbers[mb]:
                        ma, mb = metals[1], metals[0]
                    symbols = [self.atoms[i].symbol for i in extraids]
                    nma = symbols.count(ma)
                    if nma == 0:
                        composition = 4*mb
                    elif nma == 1:
                        composition = ma + 3*mb
                    elif nma == 2:
                        idd = sorted(extraids[1:], key=lambda x: 
                              np.linalg.norm(self.atoms[x].position
                              - self.atoms[extraids[0]].position))
                        opp = idd[-1]
                        close = idd[0]
                        if self.atoms[opp].symbol == \
                        self.atoms[extraids[0]].symbol:
                            composition = ma + mb + ma + mb 
                        else:
                            if self.atoms[close].symbol == \
                            self.atoms[extraids[0]].symbol:
                                composition = 2*ma + 2*mb 
                            else:
                                composition = ma + 2*mb + ma
                    elif nma == 3:
                        composition = 3*ma + mb
                    elif nma == 4:
                        composition = 4*ma
                    st['composition'] += '-{}'.format(composition)
                    del st['extra']

        ext_index, ext_coords, _ = expand_cell(self.refatoms, cutoff=5.0)
        extended_top = np.where(np.in1d(ext_index, top_indices))[0]
        ext_surf_coords = ext_coords[extended_top]
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

        # Complete information of each site
        for n, poss in enumerate([bridge_positions, fold3_positions, 
                                  fold4_positions]):
            if not poss:
                continue
            fracs = np.dot(np.stack(poss, axis=0), np.linalg.pinv(self.cell))
            xfracs, yfracs = fracs[:,0], fracs[:,1]

            # Take only the positions within the periodic boundary
            screen = np.where((xfracs > 0-self.tol) & (xfracs < 1-self.tol) & \
                              (yfracs > 0-self.tol) & (yfracs < 1-self.tol))[0]
            reduced_poss = np.asarray(poss)[screen]

            # Sort the index list of surface and subsurface atoms 
            # so that we can retrive original indices later
            sorted_indices = sorted(self.surf_ids + self.subsurf_ids)
            top2atoms = self.refatoms[sorted_indices]

            # Make a new neighborlist including sites as dummy atoms
            dummies = Atoms('X{}'.format(reduced_poss.shape[0]), 
                            positions=reduced_poss, 
                            cell=self.cell, 
                            pbc=self.pbc)

            ntop2 = len(top2atoms)
            testatoms = top2atoms.extend(dummies)
            nblist = PeriodicNeighborList(testatoms, dx=.1,   
                                          neighbor_number=1, 
                                          different_species=True) 
            # Make bridge sites  
            if n == 0:
                fold4_poss = []
                for i, refpos in enumerate(reduced_poss):
                    bridge_indices = nblist[ntop2+i]                     
                    bridgeids = [sorted_indices[j] for j in bridge_indices]
                    if len(bridgeids) != 2: 
                        if self.surface in ['fcc100', 'fcc211', 'fcc311']:
                            fold4_poss.append(refpos)
                        else:
                            'Cannot identify site {}'.format(bridgeids)
                        continue

                    si = tuple(sorted(bridgeids))
                    pos = refpos + np.average(self.delta_positions[bridgeids], 0) 
                    occurence = np.sum(cm[bridgeids], axis=0)
                    sumo = np.sum(occurence)

                    if self.surface == 'fcc110':
                        if list(occurence).count(2) == 2:
                            geometry = 'step'
                        elif list(occurence).count(2) == 3:
                            geometry = '111'
                        elif list(occurence).count(2) == 4:
                            extraids = [i for i in np.where(occurence==2)[0]
                                        if i in self.surf_ids]
                            geometry = 'terrace'
                        else:
                            print('Cannot identify site {}'.format(si)) 
                            continue         

                    elif self.surface == 'fcc311': 
                        if sumo == 14:
                            geometry = 'step'
                        elif sumo == 17:
                            if list(occurence).count(2) == 2:
                                geometry = '100'
                            elif list(occurence).count(2) == 3:
                                geometry = '111'
                        elif sumo == 20:
                            geometry = 'terrace'
                        else:
                            print('Cannot identify site {}'.format(si)) 
                            continue         
 
                    elif self.surface == 'fcc211':                    
                        if sumo == 14:
                            geometry = 'step'
                        elif sumo in [15, 16]:
                            geometry = 'upper111'
                        elif sumo == 17:
                            geometry = '100'
                        # sumo == 18 is actually terrace bridge, 
                        # but equivalent to lower111 for fcc211
                        elif sumo in [18, 19]:
                            geometry = 'lower111'
                        elif sumo == 20:
                            geometry = 'lowerstep'
                        else:
                            print('Cannot identify site {}'.format(si)) 
                            continue         
                    elif self.surface in ['fcc100', 'fcc111']:
                        geometry = self.surface[-3:]
                                    
                    site = self.new_site()                         
                    site.update({'site': 'bridge',               
                                 'surface': self.surface,
                                 'geometry': geometry,
                                 'position': pos,
                                 'normal': np.array([0,0,1]),
                                 'indices': si})           
                    if self.show_composition:                          
                        symbols = [(self.atoms[i].symbol, 
                                    self.atoms[i].number) for i in si]
                        comp = sorted(symbols, key=lambda x: x[1])
                        composition = ''.join([c[0] for c in comp])
                        if self.surface == 'fcc110' and geometry == 'terrace':
                            extrasymbols = [(self.atoms[i].symbol,
                                             self.atoms[i].number) 
                                             for i in extraids]
                            extra = sorted(extrasymbols, key=lambda x: x[1])
                            extracomp = ''.join([e[0] for e in extra])
                            composition += '-{}'.format(extracomp)
                        site.update({'composition': composition})
                    sl.append(site)
                    usi.add(si)

                if self.surface in ['fcc100', 'fcc211', 'fcc311'] \
                and fold4_poss:
                  #  print(fold4_poss)
                    fold4atoms = Atoms('X{}'.format(len(fold4_poss)), 
                                       positions=np.asarray(fold4_poss),
                                       cell=self.cell, 
                                       pbc=self.pbc)
                    sorted_top = sorted(self.surf_ids)
                    ntop = len(sorted_top)
                    topatoms = self.refatoms[sorted_top] 
                    newatoms = topatoms.extend(fold4atoms)
                    newnblist = PeriodicNeighborList(newatoms, dx=.1, 
                                                     neighbor_number=2, 
                                                     different_species=True) 

                    # Make 4-fold hollow sites
                    for i, refpos in enumerate(fold4_poss):
                        fold4_indices = newnblist[ntop+i]                     
                        fold4ids = [sorted_top[j] for j in fold4_indices]
                        occurence = np.sum(cm[fold4ids], axis=0)
                        isub = np.where(occurence == 4)[0][0]
                        newfold4ids = [k for k in fold4ids if cm[k,isub] == 1]
                        si = tuple(sorted(newfold4ids))
                        pos = refpos + np.average(
                              self.delta_positions[newfold4ids], 0)
                        site = self.new_site() 
                        site.update({'site': '4fold',               
                                     'surface': self.surface,
                                     'geometry': '100',
                                     'position': pos,
                                     'normal': np.array([0,0,1]),
                                     'indices': si})                     
                        if self.show_composition:                        
                            metals = list(set(self.atoms.symbols))      
                            ma, mb = metals[0], metals[1]
                            if atomic_numbers[ma] > atomic_numbers[mb]:
                                ma, mb = metals[1], metals[0]
                            symbols = [self.atoms[i].symbol for i in si]
                            nma = symbols.count(ma)
                            if nma == 0:
                                composition = 4*mb
                            elif nma == 1:
                                composition = ma + 3*mb
                            elif nma == 2:
                                opposite = np.where(
                                           cm[si[1:],si[0]]==0)[0]
                                opp = opposite[0] + si[1]         
                                if self.atoms[opp].symbol == \
                                self.atoms[newfold4ids[0]].symbol:
                                    composition = ma + mb + ma + mb 
                                else:
                                    composition = 2*ma + 2*mb 
                            elif nma == 3:
                                composition = 3*ma + mb
                            elif nma == 4:
                                composition = 4*ma
                            site.update({'composition': composition})   
                        if self.show_subsurface:
                            site.update({'subsurface_id': isub})
                            if self.show_composition:
                                site.update({'subsurface_element': 
                                             self.atoms[isub].symbol})
                        sl.append(site)
                        usi.add(si)
             
            # Make 3-fold hollow sites (differentiate fcc / hcp)
            if n == 1 and self.surface != 'fcc100':
                for i, refpos in enumerate(reduced_poss):
                    fold3_indices = nblist[ntop2+i]
                    fold3ids = [sorted_indices[j] for j in fold3_indices]
                    if len(fold3ids) != 3:
                        continue
                    si = tuple(sorted(fold3ids))
                    pos = refpos + np.average(
                          self.delta_positions[fold3ids], 0)

                    occurence = np.sum(cm[fold3ids], axis=0)
                    sumo = np.sum(occurence)
                    if self.surface == 'fcc211':
                        if sumo == 23:
                            sitetype, geometry = 'hcp', 'upper111'        
                        elif sumo in [24, 25]:
                            sitetype, geometry = 'fcc', 'upper111'
                        elif sumo in [27, 28]:
                            sitetype, geometry = 'hcp', 'lower111'
                        elif sumo == 29:
                            sitetype, geometry = 'fcc', 'lower111'                    
                        else:
                            print(sumo)
                            print('Cannot identify site {}'.format(si))
                            continue         
                    else: 
                        if np.max(occurence) == 3:
                            sitetype, geometry = 'hcp', '111'                  
                        else:
                            sitetype, geometry = 'fcc', '111' 
 
                    site = self.new_site()
                    site.update({'site': sitetype,
                                 'surface': self.surface,
                                 'geometry': geometry,
                                 'position': pos,
                                 'normal': np.array([0,0,1]),
                                 'indices': si})
                    if self.show_composition:                       
                        metals = list(set(self.atoms.symbols))
                        ma, mb = metals[0], metals[1]
                        if atomic_numbers[ma] > atomic_numbers[mb]:
                            ma, mb = metals[1], metals[0]
                        symbols = [self.atoms[i].symbol for i in si]
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

                    if sitetype == 'hcp' and self.show_subsurface:
                        isub = np.where(occurence == 3)[0][0]
                        site.update({'subsurface_id': isub})
                        if self.show_composition:
                            site.update({'subsurface_element': 
                                         self.atoms[isub].symbol})
                    sl.append(site)
                    usi.add(si)
            
            if n == 2 and self.surface == 'fcc100' and list(reduced_poss):
                fold4atoms = Atoms('X{}'.format(len(reduced_poss)), 
                                   positions=np.asarray(reduced_poss),
                                   cell=self.cell, 
                                   pbc=self.pbc)
                sorted_top = sorted(self.surf_ids)
                ntop = len(sorted_top)
                topatoms = self.refatoms[sorted_top] 
                newatoms = topatoms.extend(fold4atoms)
                newnblist = PeriodicNeighborList(newatoms, dx=.1, 
                                                 neighbor_number=2, 
                                                 different_species=True) 

                for i, refpos in enumerate(reduced_poss):
                    fold4_indices = newnblist[ntop+i]                     
                    fold4ids = [sorted_top[j] for j in fold4_indices]
                    occurence = np.sum(cm[fold4ids], axis=0)
                    isub = np.where(occurence == 4)[0][0]
                    newfold4ids = [k for k in fold4ids if cm[k,isub] == 1]
                    si = tuple(sorted(newfold4ids))
                    pos = refpos + np.average(
                          self.delta_positions[newfold4ids], 0)                    
                    site = self.new_site()
                    site.update({'site': '4fold',
                                 'surface': self.surface,
                                 'geometry': '100',
                                 'position': pos,
                                 'normal': np.array([0,0,1]),
                                 'indices': si})                    
                    if self.show_composition:                       
                        metals = list(set(self.atoms.symbols))      
                        ma, mb = metals[0], metals[1]
                        if atomic_numbers[ma] > atomic_numbers[mb]:
                            ma, mb = metals[1], metals[0]
                        symbols = [self.atoms[i].symbol for i in si]
                        nma = symbols.count(ma)
                        if nma == 0:
                            composition = 4*mb
                        elif nma == 1:
                            composition = ma + 3*mb
                        elif nma == 2:
                            opposite = np.where(
                                       cm[si[1:],si[0]]==0)[0]
                            opp = opposite[0] + si[1]         
                            if self.atoms[opp].symbol == \
                            self.atoms[newfold4ids[0]].symbol:
                                composition = ma + mb + ma + mb 
                            else:
                                composition = 2*ma + 2*mb 
                        elif nma == 3:
                            composition = 3*ma + mb
                        elif nma == 4:
                            composition = 4*ma
                        site.update({'composition': composition})   
                    if self.show_subsurface:
                        site.update({'subsurface_id': isub})
                        if self.show_composition:
                            site.update({'subsurface_element': 
                                         self.atoms[isub].symbol})
                    sl.append(site)
                    usi.add(si)

        index_list, pos_list, st_list = [], [], []
        for t in sl:
            stids = t['indices']
            # Take care of duplicate indices. When unit cell is small,
            # different sites can have exactly same indices
            if t['site'] in ['fcc', 'hcp']:
                if stids in index_list:
                    slid = next(si for si in range(len(sl)) if 
                                sl[si]['indices'] == stids)
                    previd = index_list.index(stids)
                    prevpos = pos_list[previd]
                    prevst = st_list[previd]
                    if min([np.linalg.norm(prevpos - pos) for pos 
                       in self.atoms[self.subsurf_ids].positions]) < \
                       min([np.linalg.norm(t['position'] - pos) for pos
                       in self.atoms[self.subsurf_ids].positions]):
                        t['site'] = 'fcc'
                        if prevst == 'fcc':
                            sl[slid]['site'] = 'hcp'
                        if self.show_subsurface:
                            t['subsurface_id'] = None
                            if self.show_composition:
                                t['subsurface_element'] = None 
                    else:
                        t['site'] = 'hcp'
                        if t['site'] == 'hcp':
                            sl[slid]['site'] = 'fcc'
                        if self.show_subsurface:
                            sl[slid]['subsurface_id'] = None
                            if self.show_composition:
                                sl[slid]['subsurface_element'] = None
                else:
                    index_list.append(t['indices'])
                    pos_list.append(t['position'])
                    st_list.append(t['site'])

      #TODO: Correct normal vectors for fcc110, 211 and 311
       # if self.surface in ['fcc110', 'fcc211', 'fcc311']:
          #  geometries = set([s['geometry'] for s in sl])
          #  geo_normal_dict = {}
          #  for geo in geometries:
          #      gindices = list(set(sum([s['indices'] for s in sl 
          #                  if s['geometry'] == geo], ())))
          #      gpositions = self.atoms.positions[gindices]
          #      gvec = get_plane_normal(gpositions)
          #      geo_normal_dict[geo] = gvec
          #  print(geo_normal_dict)
       #     for st in sl:
       #         gindices = list(st['indices'])
       #         gpositions = self.atoms.positions[gindices]
       #         gvec = get_plane_normal(gpositions)
       #         st['normal'] = gvec


    def new_site(self):
        return {'site': None, 'surface': None, 'geometry': None, 
                'position': None, 'normal': None, 'indices': None,
                'composition': None, 'subsurface_id': None,
                'subsurface_element': None}

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

    def make_neighbor_list(self, dx=0.3, neighbor_number=1):
        """Generate a periodic neighbor list (defaultdict).""" 
        self.nlist = PeriodicNeighborList(self.refatoms, dx, neighbor_number)

    def get_connectivity(self):
        """Generate a connections matrix from PeriodicNeighborList."""       
        return get_connectivity_matrix(self.nlist)

    def get_termination(self):
        """Return lists surf and subsurf containing atom indices belonging to
        those subsets of a surface atoms object.
        This function relies on PTM and the connectivity of the atoms.
        """
    
        xcell = self.cell[0][0]
        ycell = self.cell[1][1] 
        xmul = math.ceil(12/xcell)
        ymul = math.ceil(12/ycell) 
        atoms = self.atoms*(xmul,ymul,1)

        cm = self.connectivity_matrix.copy()                               
        np.fill_diagonal(cm, 0)
        indices = self.indices 

        ptmdata = PTM(atoms, rmsd_max=0.25)
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
                    self.refatoms.positions[list(x),2])))
        subsurf = []
        for a_b in bulk:
            for a_t in surf:
                if cm[a_t, a_b] > 0:
                    subsurf.append(a_b)
        subsurf = list(np.unique(subsurf))
        return sorted(surf), sorted(subsurf)

    def get_CNA(self, rCut=None):                  
        atoms = self.refatoms.copy()
        _, extcoords, _ = expand_cell(atoms, cutoff=5.0)
        fracs = np.dot(extcoords, np.linalg.pinv(self.cell))
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
                          cell = self.cell,
                          pbc = self.pbc)
        fullatoms = atoms.extend(ringatoms) 
        fullatoms.center(vacuum=5.)            
        fullCNA = FullCNA(fullatoms, rCut=rCut).get_normal_cna() 
        surfCNA = [fullCNA[i] for i in self.surf_ids]
        return fullCNA, surfCNA
                 
    def identify_surface(self):
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
                         unique_subsurface=False):
        sl = self.site_list
        key_list = ['site', 'geometry']
        if unique_composition:
            if not self.show_composition:
                raise ValueError('The site list does not include \
                                  information of composition')
            key_list.append('composition')
            if unique_subsurface:
                if not self.show_subsurface:
                    raise ValueError('The site list does not include \
                                      information of subsurface')
                key_list.append('subsurface_element') 
        else:
            if unique_subsurface:
                raise ValueError('To include the subsurface element, \
                                  unique_composition also need to \
                                  be set to True')    
        name_list = [[s[k] for k in key_list] for s in sl]
        name_list.sort()
 
        return list(name_list for name_list,_ in groupby(name_list))  


def enumerate_adsorption_sites(atoms, surface=None, geometry=None, 
                               show_composition=False, 
                               show_subsurface=False):

    if True not in atoms.pbc:
        nas = NanoparticleAdsorptionSites(atoms, show_composition,
                                          show_subsurface)
        all_sites = nas.site_list
        if surface:
            all_sites = [s for s in all_sites if s['surface'] == surface] 

    else:
        sas = SlabAdsorptionSites(atoms, surface, show_composition, 
                                  show_subsurface)
        all_sites = sas.site_list
        if geometry:
            all_sites = [s for s in all_sites if s['geometry'] == geometry]       

    return all_sites
