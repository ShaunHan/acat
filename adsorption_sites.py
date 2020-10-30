from .utilities import PeriodicNeighborList, get_connectivity_matrix
from .utilities import get_mic_distance, expand_cell
from asap3.analysis.rdf import RadialDistributionFunction
from asap3 import FullNeighborList
from asap3.analysis import FullCNA
from ase.build import molecule
from ase.geometry import Cell, get_layers
from ase import Atom, Atoms
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from operator import itemgetter
import numpy as np
from itertools import combinations
from ase.data import reference_states, covalent_radii
from ase.io import read, write
from collections import Counter, defaultdict
import networkx as nx
import scipy
import re
import random
import warnings

#warnings.filterwarnings('ignore')


icosa_dict = {
    # Triangle sites on outermost shell -- Icosa, Cubocta, Deca, Tocta
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': str({(3, 1, 1): 6, (4, 2, 1): 3}),
    # Edge sites on outermost shell -- Icosa
    str({(3, 1, 1): 4, (3, 2, 2): 2, (4, 2, 2): 2}): 'edge',
    'edge': str({(3, 1, 1): 4, (3, 2, 2): 2, (4, 2, 2): 2}),
    # Vertice sites on outermost shell -- Icosa, Deca
    str({(3, 2, 2): 5, (5, 5, 5): 1}): 'vertex',
    'vertex': str({(3, 2, 2): 5, (5, 5, 5): 1}),
}

ticosa_dict = {
    # Triangle sites on outermost shell -- Icosa, Cubocta, Deca, Tocta
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': str({(3, 1, 1): 6, (4, 2, 1): 3}),
    # Edge sites on outermost shell -- Icosa
    str({(3, 1, 1): 4, (3, 2, 2): 2, (4, 2, 2): 2}): 'edge',
    'edge': str({(3, 1, 1): 4, (3, 2, 2): 2, (4, 2, 2): 2}),
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
    'edge': str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}),
    # Square sites on outermost shell -- Cubocta, Deca, Tocta, (Surface)
    str({(2, 1, 1): 4, (4, 2, 1): 4}): 'fcc100',
    'fcc100': str({(2, 1, 1): 4, (4, 2, 1): 4}),
    # Vertice sites on outermost shell -- Cubocta
    str({(2, 1, 1): 4, (4, 2, 1): 1}): 'vertex',
    'vertex': str({(2, 1, 1): 4, (4, 2, 1): 1}),
    # Triangle sites on outermost shell -- Icosa, Cubocta, Deca, Tocta, (Surface)
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': str({(3, 1, 1): 6, (4, 2, 1): 3}),
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
    'fcc100': str({(2, 1, 1): 4, (4, 2, 1): 4}),
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
    'edge': str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}),
    # Vertice sites on outermost shell -- Octa
    str({(2, 0, 0): 4}): 'vertex',
    'vertex': str({(2, 0, 0): 4}),
    # Triangle (pentagon) sites on outermost shell -- Icosa, Cubocta, Deca, Octa
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': str({(3, 1, 1): 6, (4, 2, 1): 3}),
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
    'fcc100': str({(2, 1, 1): 4, (4, 2, 1): 4}),
    # Vertice sites on outermost shell -- Tocta
    str({(2, 0, 0): 1, (2, 1, 1): 2, (3, 1, 1): 2, (4, 2, 1): 1}): 'vertex',
    'vertex': str({(2, 0, 0): 1, (2, 1, 1): 2, (3, 1, 1): 2, (4, 2, 1): 1}),
    # Triangle (pentagon) sites on outermost shell -- Icosa, Cubocta, Deca, Octa
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': str({(3, 1, 1): 6, (4, 2, 1): 3}),
}

surf_dict = {
    # Square sites
    str({(2, 1, 1): 4, (4, 2, 1): 4}): 'fcc100',
    'fcc100': str({(2, 1, 1): 4, (4, 2, 1): 4}),
    # Triangle sites
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': str({(3, 1, 1): 6, (4, 2, 1): 3}),
    # Triangle-Triangle sites
    str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}): 'fcc110',
    'fcc110': [str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}), 
               str({(3, 1, 1): 4, (4, 2, 1): 7})],
    # Triangle-Triangle sites
    str({(3, 1, 1): 4, (4, 2, 1): 7}): 'fcc110',
    # Triangle-Square sites
    str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}): 'fcc311',
    'fcc311': [str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}), 
               str({(2, 1, 1): 1, (3, 1, 1): 4, (4, 2, 1): 5})],
    # Triangle-Square sites
    str({(2, 1, 1): 1, (3, 1, 1): 4, (4, 2, 1): 5}): 'fcc311',
}


# Set global variables
adsorbates = 'SCHON'
heights_dict = {'ontop': 2.0, 
                'bridge': 1.8, 
                'fcc': 1.8, 
                'hcp': 1.8, 
                '4fold': 1.7}


class NanoparticleAdsorptionSites(object):

    def __init__(self, atoms, heights=None, include_subsurface=False):

        assert True not in atoms.pbc
        atoms = atoms.copy()
        del atoms[[a.index for a in atoms if 'a' not in reference_states[a.number]]]
        del atoms[[a.index for a in atoms if a.symbol in adsorbates]]
        self.atoms = atoms
        self.include_subsurface = include_subsurface

        self.fullCNA = {}
        self.make_fullCNA()
        self.set_first_neighbor_distance_from_rdf()
        self.site_dict = self.get_site_dict()
        self.make_neighbor_list()
        self.surf_sites = self.get_surface_sites()

        if heights is None:
            self.heights = heights_dict                        
        else:
            self.heights = heights

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
                                     'height': self.heights['bridge'],
                                     'indices': si})
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

                            if this_site == 'hcp' and self.include_subsurface:
                                hcp_nbr_indices = []
                                for i in [s, n, m]:
                                    indices = self.nlist.get_neighbors(i, 
                                                              self.r + 1.)
                                    for idx in indices:
                                        hcp_nbr_indices.append(idx)
                                subi = [key for key, count in \
                                        Counter(hcp_nbr_indices).items() \
                                        if count == 3][0]                      
                            else:
                                subi = None                   

                            site = self.new_site()
                            site.update({'site': this_site,
                                         'surface': site_surf,
                                         'position': pos,
                                         'normal': normal,
                                         'height': self.heights[this_site],
                                         'indices': si,
                                         'subsurface_id': subi})
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
                                        
                                            if self.include_subsurface:
                                                fold4_nbr_indices = []
                                                for i in [s, n, m]:
                                                    indices = self.nlist.get_neighbors(i, 
                                                                             self.r + 1.)
                                                    for idx in indices:
                                                        fold4_nbr_indices.append(idx)
                                                subi = [key for key, count in \
                                                        Counter(fold4_nbr_indices).items() \
                                                        if count == 4][0]
                                            else:
                                                subi = None

                                            site = self.new_site()
                                            site.update({'site': '4fold',
                                                         'surface': site_surf,
                                                         'position': pos,
                                                         'normal': normal,
                                                         'height': self.heights['4fold'],
                                                         'indices': si,
                                                         'subsurface_id':subi})
                                            sl.append(site)
                                            usi.add(si)

                # ontop sites
                site = self.new_site()
                site.update({'site': 'ontop', 'surface': surface,
                             'position': self.atoms[s].position,
                             'height': self.heights['ontop'], 'indices': (s,)})
                # Find normal
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
        return {'site': None, 'surface': None, 'height': None,
                'position': None, 'normal': None, 'indices': None,
                'subsurface_id': None, 'occupied': 0}

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
            self.fullCNA[rCut] = FullCNA(
                self.atoms, rCut=rCut).get_normal_cna()

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
        atoms = self.atoms
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


class SlabAdsorptionSites(object):

    def __init__(self, atoms, surface=None, heights=None, 
                       include_subsurface=False, tol=1e-5):

        assert True in atoms.pbc    
        atoms = atoms.copy() 
        del atoms[[a.index for a in atoms if 'a' not in 
                               reference_states[a.number]]]
        del atoms[[a.index for a in atoms if a.symbol in adsorbates]]
        self.atoms = atoms
        self.cell = atoms.cell
        self.pbc = atoms.pbc
        self.tol = tol
        self.include_subsurface = include_subsurface
        self.site_dict = surf_dict 
        self.set_first_neighbor_distance_from_rdf()

        self.make_neighbor_list(dx=self.r*.1)
        self.connectivity_matrix = self.get_connectivity()         
        if surface is None: 
            self.fullCNA, self.surfCNA = self.get_CNA(rCut=self.r+.6)
            self.surface = self.identify_surface()
        else:
            self.surface = surface
        if heights is None:
            self.heights = heights_dict                          
        else:
            self.heights = heights

        self.surf_ids, self.subsurf_ids = self.get_termination()        
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
        normals_for_site = dict(list(zip(top_indices, [[] for 
                                          _ in top_indices])))
        cm = self.connectivity_matrix.copy() 

        for s in top_indices:
            occurence = np.sum(cm[s], axis=0)
            if self.surface in ['fcc100','fcc111']:
                geometry = 'terrace'
            else:
                if occurence == 7:
                    geometry = 'step'
                elif occurence in [9, 11]:
                    geometry = 'terrace'
                elif occurence == 10:
                    geometry = 'corner'
            si = (s,)
            site = self.new_site()
            site.update({'site': 'ontop',
                         'surface': self.surface,
                         'geometry': geometry,
                         'position': self.atoms.positions[s],
                         'height': self.heights['ontop'],
                         'indices': si})
            sl.append(site)
            usi.add(si)

        ext_index, ext_coords, _ = expand_cell(self.atoms, cutoff=5.0)
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
        for n, poss in enumerate([bridge_positions, 
                                  fold3_positions, 
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
            top2atoms = self.atoms[sorted_indices]

            # Make a new neighborlist including sites as dummy atoms
            dummies = Atoms('X{}'.format(reduced_poss.shape[0]), 
                            positions=reduced_poss, 
                            cell=self.cell, 
                            pbc=self.pbc)

            ntop2 = len(top2atoms)
            testatoms = top2atoms.extend(dummies)
            nblist = PeriodicNeighborList(testatoms, 
                                          neighbor_number=1, 
                                          different_species=True) 
            
            if n == 0:
                # Make bridge sites
                new_fold4_poss = []
                for i, pos in enumerate(reduced_poss):
                    bridge_indices = nblist[ntop2+i]                     
                    bridgeids = [sorted_indices[j] for j in bridge_indices]
                    if len(bridgeids) != 2: 
                        if self.surface in ['fcc211', 'fcc311']:
                            new_fold4_poss.append(pos)
                        continue

                    si = tuple(sorted(bridgeids)) 
                    occurence = np.sum(cm[bridgeids], axis=0)
                    sumo = np.sum(occurence)
                    if self.surface == 'fcc110':
                        if list(occurence).count(2) == 2:
                            geometry = 'step'
                        elif list(occurence).count(2) == 3:
                            geometry = 'step111'
                        elif list(occurence).count(2) == 4:
                            geometry = 'terrace'

                    elif self.surface == 'fcc311': 
                        if sumo == 14:
                            geometry = 'step'
                        elif sumo == 17:
                            if list(occurence).count(2) == 2:
                                geometry = 'step100'
                            elif list(occurence).count(2) == 3:
                                geometry = 'step111'
                        elif sumo == 20:
                            geometry = 'terrace'
 
                    elif self.surface == 'fcc211':                    
                        if sumo == 14:
                            geometry = 'step'
                        elif sumo == 16:
                            geometry = 'step111'
                        elif sumo == 17:
                            geometry = 'step100'
                        elif sumo == 18:
                            geometry = 'terrace'
                        elif sumo == 19:
                            geometry = 'terrace111'
                        elif sumo == 20:
                            geometry = 'corner'
                    else:
                        geometry = 'terrace'
                                    
                    site = self.new_site()                         
                    site.update({'site': 'bridge',               
                                 'surface': self.surface,
                                 'geometry': geometry,
                                 'position': pos,
                                 'height': self.heights['bridge'],
                                 'indices': si})           
                    sl.append(site)
                    usi.add(si)

                if self.surface in ['fcc211', 'fcc311']:
                    fold4atoms = Atoms('X{}'.format(len(new_fold4_poss)), 
                                       positions=np.asarray(new_fold4_poss),
                                       cell=self.cell, 
                                       pbc=self.pbc)
                    sorted_top = sorted(self.surf_ids)
                    ntop = len(sorted_top)
                    topatoms = self.atoms[sorted_top] 
                    newatoms = topatoms.extend(fold4atoms)
                    newnblist = PeriodicNeighborList(newatoms, 
                                                     dx=self.r*.02, 
                                                     neighbor_number=2, 
                                                     different_species=True) 

                    for i, pos in enumerate(new_fold4_poss):
                        fold4_indices = newnblist[ntop+i]                     
                        fold4ids = [sorted_top[j] for j in fold4_indices]
                        occurence = np.sum(cm[fold4ids], axis=0)
                        isub = np.where(occurence == 4)[0][0]
                        newfold4ids = [k for k in fold4ids if cm[k,isub] == 1]
                        si = tuple(sorted(newfold4ids))
                        normal = self.get_surface_normal(list(si))
                        for i in si:
                            normals_for_site[i].append(normal)
                        site = self.new_site()                        
                        site.update({'site': '4fold',               
                                     'surface': self.surface,
                                     'geometry': 'step100',
                                     'position': pos,
                                     'normal': normal,
                                     'height': self.heights['4fold'],
                                     'indices': si})          
                        sl.append(site)
                        usi.add(si)
                        if self.include_subsurface:
                            site.update({'subsurface_id': isub})
             
            if n == 1 and self.surface != 'fcc100':
                # Make 3-fold hollow sites (differentiate fcc / hcp)
                for i, pos in enumerate(reduced_poss):
                    fold3_indices = nblist[ntop2+i]
                    fold3ids = [sorted_indices[j] for j in fold3_indices]
                    if len(fold3ids) != 3:
                        continue
                    si = tuple(sorted(fold3ids))
                    normal = self.get_surface_normal(list(si))
                    for i in si:
                        normals_for_site[i].append(normal)

                    occurence = np.sum(cm[fold3ids], axis=0)
                    sumo = np.sum(occurence)
                    if self.surface == 'fcc211':
                        if sumo == 23:
                            sitetype, geometry = 'hcp', 'step111'               
                        elif sumo == 25:
                            sitetype, geometry = 'fcc', 'step111'                    
                        elif sumo == 28:
                            sitetype, geometry = 'hcp', 'terrace111'
                        elif sumo == 29:
                            sitetype, geometry = 'fcc', 'terrace111'                
                    else: 
                        if np.max(occurence) == 3:
                            sitetype, geometry = 'hcp', 'terrace111'                  
                        else:
                            sitetype, geometry = 'fcc', 'terrace111'                        

                    site = self.new_site()                      
                    site.update({'site': sitetype,
                                 'surface': self.surface,
                                 'geometry': geometry,
                                 'position': pos,
                                 'normal': normal,
                                 'height': self.heights['fcc'],
                                 'indices': si})
                    if sitetype == 'hcp' and self.include_subsurface:
                        isub = np.where(occurence == 3)[0][0]
                        site.update({'subsurface_id': isub})
                    sl.append(site)
                    usi.add(si)
 
            if n == 2 and self.surface == 'fcc100':
                # Make 4-fold hollow sites
                for i, pos in enumerate(reduced_poss):
                    fold4_indices = nblist[ntop2+i]
                    fold4ids = [sorted_indices[i] for i in fold4_indices]
                    if len(fold4ids) != 4:
                        continue
                    si = tuple(sorted(fold4ids))
                    normal = self.get_surface_normal(list(si))
                    for i in si:
                        normals_for_site[i].append(normal)
                    site = self.new_site()                           
                    site.update({'site': '4fold',
                                 'surface': self.surface,
                                 'geometry': 'terrace100',
                                 'position': pos,
                                 'normal': normal,
                                 'height': self.heights['4fold'],
                                 'indices': si})
                    if self.include_subsurface:                        
                        occurence = np.sum(cm[fold4ids], axis=0)
                        isub = np.where(occurence == 4)[0][0]
                        site.update({'subsurface_id': isub})
                    sl.append(site)
                    usi.add(si)

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
        return {'site': None, 'surface': None, 'height': None,
               'position': None, 'normal': None, 'indices': None,
               'subsurface_id': None, 'occupied': 0}

    def make_neighbor_list(self, dx=0.3, neighbor_number=1):
        """Generate a periodic neighbor list (defaultdict).""" 
        self.nlist = PeriodicNeighborList(self.atoms, dx, neighbor_number)

    def get_connectivity(self):
        """Generate a connections matrix from PeriodicNeighborList."""       
        return get_connectivity_matrix(self.nlist)

    def get_termination(self):
        """Return lists surf and subsurf containing atom indices belonging to
        those subsets of a surface atoms object.
        This function relies on the connectivity of the atoms.
        """

        cm = self.connectivity_matrix.copy()                               
        np.fill_diagonal(cm, 0)
        indices = [a.index for a in self.atoms] 
        coord = np.count_nonzero(cm[indices,:][:,indices], axis=1)
        allsurf = []
        bulk = []
        max_coord = np.max(coord)
        dcoord = 0 if self.surface == 'fcc110' else 1
        for i, c in enumerate(coord):
            a_s = indices[i]
            if c >= max_coord - dcoord: 
                bulk.append(a_s)
            else:
                allsurf.append(a_s)
        notsurf = [a.index for a in self.atoms if a.index not in allsurf]
        surfcm = cm.copy()
        surfcm[notsurf] = 0
        surfcm[:,notsurf] = 0
        rows, cols = np.where(surfcm == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        G.add_edges_from(edges)
        components = nx.connected_components(G)
        surf = list(max(components, 
                    key=lambda x:np.mean(self.atoms.positions[list(x),2])))
        subsurf = []
        for a_b in bulk:
            for a_t in surf:
                if cm[a_t, a_b] > 0:
                    subsurf.append(a_b)
        subsurf = list(np.unique(subsurf))
        return sorted(surf), sorted(subsurf)

    def get_two_vectors(self, sites):
        p1 = self.atoms[int(sites[1])].position
        p2 = self.atoms[int(sites[2])].position
        vec1 = p1 - self.atoms[int(sites[0])].position
        vec2 = p2 - self.atoms[int(sites[0])].position
        return vec1, vec2

    def get_surface_normal(self, sites):
        vec1, vec2 = self.get_two_vectors(sites)
        n = np.cross(vec1, vec2)
        l = np.sqrt(np.dot(n, n.conj()))
        new_pos = self.atoms[sites[0]].position + self.r * n / l
        # Add support for having adsorbates on the particles already
        # by putting in elements to check for in the function below
        j = 2 * int(self.no_atom_too_close_to_pos(new_pos, (5./6)*self.r)) - 1
        return j * n / l

    def no_atom_too_close_to_pos(self, pos, mindist):              
        """Returns True if no atoms are closer than mindist to pos,
        otherwise False."""
        dists = [np.linalg.norm(atom.position - pos) > mindist
                 for atom in self.atoms]
        return all(dists)

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

    def get_CNA(self, rCut=None):                  
        atoms = self.atoms.copy()
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

        cm = self.connectivity_matrix.copy()                               
        np.fill_diagonal(cm, 0)
        indices = [a.index for a in self.atoms] 
        coord = np.count_nonzero(cm[indices,:][:,indices], axis=1)
        allsurf = []
        bulk = []
        max_coord = np.max(coord)
        for i, c in enumerate(coord):
            a_s = indices[i]
            if c == max_coord: 
                bulk.append(a_s)
            else:
                allsurf.append(a_s)
        notsurf = [a.index for a in self.atoms if a.index not in allsurf]
        surfcm = cm.copy()
        surfcm[notsurf] = 0
        surfcm[:,notsurf] = 0
        rows, cols = np.where(surfcm == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        G.add_edges_from(edges)
        components = nx.connected_components(G)
        surf = list(max(components, 
                    key=lambda x:np.mean(self.atoms.positions[list(x),2])))

        surfCNA = [fullCNA[i] for i in surf]
        return fullCNA, surfCNA
                 
    def identify_surface(self):
        sd = self.site_dict
        scna = self.surfCNA
        fcc100_weight = fcc111_weight = fcc110_weight = \
        fcc211_weight = fcc311_weight = 0
        for s in scna:
            found = 0
            if str(s) in sd['fcc100']:
                fcc100_weight += 1
                found += 1
            if str(s) in sd['fcc111']:
                fcc111_weight += 1
                found += 1
            if str(s) in sd['fcc110']:
                fcc110_weight += 1
                found += 1
            if str(s) in sd['fcc311']:
                fcc311_weight += 1
                found += 1
            if found == 0:
                fcc211_weight += 1
 
        full_weights = [fcc100_weight, fcc111_weight, fcc110_weight,
                        fcc211_weight, fcc311_weight]
        if len([w for w in full_weights if w == max(full_weights)]) > 1:
            raise ValueError('Cannot identify the surface. Please specify it!')
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


















def get_neighbors_from_position(atoms, position, cutoff=3.):
    '''Get a neighbor list from a position'''

    cell = atoms.get_cell()
    pbc = np.array([cell[0][0], cell[1][1], 0])
    lst = []
    for a in atoms:
        d = get_mic_distance(position, a.position, atoms.cell, atoms.pbc)
        if d < cutoff:
            lst.append((a.index, d))
    return lst


def add_adsorbate(atoms, adsorbate, site):
    # Make the correct position
    height = site['height']
    normal = np.array(site['normal'])
    pos = np.array(site['position']) + normal * height

    ads = adsorbate.copy()
    if len(ads) > 1:
        avg_pos = np.average(ads[1:].positions, 0)
        ads.rotate(avg_pos - ads[0].position, normal)
        #pvec = np.cross(np.random.rand(3) - ads[0].position, normal)
        #ads.rotate(-45, pvec, center=ads[0].position)
    ads.translate(pos - ads[0].position)

    atoms.extend(ads)


def monometallic_add_adsorbate(atoms, adsorbate, site, surface=None, height=None, 
                               nsite='all', min_adsorbate_distance=1.):
    """A function for adding adsorbate to a specific adsorption site on a 
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

    nsite: The number of such adsorption site that is attached with the adsorbate. 
        Default is 1. Set nsite = 'all' to attach the adsorbate to all such sites.

    min_adsorbate_distance: The minimum distance between two adsorbate atoms.
        Default value 1. is good in most cases. Play around to find the best value.
    
    Example
    ------- 
    monometallic_add_adsorbate(atoms,adsorbate='CO',site='4fold',surface='fcc100',
                               nsite='all')"""

    atoms.info['data'] = {}        

    if True not in atoms.pbc:
        if surface is None:
            raise ValueError('Surface must be specified for a nanoparticle')
        if height is None:
            ads = NanoparticleAdsorptionSites(atoms)
        else:
            heights_dict[site] = height
            ads = NanoparticleAdsorptionSites(atoms, heights=heights_dict)
        nl = ads.nlist    
        sites  = ads.get_sites_from_surface(site, surface)
        for site in special_sites:
            sites.append(site)
#        if not sites:
#            print('No such adsorption site found on this nanoparticle.')
        if adsorbate == 'CO':
            if nsite == 'all':
                for site in sites:                                            
                    add_adsorbate(atoms, molecule(adsorbate)[::-1], site)                
                atom_indices = [a.index for a in atoms if a.symbol == 'O']            
                n_ads_atoms = 2
                overlap_atoms_indices = []
                for idx in atom_indices:   
                    neighbor_indices = nl.get_neighbors(idx, min_adsorbate_distance)
                    overlap = 0
                    for i in neighbor_indices:
                        if (atoms[i].symbol in adsorbates) and \
                           (i not in overlap_atoms_indices):
                            overlap += 1
                    if overlap > 0:
                        overlap_atoms_indices += list(set([idx-n_ads_atoms+1, idx]))
                del atoms[overlap_atoms_indices]
            else:
                final_sites = random.sample(sites, nsite)
                for site in final_sites:
                    add_adsorbate(atoms, molecule(adsorbate)[::-1], site)
        else:
            if nsite == 'all':            
                for site in sites:
                    add_adsorbate(atoms, molecule(adsorbate), site)
                atom_indices = [a.index for a in atoms if a.symbol == adsorbate[-1]]
                ads_symbols = molecule(adsorbate).get_chemical_symbols()
                n_ads_atoms = len(ads_symbols)
                overlap_atoms_indices = []
                for idx in atom_indices:   
                    neighbor_indices = nl.get_neighbors(idx, min_adsorbate_distance)
                    overlap = 0
                    for i in neighbor_indices:                                                                
                        if (atoms[i].symbol in adsorbates) and \
                           (i not in overlap_atoms_indices):                       
                            overlap += 1                                                                      
                    if overlap > 0:                                                                           
                        overlap_atoms_indices += list(set([idx-n_ads_atoms+1, idx]))                                
                del atoms[overlap_atoms_indices]                                                                    
            else:
                final_sites = random.sample(sites, nsite)
                for site in final_sites:
                    add_adsorbate(atoms, molecule(adsorbate), site)

    else:
        if height is None:
            hdiff = 0
        else:
            hdiff = height-2.
        ads_indices = [a.index for a in atoms if a.symbol in adsorbates]
        ads_atoms = None
        if ads_indices:
            ads_atoms = atoms[ads_indices]
            atoms = atoms[[a.index for a in atoms if a.symbol not in adsorbates]]
        top_indices = []
        second_indices = []
        layerlist = get_layers(atoms, (0,0,1), tolerance=0.5)[0].tolist()
        nlayers = max(layerlist)
        for atom in atoms:
            i = atom.index
            if layerlist[i] == nlayers:
                top_indices.append(i)
            elif layerlist[i] == nlayers-1:
                second_indices.append(i)
        layer_thickness = abs(np.mean(atoms.positions[top_indices,2]) \
                              - np.mean(atoms.positions[second_indices,2]))
        d100 = np.sqrt(2)*layer_thickness 
        d111 = np.sqrt(6)*layer_thickness/2

        if surface is None: 
            surface = identify_surface(atoms[top_indices+second_indices]) 
        dummy = atoms.copy()
        #Expand cell for small unit cell
        if (surface == 'fcc100' and len(top_indices) < 8)\
        or (surface == 'fcc111' and len(top_indices) < 4):            
            xcell = dummy.cell[0][0]
            ycell = dummy.cell[1][1]
            atoms *= (2,2,1)

        struct = AseAtomsAdaptor.get_structure(atoms)
        asf = AdsorbateSiteFinder(struct)
        ads_sites = asf.find_adsorption_sites(symm_reduce=0)
        ads = molecule(adsorbate)[::-1]
        if str(ads.symbols) != 'CO':
            ads.set_chemical_symbols(ads.get_chemical_symbols()[::-1])
                                                  
        if surface == 'fcc100':
            if site == 'ontop':
                site_positions = [np.array([ary[0],ary[1],ary[2]+hdiff]) \
                                  for ary in ads_sites[site]]
                if len(top_indices) < 8:
                    atoms = dummy.copy()
                    site_positions = [p for p in site_positions if \
                                     (0 <= p[0] <= xcell) and (0 <= p[1] <= ycell)]
                if nsite == 'all':
                    for pos in site_positions:
                        ads.translate(pos - ads[0].position)
                        atoms.extend(ads)
                    if ads_indices:
                        atoms.extend(ads_atoms)
                else:
                    final_sites = random.sample(site_positions, nsite)
                    for pos in final_sites:
                        ads.translate(pos - ads[0].position)
                        atoms.extend(ads)
                    if ads_indices:
                        atoms.extend(ads_atoms)
            elif site in ['bridge', 'hollow']:
                site_positions = ads_sites['bridge']
                tup_lst = [(pos, get_neighbors_from_position(atoms, pos, 
                                                             cutoff=1.2*d100)) 
                           for pos in site_positions]
                bridge_positions = []
                hollow_positions = []
                nbr_lst = []
                for tup1 in tup_lst:
                    pos = tup1[0]
                    nbr_tup_lst = tup1[1]
                    idx_lst = [tup2[0] for tup2 in nbr_tup_lst]
                    nbr_lst.append((pos, idx_lst))
                for (pos, nbrs) in nbr_lst:                    
                    if len(nbrs) == 2:
                        bridge_positions.append(np.array([pos[0],pos[1],
                                                          pos[2]+hdiff]))
                    else:                    
                        hollow_positions.append(np.array([pos[0],pos[1],
                                                          pos[2]+hdiff]))
                if len(top_indices) < 8:
                    atoms = dummy.copy()
                    bridge_positions = [p for p in bridge_positions if \
                                       (0 <= p[0] <= xcell) and (0 <= p[1] <= ycell)]
                    hollow_positions = [p for p in hollow_positions if \
                                       (0 <= p[0] <= xcell) and (0 <= p[1] <= ycell)]
                if nsite == 'all':
                    for pos in locals()['{}_positions'.format(site)]:
                        ads.translate(pos - ads[0].position)
                        atoms.extend(ads)
                    if ads_indices:
                        atoms.extend(ads_atoms)
                else:
                    final_sites = random.sample(locals()['{}_positions'.format(site)], 
                                                nsite)
                    for pos in final_sites:
                        ads.translate(pos - ads[0].position)
                        atoms.extend(ads)
                    if ads_indices:
                        atoms.extend(ads_atoms)
                                                                                                                             
        elif surface == 'fcc111':
            if site in ['ontop', 'bridge']:
                site_positions = [np.array([ary[0],ary[1],ary[2]+hdiff]) \
                                  for ary in ads_sites[site]] 
                if len(top_indices) < 4:
                    atoms = dummy.copy()
                    site_positions = [p for p in site_positions \
                    if (0 <= np.dot(p, np.linalg.inv(atoms.cell))[0] <= 1)\
                    and (0 <= np.dot(p, np.linalg.inv(atoms.cell))[1] <= 1)]

                if nsite == 'all':
                    for pos in site_positions:
                        ads.translate(pos - ads[0].position)
                        atoms.extend(ads)
                    if ads_indices:
                        atoms.extend(ads_atoms)
                else:
                    final_sites = random.sample(site_positions, nsite)
                    for pos in final_sites: 
                        ads.translate(pos - ads[0].position)
                        atoms.extend(ads)
                    if ads_indices:
                        atoms.extend(ads_atoms)
            elif site in ['fcc', 'hcp']:
                site_positions = ads_sites['hollow']
                nbr_lst = [(pos, get_neighbors_from_position(atoms, pos, 
                            cutoff=1.9*d111)) for pos in site_positions]
                fcc_positions = []
                hcp_positions = []
                for (pos, nbrs) in nbr_lst: 
                    test_pos = pos - np.array([0,0,layer_thickness+2])
                    new_nbr_lst = get_neighbors_from_position(atoms, test_pos, 
                                                              cutoff=0.5)
                    if not new_nbr_lst:                    
                        fcc_positions.append(np.array([pos[0],pos[1],pos[2]+hdiff]))
                    else:                    
                        hcp_positions.append(np.array([pos[0],pos[1],pos[2]+hdiff]))

                if len(top_indices) < 4:
                    atoms = dummy.copy()
                    fcc_positions = [p for p in fcc_positions \
                    if (0 <= np.dot(p, np.linalg.inv(atoms.cell))[0] <= 1)\
                    and (0 <= np.dot(p, np.linalg.inv(atoms.cell))[1] <= 1)]
                    hcp_positions = [p for p in hcp_positions \
                    if (0 <= np.dot(p, np.linalg.inv(atoms.cell))[0] <= 1)\
                    and (0 <= np.dot(p, np.linalg.inv(atoms.cell))[1] <= 1)]

                if nsite == 'all':
                    for pos in locals()['{}_positions'.format(site)]:
                        ads.translate(pos - ads[0].position)
                        atoms.extend(ads)
                    if ads_indices:
                        atoms.extend(ads_atoms)
                else:
                    final_sites = random.sample(locals()['{}_positions'.format(site)], 
                                                nsite)
                    for pos in final_sites:
                        ads.translate(pos - ads[0].position)
                        atoms.extend(ads)
                    if ads_indices:
                        atoms.extend(ads_atoms)

    return atoms        


def get_monometallic_sites(atoms, site, surface=None, height=None, include_subsurface=False): 
    """Get all sites of a specific type from a nanoparticle or surface slab and assign 
    a label to it. Elemental composition is ignored.""" 

    label_dict = {'ontop' : '1', 
                 'bridge' : '2', 
                 'fcc' : '3', 
                 'hcp' : '4',
                 'hollow' : '5'}
    atoms.info['data'] = {}                      
    atoms = atoms[[a.index for a in atoms if a.symbol not in adsorbates]]         
    sites = []
    if True not in atoms.pbc:
        if surface is None:
            raise ValueError('Surface must be specified for a nanoparticle')
        if height is None:
            ads = NanoparticleAdsorptionSites(atoms)
        else:
            heights_dict[site] = height
            ads = NanoparticleAdsorptionSites(atoms, heights=heights_dict)
        nl = ads.nlist           
 
        special_sites = ads.get_sites_from_surface(site, surface)
        if special_sites:
            for site in special_sites:                 
                site_name = site['site']
                if include_subsurface:
                    hcp_neighbor_indices = []
                    hollow_neighbor_indices = []
                    if site_name == 'hcp':
                        for i in site['indices']:
                            indices = nl.get_neighbors(i, 3.5)
                            for idx in indices:
                                hcp_neighbor_indices.append(idx)
                        subsurf_index = [key for key, count in \
                        Counter(hcp_neighbor_indices).items() if count == 3][0]
                        site['indices'] += (subsurf_index,)                    
                    elif site_name == 'hollow':
                        for i in site['indices']:
                            indices = nl.get_neighbors(i, 3.5)
                            for idx in indices:
                                hollow_neighbor_indices.append(idx)
                        subsurf_index = [key for key, count in \
                        Counter(hollow_neighbor_indices).items() if count == 4][0]
                        site['indices'] += (subsurf_index,)
                    else:
                        raise ValueError('{0} sites do not have second shell'.format(
                                         site_name))
                site['label'] = label_dict[site_name]
        sites += special_sites

    else:
        if height is None:
            hdiff = 0
        else:
            hdiff = height-2.
        top_indices = []        
        second_indices = []
        layerlist = get_layers(atoms, (0,0,1), tolerance=0.5)[0].tolist()
        nlayers = max(layerlist)
        for atom in atoms:
            i = atom.index
            if layerlist[i] == nlayers:
                top_indices.append(i)
            elif layerlist[i] == nlayers-1:
                second_indices.append(i)
        layer_thickness = abs(np.mean(atoms.positions[top_indices,2]) \
                              - np.mean(atoms.positions[second_indices,2]))
        d100 = np.sqrt(2)*layer_thickness 
        d111 = np.sqrt(6)*layer_thickness/2
        if surface is None: 
            surface = identify_surface(atoms[top_indices+second_indices]) 
        struct = AseAtomsAdaptor.get_structure(atoms)
        asf = AdsorbateSiteFinder(struct)
        ads_sites = asf.find_adsorption_sites(symm_reduce=0) 
                                                                                                                                                 
        if surface == 'fcc100':
            if len(top_indices) < 4:
                raise ValueError('Cell is too small to get complete site information')
            if site == 'ontop':
                for i in top_indices:
                    special_site = {}
                    special_site['indices'] = (i,)
                    special_site['surface'] = 'fcc100'
                    special_site['site'] = 'ontop'
                    special_site['position'] = atoms[i].position \
                                                         + np.array([0,0,2+hdiff])
                    special_site['label'] = label_dict['ontop'] 
                    sites.append(special_site)            
            elif site in ['bridge', 'hollow']:
                site_positions = ads_sites['bridge']
                tup_lst = [(pos, get_neighbors_from_position(atoms, pos, 
                            cutoff=1.2*d100)) for pos in site_positions]
                nbr_lst = []
                for tup1 in tup_lst:
                    pos = tup1[0]
                    nbr_tup_lst = tup1[1]
                    idx_lst = [tup2[0] for tup2 in nbr_tup_lst]
                    nbr_lst.append((pos, idx_lst))
                for (pos, nbrs) in nbr_lst:
                    special_site = {}            
                    if len(nbrs) == 2:                    
                        special_site['indices'] = tuple(nbrs)
                        special_site['surface'] = 'fcc100'
                        special_site['site'] = 'bridge'
                        special_site['position'] = np.array([pos[0],pos[1],
                                                                      pos[2]+hdiff])
                        special_site['label'] = label_dict['bridge'] 
                        if site == 'bridge':
                            sites.append(special_site)
                    else:
                        test_pos = pos - np.array([0,0,2])
                        new_tup_lst = get_neighbors_from_position(atoms, test_pos, 
                                                                  cutoff=2.2*d100)
                        new_tup_lst.sort(key=itemgetter(1))
                        if include_subsurface:
                            new_nbrs = [x[0] for x in new_tup_lst[:5]]
                            for n, nbridx in enumerate(new_nbrs):
                                if nbridx not in top_indices:
                                    new_nbrs.append(new_nbrs.pop(n))
                        else:
                            new_nbrs = [x[0] for x in new_tup_lst[:5] \
                                        if x[0] in top_indices] 
                        special_site['indices'] = tuple(new_nbrs)
                        special_site['surface'] = 'fcc100'
                        special_site['site'] = 'hollow'
                        special_site['position'] = np.array([pos[0],pos[1],
                                                                      pos[2]+hdiff])
                        special_site['label'] = label_dict['hollow'] 
                        if site == 'hollow':
                            sites.append(special_site)
                
        elif surface == 'fcc111':
            if len(top_indices) < 4:
                raise ValueError('Cell is too small to get complete site information')
            if site in ['ontop', 'bridge']:
                site_positions = ads_sites[site]
                if site == 'ontop':
                    for i in top_indices:
                        special_site = {}                                                                                         
                        special_site['indices'] = (i,)
                        special_site['surface'] = 'fcc100'
                        special_site['site'] = 'ontop'
                        special_site['position'] = atoms[i].position \
                                                             + np.array([0,0,2+hdiff])
                        special_site['label'] = label_dict['ontop'] 
                        sites.append(special_site)            
                else:
                    for pos in site_positions:
                        special_site = {}                                    
                        new_tup_lst = get_neighbors_from_position(atoms, pos, 
                                                                  cutoff=1.3*d111)
                        new_tup_lst.sort(key=itemgetter(1))
                        new_nbrs = [x[0] for x in new_tup_lst[:2]]                    
                        special_site['indices'] = tuple(new_nbrs)
                        special_site['surface'] = 'fcc111'
                        special_site['site'] = 'bridge'
                        special_site['position'] = np.array([pos[0],pos[1],
                                                                      pos[2]+hdiff])
                        special_site['label'] = label_dict['bridge'] 
                        sites.append(special_site)            
            elif site in ['fcc', 'hcp']:
                site_positions = ads_sites['hollow']                
                nbr_lst = [(pos, get_neighbors_from_position(atoms, pos, 
                           cutoff=1.9*d111)) for pos in site_positions]            
                for (pos, tup_lst) in nbr_lst:
                    test_pos = pos - np.array([0,0,layer_thickness+2])
                    new_nbr_lst = get_neighbors_from_position(atoms, test_pos, 
                                                              cutoff=0.5)
                    if not new_nbr_lst:                    
                        special_site = {}
                        tup_lst.sort(key=itemgetter(1))                    
                        nbrs = [x[0] for x in tup_lst[:3]]
                        special_site['indices'] = tuple(nbrs)
                        special_site['surface'] = 'fcc111'
                        special_site['site'] = 'fcc'
                        special_site['position'] = np.array([pos[0],pos[1],
                                                                      pos[2]+hdiff])
                        special_site['label'] = label_dict['fcc'] 
                        if site == 'fcc':
                            sites.append(special_site)
                    else:
                        special_site = {}
                        tup_lst.sort(key=itemgetter(1))
                        if include_subsurface:
                            nbrs = [x[0] for x in tup_lst if x[0] in top_indices][:3] \
                                    + [new_nbr_lst[0][0]]
                        else:
                            nbrs = [x[0] for x in tup_lst if x[0] in top_indices][:3]
                        special_site['indices'] = tuple(nbrs)
                        special_site['surface'] = 'fcc111'
                        special_site['site'] = 'hcp'
                        special_site['position'] = np.array([pos[0],pos[1],
                                                                      pos[2]+hdiff])
                        special_site['label'] = label_dict['hcp']  
                        if site == 'hcp':
                            sites.append(special_site)

    return sites


def enumerate_monometallic_sites(atoms, show_occupation=False, surface=None,
                                 heights=heights_dict, include_subsurface=False):
    """Get all sites from a nanoparticle or a surface slab. 
       Elemental composition is ignored.

       Please only specify the surface type for a surface structure."""

    all_sites = []

    if True not in atoms.pbc:
        np_sites = NanoparticleAdsorptionSites(atoms).site_list
        all_sites += np_sites 

    else:
        ontop_sites = get_monometallic_sites(atoms, 'ontop', surface, 
                                             height=heights['ontop'], include_subsurface=False)
        if ontop_sites:
            all_sites += ontop_sites
        bridge_sites = get_monometallic_sites(atoms, 'bridge', surface, 
                                              height=heights['bridge'], include_subsurface=False)
        if bridge_sites:
            all_sites += bridge_sites
        fcc_sites = get_monometallic_sites(atoms, 'fcc', surface, 
                                           height=heights['fcc'], include_subsurface=False)
        if fcc_sites:
            all_sites += fcc_sites
        hcp_sites = get_monometallic_sites(atoms, 'hollow', surface, 
                                           height=heights['hcp'], include_subsurface=include_subsurface)
        if hcp_sites:
            all_sites += hcp_sites
        hollow_sites = get_monometallic_sites(atoms, 'hollow', surface, 
                                              height=heights['hollow'], include_subsurface=include_subsurface)
        if hollow_sites:
            all_sites += hollow_sites

    if show_occupation:
        for site in all_sites:
            if _is_site_occupied(atoms, site, min_adsorbate_distance=0.5):
                site['occupied'] = 1
                    
    return all_sites


def bimetallic_add_adsorbate(atoms, adsorbate, site, surface=None, composition=None, 
                             include_subsurface=False, height=None, nsite=1, 
                             min_adsorbate_distance=1.):
    """A function for adding adsorbate to a specific adsorption site on a bimetalic 
    nanoparticle in icosahedron / cuboctahedron / decahedron / truncated-octahedron 
    shapes or 100 / 111 surface slab.

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
        4-fold hollow site 'hollow'.

    surface: Support 4 typical surfaces (positions) for fcc crystal where the adsorbate 
    is attached: 
        'vertex', 
        'edge', 
        'fcc100', 
        'fcc111'.

    composition: All possible elemental composition of the adsorption site for bimetalics:
        'ontop' sites include 2 compositions: 'A' or 'B'.
        'bridge' sites include 3 compositions: 'AA' or 'AB' or 'BB'.
        'hcp' and 'fcc' sites include 4 compositions: 'AAA' or 'AAB' or 'ABB' or 'BBB'.
        'hollow' sites include 6 compositions: 'AAAA' or 'AAAB' or 'AABB' or 'ABAB' or 
         'ABBB' or 'BBBB'.

    include_subsurface: The second shell element beneath the adsorption site.
        Default is False. This keyword can only be set to 'A' or 'B' for 'hcp' and 'hollow'.

    height: The height from the adsorbate to the surface.
        Default is {'ontop': 2.0, 'bridge': 1.8, 'fcc': 1.8, 'hcp': 1.8, 
        'hollow': 1.7} for nanoparticles and 2.0 for all sites on surface slabs.

    nsite: The number of such adsorption site that is attached with the adsorbate. 
        Default is 1. Set nsite = 'all' to attach the adsorbate to all such sites.

    min_adsorbate_distance: The minimum distance between two adsorbate atoms.                             
        Default value 0.1 is good in most cases. Play around to find the best value.
    
    Example
    -------
    bimetallic_add_adsorbate(atoms, adsorbate='CO', site='hollow', surface='fcc100', 
    composition='NiPtNiPt', include_subsurface='Pt', nsite='all')"""

    #print('System: adsorbate {0}, site {1}, surface {2}, composition {3}, 
    #second shell {4}'.format(adsorbate, site, surface, composition, include_subsurface))

    atoms.info['data'] = {}
    if composition is None:
        raise ValueError('Composition must be specified. '+
                         'Otherwise use the monometallic function.')

    if True not in atoms.pbc:
        if surface is None:
            raise ValueError('Surface must be specified for a nanoparticle')
        if height is None:
            ads = NanoparticleAdsorptionSites(atoms)
        else:
            heights_dict[site] = height
            ads = NanoparticleAdsorptionSites(atoms, heights=heights_dict)
        nl = ads.nlist
        sites = ads.get_sites_from_surface(site, surface)
        if not sites:
            print('This site is not possible at all. Please check your input parameters.')
        else:
            final_sites = []
            if sites[0]['site'] == 'ontop' and not include_subsurface:
                final_sites += [site for site in sites if \
                                atoms[site['indices'][0]].symbol == composition]
            elif sites[0]['site'] == 'bridge' and not include_subsurface:
                for site in sites:
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    if composition in [a+b, b+a]:
                        final_sites.append(site)
            elif sites[0]['site'] == 'fcc' and not include_subsurface:
                for site in sites:
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    c = atoms[site['indices'][2]].symbol
                    if composition in [a+b+c, a+c+b, b+a+c, b+c+a, c+a+b, c+b+a]:
                        final_sites.append(site)
            elif sites[0]['site'] == 'hcp':
                for site in sites:
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    c = atoms[site['indices'][2]].symbol
                    if composition in [a+b+c, a+c+b, b+a+c, b+c+a, c+a+b, c+b+a]:
                        if not include_subsurface:
                            final_sites.append(site)
                        else:
                            neighbor_indices = []
                            for i in site['indices']:
                                indices = nl.get_neighbors(i, 3.5)
                                for idx in indices:
                                    if atoms[idx].symbol not in adsorbates:
                                        neighbor_indices.append(idx)                        
                            subsurf_index = [key for key, count in \
                            Counter(neighbor_indices).items() if count == 3][0]
                            subsurf_element = atoms[subsurf_index].symbol
                            if include_subsurface == subsurf_element:
                                final_sites.append(site)
            elif sites[0]['site'] == 'hollow':
                for site in sites:
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    c = atoms[site['indices'][2]].symbol
                    d = atoms[site['indices'][3]].symbol
                    comp = re.findall('[A-Z][^A-Z]*', composition)
                    if (comp[0] != comp[1]) and (comp[0]+comp[1] == comp[2]+comp[3]):
                        d0 = 0                                                                                                             
                        idmax = None
                        for i in range(1,4):
                            d = np.linalg.norm(atoms[site['indices'][0]].position - 
                                atoms[site['indices'][i]].position)          
                            if d > d0:
                                d0 = d
                                idmax = i
                        short = [x for x in range(1,4) if x != idmax]
                        b = atoms[site['indices'][short[0]]].symbol 
                        c = atoms[site['indices'][idmax]].symbol
                        d = atoms[site['indices'][short[1]]].symbol                                                
                        if (a != b) and (a+b == c+d):
                            if not include_subsurface:
                                final_sites.append(site)
                            else:
                                neighbor_indices = []
                                for i in site['indices']:
                                    indices = nl.get_neighbors(i, 3.5)
                                    for idx in indices:
                                        if atoms[idx].symbol not in adsorbates:
                                            neighbor_indices.append(idx)
                                subsurf_index = [key for key, count in \
                                Counter(neighbor_indices).items() if count == 4][0] 
                                subsurf_element = atoms[subsurf_index].symbol
                                if include_subsurface == subsurf_element:
                                    final_sites.append(site)
                    elif composition in [a+b+c+d, a+d+c+b, b+a+d+c, b+c+d+a, 
                                         c+b+a+d, c+d+a+b, d+a+b+c, d+c+b+a]:
                        if not include_subsurface:
                            final_sites.append(site)
                        else:
                            neighbor_indices = []
                            for i in site['indices']:
                                indices = nl.get_neighbors(i, 3.5)
                                for idx in indices:
                                    if atoms[idx].symbol not in adsorbates:
                                        neighbor_indices.append(idx)
                            subsurf_index = [key for key, count in \
                            Counter(neighbor_indices).items() if count == 4][0]
                            subsurf_element = atoms[subsurf_index].symbol
                            if include_subsurface == subsurf_element:
                                final_sites.append(site)
            else:
                raise ValueError('{0} sites do not have second shell'.format(site))
 
            if not final_sites:
                print('No such adsorption site found on this nanoparticle')
            elif adsorbate == 'CO':
                if (nsite == 'all') or (nsite > len(final_sites)):
                    for site in final_sites:                                            
                        add_adsorbate(atoms, molecule(adsorbate)[::-1], site)
                    atom_indices = [a.index for a in atoms if a.symbol == 'O']            
                    n_ads_atoms = 2
                    overlap_atoms_indices = []
                    for idx in atom_indices:   
                        neighbor_indices = nl.get_neighbors(idx, min_adsorbate_distance)
                        overlap = 0
                        for i in neighbor_indices:
                            if (atoms[i].symbol in adsorbates) and \
                               (i not in overlap_atoms_indices):
                                overlap += 1
                        if overlap > 0:
                            overlap_atoms_indices += list(set([idx-n_ads_atoms+1, idx]))
                    del atoms[overlap_atoms_indices]
                else:
                    final_sites = random.sample(final_sites, nsite)
                    for site in final_sites:
                        add_adsorbate(atoms, molecule(adsorbate)[::-1], site)
            else:
                if (nsite == 'all') or (nsite > len(final_sites)):
                    for site in final_sites:
                        add_adsorbate(atoms, molecule(adsorbate), site)
                    atom_indices = [a.index for a in atoms if a.symbol == adsorbate[-1]]
                    ads_symbols = molecule(adsorbate).get_chemical_symbols()
                    n_ads_atoms = len(ads_symbols)
                    overlap_atoms_indices = []
                    for idx in atom_indices:   
                        neighbor_indices = nl.get_neighbors(idx, min_adsorbate_distance)
                        overlap = 0
                        for i in neighbor_indices:                                                                
                            if (atoms[i].symbol in adsorbates) and \
                               (i not in overlap_atoms_indices):                       
                                overlap += 1                                                                      
                        if overlap > 0:                                                                           
                            overlap_atoms_indices += list(set([idx-n_ads_atoms+1, idx]))                                
                    del atoms[overlap_atoms_indices]    
                else:
                    final_sites = random.sample(final_sites, nsite)
                    for site in final_sites:
                        add_adsorbate(atoms, molecule(adsorbate), site)
    else:
        ads_indices = [a.index for a in atoms if a.symbol in adsorbates]
        ads_atoms = None
        if ads_indices:
            ads_atoms = atoms[ads_indices]
            atoms = atoms[[a.index for a in atoms if a.symbol not in adsorbates]]
        ads = molecule(adsorbate)[::-1]
        if str(ads.symbols) != 'CO':
            ads.set_chemical_symbols(ads.get_chemical_symbols()[::-1])
        final_sites = []
        if site in ['ontop','bridge','fcc']:
            sites = get_monometallic_sites(atoms, site, surface, height=height, 
                                           include_subsurface=False)
            if not sites:
                print('No such adsorption site found on this surface slab')
            elif site == 'ontop':
                final_sites += [site for site in sites if \
                                atoms[site['indices'][0]].symbol == composition]
            elif site == 'bridge':
                for site in sites:                          
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    if composition in [a+b, b+a]:
                        final_sites.append(site)
            elif site == 'fcc':
                for site in sites:                                                
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    c = atoms[site['indices'][2]].symbol
                    if composition in [a+b+c, a+c+b, b+a+c, b+c+a, c+a+b, c+b+a]:
                        final_sites.append(site)
        elif site in ['hcp','hollow']:
            sites = get_monometallic_sites(atoms, site, surface, height=height, 
                                           include_subsurface=True)
            if not sites:
                print('No such adsorption site found on this surface slab')
            elif site == 'hcp':
                for site in sites:
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    c = atoms[site['indices'][2]].symbol
                    if composition in [a+b+c, a+c+b, b+a+c, b+c+a, c+a+b, c+b+a]:
                        if not include_subsurface:
                            final_sites.append(site)
                        else:
                            subsurf_element = atoms[site['indices'][-1]].symbol
                            if include_subsurface == subsurf_element:
                                final_sites.append(site)
            elif site == 'hollow':
                for site in sites:
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    c = atoms[site['indices'][2]].symbol
                    d = atoms[site['indices'][3]].symbol
                    comp = re.findall('[A-Z][^A-Z]*', composition)
                    if (comp[0] != comp[1]) and (comp[0]+comp[1] == comp[2]+comp[3]):
                        d0 = 0
                        idmax = None
                        for i in range(1,4):
                            d = get_mic_distance(atoms[site['indices'][0]].position,
                                atoms[site['indices'][i]].position, atoms.cell, atoms.pbc)                        
                            if d > d0:
                                d0 = d
                                idmax = i
                        short = [x for x in range(1,4) if x != idmax]
                        b = atoms[site['indices'][short[0]]].symbol 
                        c = atoms[site['indices'][idmax]].symbol
                        d = atoms[site['indices'][short[1]]].symbol                                                
                        if (a != b) and (a+b == c+d):
                            if not include_subsurface:
                                final_sites.append(site)
                            else:
                                subsurf_element = atoms[site['indices'][-1]].symbol
                                if include_subsurface == subsurf_element:
                                    final_sites.append(site)
                    elif composition in [a+b+c+d, a+d+c+b, b+a+d+c, b+c+d+a, c+b+a+d, 
                                         c+d+a+b, d+a+b+c, d+c+b+a]:
                        if not include_subsurface:
                            final_sites.append(site)
                        else:
                            subsurf_element = atoms[site['indices'][-1]].symbol
                            if include_subsurface == subsurf_element:
                                final_sites.append(site)
        if nsite == 'all':
            for pos in [s['position'] for s in final_sites]:
                ads.translate(pos - ads[0].position)
                atoms.extend(ads)
            if ads_indices:
                atoms.extend(ads_atoms)
        else:
            final_sites = random.sample([s['position'] for s in final_sites], 
                                        nsite)
            for pos in final_sites:
                ads.translate(pos - ads[0].position)
                atoms.extend(ads)
            if ads_indices:
                atoms.extend(ads_atoms)

    return atoms


def get_bimetallic_sites(atoms, site, surface=None, composition=None, height=None, 
                         include_subsurface=False):
    """Get all sites of a specific type from a bimetallic nanoparticle or slab.  
    Elemental composition is included."""  

    atoms.info['data'] = {}
    atoms = atoms[[a.index for a in atoms if a.symbol not in adsorbates]]        

    if composition is None:
        raise ValueError('Composition must be specified. ' +
                         'Otherwise use the monometallic function.')
    final_sites = []
    if True not in atoms.pbc:
        if surface is None:
            raise ValueError('Surface must be specified for a nanoparticle')
        system = 'site {0}, surface {1}, composition {2}, second shell {3}'.format(site, 
                  surface, composition, include_subsurface)
        if height is None:
            ads = NanoparticleAdsorptionSites(atoms)
        else:
            heights_dict[site] = height
            ads = NanoparticleAdsorptionSites(atoms, heights=heights_dict)
        nl = ads.nlist
        sites = ads.get_sites_from_surface(site, surface)
        if sites:            
            if sites[0]['site'] == 'ontop' and not include_subsurface:
                final_sites += [site for site in sites if \
                                atoms[site['indices'][0]].symbol == composition]
            elif sites[0]['site'] == 'bridge' and not include_subsurface:
                for site in sites:
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    if composition in [a+b, b+a]:
                        final_sites.append(site)
            elif sites[0]['site'] == 'fcc' and not include_subsurface:
                for site in sites:
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    c = atoms[site['indices'][2]].symbol
                    if composition in [a+b+c, a+c+b, b+a+c, b+c+a, c+a+b, c+b+a]:
                        final_sites.append(site)
            elif sites[0]['site'] == 'hcp':
                for site in sites:
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    c = atoms[site['indices'][2]].symbol
                    if composition in [a+b+c, a+c+b, b+a+c, b+c+a, c+a+b, c+b+a]:
                        if not include_subsurface:
                            final_sites.append(site)
                        else:
                            neighbor_indices = []
                            for i in site['indices']: 
                                indices = nl.get_neighbors(i, 3.5)
                                for idx in indices:
                                    neighbor_indices.append(idx)                        
                            subsurf_index = [key for key, count in \
                            Counter(neighbor_indices).items() if count == 3][0]
                            subsurf_element = atoms[subsurf_index].symbol
                            if include_subsurface == subsurf_element:
                                site['indices'] += (subsurf_index,)
                                final_sites.append(site)
            elif sites[0]['site'] == 'hollow':
                for site in sites:
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    c = atoms[site['indices'][2]].symbol
                    d = atoms[site['indices'][3]].symbol
                    comp = re.findall('[A-Z][^A-Z]*', composition)
                    if (comp[0] != comp[1]) and (comp[0]+comp[1] == comp[2]+comp[3]):
                        d0 = 0                                                                                                             
                        idmax = None
                        for i in range(1,4):
                            d = np.linalg.norm(atoms[site['indices'][0]].position - 
                                atoms[site['indices'][i]].position)          
                            if d > d0:
                                d0 = d
                                idmax = i
                        short = [x for x in range(1,4) if x != idmax]
                        b = atoms[site['indices'][short[0]]].symbol 
                        c = atoms[site['indices'][idmax]].symbol
                        d = atoms[site['indices'][short[1]]].symbol                                                
                        if (a != b) and (a+b == c+d):
                            if not include_subsurface:
                                final_sites.append(site)
                            else:
                                neighbor_indices = []
                                for i in site['indices']:
                                    indices = nl.get_neighbors(i, 3.5)
                                    for idx in indices:
                                        neighbor_indices.append(idx)
                                subsurf_index = [key for key, count in \
                                Counter(neighbor_indices).items() if count == 4][0] 
                                subsurf_element = atoms[subsurf_index].symbol
                                if include_subsurface == subsurf_element:
                                    site['indices'] += (subsurf_index,)
                                    final_sites.append(site)
                    elif composition in [a+b+c+d, a+d+c+b, b+a+d+c, b+c+d+a, 
                                         c+b+a+d, c+d+a+b, d+a+b+c, d+c+b+a]:
                        if not include_subsurface:
                            final_sites.append(site)
                        else:
                            neighbor_indices = []
                            for i in site['indices']:
                                indices = nl.get_neighbors(i, 3.5)
                                for idx in indices:
                                    neighbor_indices.append(idx)
                            subsurf_index = [key for key, count in \
                            Counter(neighbor_indices).items() if count == 4][0]
                            subsurf_element = atoms[subsurf_index].symbol
                            if include_subsurface == subsurf_element:
                                site['indices'] += (subsurf_index,)
                                final_sites.append(site)
            else:
                raise ValueError('{0} sites do not have second shell'.format(site))

    else:
        if surface is None:            
            top_indices = []        
            second_indices = []
            layerlist = get_layers(atoms, (0,0,1), tolerance=0.5)[0].tolist()
            nlayers = max(layerlist)
            for atom in atoms:
                i = atom.index
                if layerlist[i] == nlayers:
                    top_indices.append(i)
                elif layerlist[i] == nlayers-1:
                    second_indices.append(i)
            surface = identify_surface(atoms[top_indices+second_indices])             

        system = 'site {0}, surface {1}, composition {2}, second shell {3}'.format(site, 
                  surface, composition, include_subsurface)
        if site in ['ontop','bridge','fcc']:
            sites = get_monometallic_sites(atoms, site, surface, height, include_subsurface=False)
            if not sites:
                print('No such adsorption site found on this surface slab')
            elif site == 'ontop':
                final_sites += [site for site in sites if \
                                atoms[site['indices'][0]].symbol == composition]
            elif site == 'bridge':
                for site in sites:                          
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    if composition in [a+b, b+a]:
                        final_sites.append(site)
            elif site == 'fcc':
                for site in sites:                                                
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    c = atoms[site['indices'][2]].symbol
                    if composition in [a+b+c, a+c+b, b+a+c, b+c+a, c+a+b, c+b+a]:
                        final_sites.append(site)
        elif site in ['hcp','hollow']:
            sites = get_monometallic_sites(atoms, site, surface, height, include_subsurface=True)
            if not sites:
                print('No such adsorption site found on this surface slab')
            elif site == 'hcp':
                for site in sites:
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    c = atoms[site['indices'][2]].symbol
                    if composition in [a+b+c, a+c+b, b+a+c, b+c+a, c+a+b, c+b+a]:
                        if not include_subsurface:
                            final_sites.append(site)
                        else:
                            subsurf_element = atoms[site['indices'][-1]].symbol
                            if include_subsurface == subsurf_element:
                                final_sites.append(site)
            elif site == 'hollow':
                for site in sites:
                    a = atoms[site['indices'][0]].symbol
                    b = atoms[site['indices'][1]].symbol
                    c = atoms[site['indices'][2]].symbol
                    d = atoms[site['indices'][3]].symbol
                    comp = re.findall('[A-Z][^A-Z]*', composition)
                    if (comp[0] != comp[1]) and (comp[0]+comp[1] == comp[2]+comp[3]):
                        d0 = 0
                        idmax = None
                        for i in range(1,4):
                            d = get_mic_distance(atoms[site['indices'][0]].position,
                                atoms[site['indices'][i]].position, atoms.cell, atoms.pbc) 
                            if d > d0:
                                d0 = d
                                idmax = i
                        short = [x for x in range(1,4) if x != idmax]
                        b = atoms[site['indices'][short[0]]].symbol 
                        c = atoms[site['indices'][idmax]].symbol
                        d = atoms[site['indices'][short[1]]].symbol                                                
                        if (a != b) and (a+b == c+d):
                            if not include_subsurface:
                                final_sites.append(site)
                            else:
                                subsurf_element = atoms[site['indices'][-1]].symbol
                                if include_subsurface == subsurf_element:
                                    final_sites.append(site)
                    elif composition in [a+b+c+d, a+d+c+b, b+a+d+c, b+c+d+a, 
                                         c+b+a+d, c+d+a+b, d+a+b+c, d+c+b+a]:
                        if not include_subsurface:
                            final_sites.append(site)
                        else:
                            subsurf_element = atoms[site['indices'][-1]].symbol
                            if include_subsurface == subsurf_element:
                                final_sites.append(site)
    if final_sites:
        for site in final_sites:
            site['system'] = system  

    return final_sites


def enumerate_bimetallic_sites(atoms, show_occupation=False, surface=None, 
                               heights=heights_dict, include_subsurface=False):
    """Get all sites from a bimetallic nanoparticle or slab. 
    Elemental composition is included.

    Please only specify the surface type for a surface structure.""" 

    all_sites = []
    atoms = atoms[[a.index for a in atoms if a.symbol not in adsorbates]]        
    elements = list(set(atoms.symbols))
    metals = [element for element in elements if element not in adsorbates]

    if True not in atoms.pbc:
        for surface in ['vertex', 'edge', 'fcc100', 'fcc111']:
            for composition in metals:
                ontop_sites = get_bimetallic_sites(atoms, 'ontop', surface, composition, 
                                                   heights['ontop'], include_subsurface=False)
                if ontop_sites:
                    all_sites += ontop_sites
        for surface in ['edge', 'fcc100', 'fcc111']:
            for composition in [metals[0]+metals[0], metals[0]+metals[1], 
                                metals[1]+metals[1]]:
                bridge_sites = get_bimetallic_sites(atoms, 'bridge', surface, composition, 
                                                    heights['bridge'], include_subsurface=False)
                if bridge_sites:
                    all_sites += bridge_sites
        for composition in [metals[0]+metals[0]+metals[0], metals[0]+metals[0]+metals[1], 
                            metals[0]+metals[1]+metals[1], metals[1]+metals[1]+metals[1]]:
            fcc_sites = get_bimetallic_sites(atoms, 'fcc', 'fcc111', composition, 
                                             heights['fcc'], include_subsurface=False)
            if fcc_sites:
                all_sites += fcc_sites
        for composition in [metals[0]+metals[0]+metals[0], metals[0]+metals[0]+metals[1], 
                            metals[0]+metals[1]+metals[1], metals[1]+metals[1]+metals[1]]:
            if include_subsurface:
                for subsurf_element in metals:
                    hcp_sites = get_bimetallic_sites(atoms, 'hcp', 'fcc111', composition, 
                                                     heights['hcp'], subsurf_element)
                    if hcp_sites:
                        all_sites += hcp_sites
            else:
                hcp_sites = get_bimetallic_sites(atoms, 'hcp', 'fcc111', composition, 
                                                 heights['hcp'], include_subsurface=False)
                if hcp_sites:
                    all_sites += hcp_sites
        for composition in [metals[0]+metals[0]+metals[0]+metals[0], 
                            metals[0]+metals[0]+metals[0]+metals[1], 
                            metals[0]+metals[0]+metals[1]+metals[1], 
                            metals[0]+metals[1]+metals[0]+metals[1], 
                            metals[0]+metals[1]+metals[1]+metals[1], 
                            metals[1]+metals[1]+metals[1]+metals[1]]:
            if include_subsurface:
                for subsurf_element in metals:
                    hollow_sites = get_bimetallic_sites(atoms, 'hollow', 'fcc100', 
                                   composition, heights['hollow'], subsurf_element)
                    if hollow_sites:
                        all_sites += hollow_sites
            else:
                hollow_sites = get_bimetallic_sites(atoms, 'hollow', 'fcc100', composition,
                                                    heights['hollow'], include_subsurface=False)
                if hollow_sites:
                    all_sites += hollow_sites

    else:
        for composition in metals:
            ontop_sites = get_bimetallic_sites(atoms, 'ontop', surface, composition, 
                                               heights['ontop'], include_subsurface=False)
            if ontop_sites:
                all_sites += ontop_sites
        for composition in [metals[0]+metals[0], metals[0]+metals[1], metals[1]+metals[1]]:
            bridge_sites = get_bimetallic_sites(atoms, 'bridge', surface, composition, 
                                                heights['bridge'], include_subsurface=False)
            if bridge_sites:
                all_sites += bridge_sites
        if surface == 'fcc111':
            for composition in [metals[0]+metals[0]+metals[0], 
                                metals[0]+metals[0]+metals[1], 
                                metals[0]+metals[1]+metals[1], 
                                metals[1]+metals[1]+metals[1]]:
                fcc_sites = get_bimetallic_sites(atoms, 'fcc', surface, composition, 
                                                 heights['fcc'], include_subsurface=False)
                if fcc_sites:
                    all_sites += fcc_sites
            for composition in [metals[0]+metals[0]+metals[0], 
                                metals[0]+metals[0]+metals[1], 
                                metals[0]+metals[1]+metals[1], 
                                metals[1]+metals[1]+metals[1]]:
                if include_subsurface:
                    for subsurf_element in metals:
                        hcp_sites = get_bimetallic_sites(atoms, 'hcp', surface, composition,
                                    heights['hcp'], include_subsurface=subsurf_element)
                        if hcp_sites:
                            all_sites += hcp_sites
                else:
                    hcp_sites = get_bimetallic_sites(atoms, 'hcp', surface, composition, 
                                                     heights['hcp'], include_subsurface=False)
                    if hcp_sites:
                        all_sites += hcp_sites
        elif surface == 'fcc100':
            for composition in [metals[0]+metals[0]+metals[0]+metals[0], 
                                metals[0]+metals[0]+metals[0]+metals[1], 
                                metals[0]+metals[0]+metals[1]+metals[1], 
                                metals[0]+metals[1]+metals[0]+metals[1], 
                                metals[0]+metals[1]+metals[1]+metals[1], 
                                metals[1]+metals[1]+metals[1]+metals[1]]:
                if include_subsurface:
                    for subsurf_element in metals:
                        hollow_sites = get_bimetallic_sites(atoms, 'hollow', surface, 
                        composition, heights['hollow'], include_subsurface=subsurf_element)
                        if hollow_sites:
                            all_sites += hollow_sites
                else:
                    hollow_sites = get_bimetallic_sites(atoms, 'hollow', surface, 
                                   composition, heights['hollow'], include_subsurface=False)
                    if hollow_sites:
                        all_sites += hollow_sites

    if show_occupation:
        for site in all_sites:
            if _is_site_occupied(atoms, site, min_adsorbate_distance=0.5):
                site['occupied'] = 1

    return all_sites


def enumerate_sites_with_adsorbates(atoms, adsorbate, heights=heights_dict):
    '''Add adsorbates to all possible sites.
    Might be useful in some occasions (e.g. visualization).'''
 
    all_sites = enumerate_monometallic_sites(atoms, heights, include_subsurface=False)
                                                                                 
    if True not in atoms.pbc:
        for site in all_sites:
            add_adsorbate(atoms, molecule(adsorbate), site)
    else:
        ads = molecule(adsorbate)[::-1]
        if str(ads.symbols) != 'CO':
            ads.set_chemical_symbols(ads.get_chemical_symbols()[::-1])
        positions = [s['position'] for s in all_sites]
        for pos in positions:
            ads.translate(pos - ads[0].position)
            atoms.extend(ads)

    return atoms


def _is_site_occupied(atoms, site, min_adsorbate_distance=0.5):
    """Returns True if the site on the atoms object is occupied by
    creating a sphere of radius min_adsorbate_distance and checking
    that no other adsorbate is inside the sphere.
   
    Don't call this function directly."""

    # if site['occupied']:
    #     return True
    if True not in atoms.pbc:
        height = site['height']
        normal = np.array(site['normal'])
        pos = np.array(site['position']) + normal * height
        dists = [np.linalg.norm(pos - a.position)
                 for a in atoms if a.symbol in adsorbates]
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
                 for a in atoms if a.symbol in adsorbates]
        for d in dists:
            if d < min_adsorbate_distance:
                # print('under min d', d, pos)
                # site['occupied'] = 1
                return True
        return False
                                                                                   
                                                                                   
def _is_site_occupied_by(atoms, adsorbate, site, 
                         min_adsorbate_distance=0.5):
    """Returns True if the site on the atoms object is occupied 
    by a specific species.
    
    Don't call this function directly."""
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


def label_occupied_sites(atoms, adsorbate, include_subsurface=False):
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
    
    Set include_subsurface=True if you also want to label the second shell atoms.'''

    species_pseudo_mapping = [('As','Sb'),('Se','Te'),('Br','I')]  
    elements = list(set(atoms.symbols))
    metals = [element for element in elements if element not in adsorbates]
    mA = metals[0]
    mB = metals[1]
    if Atom(metals[0]).number > Atom(metals[1]).number:
        mA = metals[1]
        mB = metals[0]
    sites = enumerate_monometallic_sites(atoms, include_subsurface=include_subsurface)
    n_occupied_sites = 0
    atoms.set_tags(0)
    if isinstance(adsorbate, list):               
        if len(adsorbate) == 2:
            for site in sites:            
                for ads in adsorbate:
                    k = adsorbate.index(ads)
                    if _is_site_occupied_by(atoms, ads, site, 
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
            if _is_site_occupied(atoms, site, min_adsorbate_distance=0.5):
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
    print('{0} sites labeled with tags including {1}'.format(n_occupied_sites, 
                                                             tag_set))

    return atoms


def multi_label_counter(atoms, adsorbate, include_subsurface=False):
    '''Encoding the labels into 5d numpy arrays. 
    This can be further used as a fingerprint.

    Atoms that constitute an occupied adsorption site will be labeled as 1.
    If an atom contributes to multiple sites of same type, the number wil 
    increase. One atom can encompass multiple non-zero values if it 
    contributes to multiple types of sites.

    Note: Please provide atoms including adsorbate(s), with adsorbate being a 
    string or a list of strings.

    Set include_subsurface=True if you also want to label the second shell atoms.'''

    labeled_atoms = label_occupied_sites(atoms, adsorbate, include_subsurface)
    np_indices = [a.index for a in labeled_atoms if a.symbol not in adsorbates]
    np_atoms = labeled_atoms[np_indices]
    
    counter_lst = []
    for atom in np_atoms:
        if atom.symbol not in adsorbates:
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
