from .adsorbate_operators import AdsorbateOperator
from ase.cluster import Icosahedron, Octahedron
from asap3.analysis.rdf import RadialDistributionFunction
from asap3 import FullNeighborList
from asap3.analysis import FullCNA
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.visualize import view
from ase.build import molecule
from ase import Atom
import numpy as np
from itertools import combinations
from ase.data import reference_states as refstate
from ase.io import read, write
from collections import Counter
import re
import random
import warnings


warnings.filterwarnings('ignore')

# TODO: more robust way of going from key to surf site description
# e.g. the dictionary could be arranged in a different way
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

cubocta_dict = {
    # Edge sites on outermost shell -- Cubocta, Tocta
    str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}): 'edge',
    'edge': str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}),
    # Square sites on outermost shell -- Cubocta, Deca, Tocta
    str({(2, 1, 1): 4, (4, 2, 1): 4}): 'fcc100',
    'fcc100': str({(2, 1, 1): 4, (4, 2, 1): 4}),
    # Vertice sites on outermost shell -- Cubocta
    str({(2, 1, 1): 4, (4, 2, 1): 1}): 'vertex',
    'vertex': str({(2, 1, 1): 4, (4, 2, 1): 1}),
    # Triangle sites on outermost shell -- Icosa, Cubocta, Deca, Tocta
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': str({(3, 1, 1): 6, (4, 2, 1): 3}),
}

deca_dict = {
    # Edge sites (111)-(111) on outermost shell -- Deca
    str({(3, 1, 1): 4, (3, 2, 2): 2, (4, 2, 2): 2}): 'edge',
    'edge': str({(3, 1, 1): 4, (3, 2, 2): 2, (4, 2, 2): 2}),
    # Edge sites (111)-(100) on outermost shell -- Deca
    str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}): 'edge',
    'edge': str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}),
    # Edge sites (111)-(111)notch on outermost shell -- Deca
    str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}): 'edge',
    'edge': str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}),
    # Square sites on outermost shell -- Cubocta, Deca, Tocta
    str({(2, 1, 1): 4, (4, 2, 1): 4}): 'fcc100',
    'fcc100': str({(2, 1, 1): 4, (4, 2, 1): 4}),
    # Vertice sites on outermost shell -- Icosa, Deca
    str({(3, 2, 2): 5, (5, 5, 5): 1}): 'vertex',
    'vertex': str({(3, 2, 2): 5, (5, 5, 5): 1}),
    # Vertice sites A on outermost shell -- Deca
    str({(2, 0, 0): 1, (2, 1, 1): 2, (3, 1, 1): 2, (4, 2, 1): 1}): 'vertex',
    'vertex': str({(2, 0, 0): 1, (2, 1, 1): 2, (3, 1, 1): 2, (4, 2, 1): 1}),
    # Vertice sites B on outermost shell -- Deca
    str({(2, 0, 0): 2, (3, 0, 0): 1, (3, 1, 1): 2, (3, 2, 2): 1, (4, 2, 2): 1}): 'vertex',
    'vertex': str({(2, 0, 0): 2, (3, 0, 0): 1, (3, 1, 1): 2, (3, 2, 2): 1, (4, 2, 2): 1}),
    # Triangle (pentagon) sites on outermost shell -- Icosa, Cubocta, Deca, Tocta
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': str({(3, 1, 1): 6, (4, 2, 1): 3}),
    # Triangle (pentagon) notch sites on outermost shell -- Deca
    str({(3, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 2, (4, 2, 2): 2}): 'fcc111',
    'fcc111': str({(3, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 2, (4, 2, 2): 2}),
}

tocta_dict = {
    # Edge sites on outermost shell -- Cubocta, Tocta
    str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}): 'edge',
    'edge': str({(2, 1, 1): 3, (3, 1, 1): 2, (4, 2, 1): 2}),
    # Edge sites (111)-(111) on outermost shell -- Tocta
    str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}): 'edge',
    'edge': str({(2, 0, 0): 2, (3, 1, 1): 4, (4, 2, 1): 1}),
    # Square sites on outermost shell -- Cubocta, Deca, Tocta
    str({(2, 1, 1): 4, (4, 2, 1): 4}): 'fcc100',
    'fcc100': str({(2, 1, 1): 4, (4, 2, 1): 4}),
    # Vertice sites on outermost shell -- Tocta
    str({(2, 0, 0): 1, (2, 1, 1): 2, (3, 1, 1): 2, (4, 2, 1): 1}): 'vertex',
    'vertex': str({(2, 0, 0): 1, (2, 1, 1): 2, (3, 1, 1): 2, (4, 2, 1): 1}),
    # Triangle (pentagon) sites on outermost shell -- Icosa, Cubocta, Deca, Tocta
    str({(3, 1, 1): 6, (4, 2, 1): 3}): 'fcc111',
    'fcc111': str({(3, 1, 1): 6, (4, 2, 1): 3}),
}


adsorbates = 'SCHON'


class AdsorptionSites(object):
    def __init__(self, atoms, heights=None):
        atoms = atoms.copy()
        del atoms[[a.index for a in atoms if 'a' not in refstate[a.number]]]
        del atoms[[a.index for a in atoms if a.symbol in adsorbates]]
        self.atoms = atoms

        self.fullCNA = {}
        self.make_fullCNA()
        self.set_first_neighbor_distance_from_rdf()
        self.site_dict = self.get_site_dict()
        self.make_neighbor_list()
        self.surf_sites = self.get_surface_sites()

        if heights is None:
            self.heights = {'ontop': 2.0, 'bridge': 1.8,
                            'fcc': 1.8, 'hcp': 1.8, 'hollow': 1.7}
        else:
            self.heights = heights

        self.site_list = []
        self.populate_site_list()

    def populate_site_list(self):
        ss = self.surf_sites
        ssall = set(ss['all'])
        fcna = self.get_fullCNA()
        sd = self.site_dict
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
                                     'adsorbate_position': pos,
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
                            # fcc or hcp site
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
                                         'adsorbate_position': pos,
                                         'normal': normal,
                                         'height': self.heights[this_site],
                                         'indices': si})
                            sl.append(site)
                            usi.add(si)
                        elif self.is_eq(angle, np.pi/2.):
                            # hollow site
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
                                            # hollow site found
                                            normal = self.get_surface_normal(
                                                [s, n, m])
                                            ps = self.atoms[[
                                                n, m, s, k]].positions
                                            pos = np.average(ps, 0)
                                            site = self.new_site()
                                            site.update({'site': 'hollow',
                                                         'surface': site_surf,
                                                         'adsorbate_position': pos,
                                                         'normal': normal,
                                                         'height': self.heights['hollow'],
                                                         'indices': si})
                                            sl.append(site)
                                            usi.add(si)

                # ontop sites
                site = self.new_site()
                site.update({'site': 'ontop', 'surface': surface,
                             'adsorbate_position': self.atoms[s].position,
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
                'adsorbate_position': None, 'normal': None,
                'occupied': 0}

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
        vec2 = p2 - self.atoms[sites[0]].position
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
        '''
        Returns a dictionary with all the surface designations
        '''
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
        icosa_weight = cubocta_weight = deca_weight = tocta_weight = 0
        for s in fcna:
            if str(s) in icosa_dict:
                icosa_weight += 1
            if str(s) in cubocta_dict:
                cubocta_weight += 1
            if str(s) in deca_dict:
                deca_weight += 1
            if str(s) in tocta_dict:
                tocta_weight += 1
        full_weights = [icosa_weight, cubocta_weight, deca_weight, tocta_weight]
        if icosa_weight == max(full_weights):
            return icosa_dict
        elif cubocta_weight == max(full_weights):
            return cubocta_dict
        elif deca_weight == max(full_weights):
            return deca_dict
        else:
            return tocta_dict

    def set_first_neighbor_distance_from_rdf(self, rMax=10, nBins=200):
        rdf = RadialDistributionFunction(self.atoms, rMax, nBins).get_rdf()
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


def add_adsorbate(atoms, adsorbate, site):
    # Make the correct position
    height = site['height']
    normal = np.array(site['normal'])
    pos = np.array(site['adsorbate_position']) + normal * height

    ads = adsorbate.copy()
    if len(ads) > 1:
        avg_pos = np.average(ads[1:].positions, 0)
        ads.rotate(avg_pos - ads[0].position, normal)
        #pvec = np.cross(np.random.rand(3) - ads[0].position, normal)
        #ads.rotate(-45, pvec, center=ads[0].position)
    ads.translate(pos - ads[0].position)

    atoms.extend(ads)


def monometallic_add_adsorbate(atoms, adsorbate, site, surface='all', nsite='all'):
    atoms.info['data'] = {}
    ads = AdsorptionSites(atoms)
    #print(ads.get_surface_sites())
    sites = []
    if surface == 'all':
        for surf in ['vertex', 'edge', 'fcc100', 'fcc111']:
            special_sites = ads.get_sites_from_surface(site, surf)
            sites += special_sites
    else:        
        special_sites = ads.get_sites_from_surface(site, surface)
        for site in special_sites:
            sites.append(site)
    if not sites:
        print('No such adsorption site found on this nanoparticle.')
    elif adsorbate == 'CO':
        if nsite == 'all':
            for site in sites:                                            
                add_adsorbate(atoms, molecule(adsorbate)[::-1], site)
        else:
            final_sites = random.sample(sites, nsite)
            for site in final_sites:
                add_adsorbate(atoms, molecule(adsorbate)[::-1], site)
    else:
        if nsite == 'all':
            for site in sites:
                add_adsorbate(atoms, molecule(adsorbate), site)
        else:
            final_sites = random.sample(sites, nsite)
            for site in final_sites:
                add_adsorbate(atoms, molecule(adsorbate), site)
    return atoms


def get_monometallic_sites(atoms, site, surface, second_shell=False): 
    label_dct = {'site ontop, surface vertex' : '90', 
                 'site ontop, surface edge' : '91', 
                 'site ontop, surface fcc100' : '92', 
                 'site ontop, surface fcc111' : '93',
                 'site bridge, surface edge' : '94',
                 'site bridge, surface fcc100' : '95',
                 'site bridge, surface fcc111' : '96',
                 'site fcc, surface fcc111' : '97',
                 'site hcp, surface fcc111' : '98',
                 'site hollow, surface fcc100' : '99'}
    atoms.info['data'] = {}                      
    cutoff = natural_cutoffs(atoms)
    nl = NeighborList(cutoff, self_interaction=False, bothways=True)
    nl.update(atoms)            
    ads = AdsorptionSites(atoms)
    #print(ads.get_surface_sites())
    sites = []    
    special_sites = ads.get_sites_from_surface(site, surface)
    if special_sites:
        for site in special_sites:                 
            site_name = site['site']
            surface_name = site['surface']
            if second_shell:
                hcp_neighbor_indices = []
                hollow_neighbor_indices = []
                if site_name == 'hcp':
                    for i in site['indices']:
                        indices, offsets = nl.get_neighbors(i)
                        for inb in indices:
                            if atoms[inb].symbol not in adsorbates:
                                hcp_neighbor_indices.append(inb)
                    second_shell_index = [key for key, count in Counter(hcp_neighbor_indices).items() if count == 3][0]
                    site['indices'] += (second_shell_index,)
                elif site_name == 'hollow':
                    for i in site['indices']:
                        indices, offsets = nl.get_neighbors(i)
                        for inb in indices:
                            if atoms[inb].symbol not in adsorbates:
                                hollow_neighbor_indices.append(inb)
                    second_shell_index = [key for key, count in Counter(hollow_neighbor_indices).items() if count == 4][0]
                    site['indices'] += (second_shell_index,)
                else:
                    raise ValueError('{0} sites do not have second shell'.format(site_name))
            system = 'site {0}, surface {1}'.format(site_name, surface_name)
            site['label'] = label_dct[system]
    sites += special_sites

    return sites


def enumerate_monometallic_sites(atoms):
    all_sites = []
    for surface in ['vertex', 'edge', 'fcc100', 'fcc111']:
        ontop_sites = get_monometallic_sites(atoms, 'ontop', surface, second_shell=False)
        if ontop_sites:
            all_sites += ontop_sites
    for surface in ['edge', 'fcc100', 'fcc111']:
        bridge_sites = get_monometallic_sites(atoms, 'bridge', surface, second_shell=False)
        if bridge_sites:
            all_sites += bridge_sites
    fcc_sites = get_monometallic_sites(atoms, 'fcc', 'fcc111', second_shell=False)
    if fcc_sites:
        all_sites += fcc_sites
    hcp_sites = get_monometallic_sites(atoms, 'hcp', 'fcc111', second_shell=True)
    if hcp_sites:
        all_sites += hcp_sites
    hollow_sites = get_monometallic_sites(atoms, 'hollow', 'fcc100', second_shell=True)
    if hollow_sites:
        all_sites += hollow_sites

    return all_sites


def bimetallic_add_adsorbate(atoms, adsorbate, site, surface, composition, second_shell=False, nsite=1):
    """A function for adding adsorbate to a specific adsorption site on a bimetalic nanoparticle in 
    icosahedron / cuboctahedron / decahedron / truncated-octahedron shapes.

    Parameters:

    atoms: The nanoparticle onto which the adsorbate should be added.
        
    adsorbate: The adsorbate. Must be one of the following three types:
        A string containing the chemical symbol for a single atom.
        An atom object.
        An atoms object (for a molecular adsorbate).

    site: Support 5 typical adsorption sites: 
        1-fold site 'ontop', 
        2-fold site 'bridge', 
        3-fold hollow sites 'fcc' and 'hcp', 
        4-fold hollow site 'hollow'.

    surface: Support 4 typical surfaces (positions) for fcc crystal where the adsorbate is attached: 
        'vertex', 
        'edge', 
        'fcc100', 
        'fcc111'.

    composition: All possible elemental composition of the adsorption site for bimetalic nanoparticles:
        'ontop' sites include 2 compositions: 'A' or 'B'.
        'bridge' sites include 3 compositions: 'AA' or 'AB' or 'BB'.
        'hcp' and 'fcc' sites include 4 compositions: 'AAA' or 'AAB' or 'ABB' or 'BBB'.
        'hollow' sites include 6 compositions: 'AAAA' or 'AAAB' or 'AABB' or 'ABAB' or 'ABBB' or 'BBBB'.

    second_shell: The second shell element beneath the adsorption site.
        Default is False. This keyword can only be set to 'A' or 'B' for 'hcp' and 'hollow'.

    nsite: The number of such adsorption site that is attached with the adsorbate. 
        Default is 1. Set nsite = 'all' to attach the adsorbate to all such sites.
    
    Example: bimetallic_add_adsorbate(atoms, adsorbate='CO', site='hollow', surface='fcc100', 
        composition='NiPtNiPt', second_shell='Pt', nsite='all')"""

#    print('System: adsorbate {0}, site {1}, surface {2}, composition {3}, second shell {4}'.format(
#           adsorbate, site, surface, composition, second_shell))
    atoms.info['data'] = {}
    ads = AdsorptionSites(atoms)
    #print(ads.get_surface_sites())
    sites = ads.get_sites_from_surface(site, surface)
    if not sites:
        print('This site is not possible at all. Please check your input parameters.')
    else:
        final_sites = []
        if sites[0]['site'] == 'ontop' and not second_shell:
            final_sites += [site for site in sites if atoms[site['indices'][0]].symbol == composition]
        elif sites[0]['site'] == 'bridge' and not second_shell:
            for site in sites:
                a = atoms[site['indices'][0]].symbol
                b = atoms[site['indices'][1]].symbol
                if composition in [a+b, b+a]:
                    final_sites.append(site)
        elif sites[0]['site'] == 'fcc' and not second_shell:
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
                    if not second_shell:
                        final_sites.append(site)
                    else:
                        neighbor_indices = []
                        for i in site['indices']:
                            cutoff = natural_cutoffs(atoms)
                            nl = NeighborList(cutoff, self_interaction=False, bothways=True)
                            nl.update(atoms)
                            indices, offsets = nl.get_neighbors(i)
                            for inb in indices:
                                if atoms[inb].symbol not in adsorbates:
                                    neighbor_indices.append(inb)                        
                        second_shell_index = [key for key, count in Counter(neighbor_indices).items() 
                                              if count == 3][0]
                        second_shell_element = atoms[second_shell_index].symbol
                        if second_shell == second_shell_element:
                            final_sites.append(site)
        elif sites[0]['site'] == 'hollow':
            for site in sites:
                a = atoms[site['indices'][0]].symbol
                b = atoms[site['indices'][1]].symbol
                c = atoms[site['indices'][2]].symbol
                d = atoms[site['indices'][3]].symbol
                comp = re.findall('[A-Z][^A-Z]*', composition)
                if (comp[0] != comp[1]) and (comp[0]+comp[1] == comp[2]+comp[3]):
                    if (a != b) and (a+b == c+d):
                        if not second_shell:
                            final_sites.append(site)
                        else:
                            neighbor_indices = []
                            for i in site['indices']:
                                cutoff = natural_cutoffs(atoms)
                                nl = NeighborList(cutoff, self_interaction=False, bothways=True)
                                nl.update(atoms)
                                indices, offsets = nl.get_neighbors(i)
                                for inb in indices:
                                    if atoms[inb].symbol not in adsorbates:
                                        neighbor_indices.append(inb)
                            second_shell_index = [key for key, count in Counter(neighbor_indices).items() 
                                                  if count == 4][0] 
                            second_shell_element = atoms[second_shell_index].symbol
                            if second_shell == second_shell_element:
                                final_sites.append(site)
                elif composition in [a+b+c+d, a+d+c+b, b+a+d+c, b+c+d+a, c+b+a+d, c+d+a+b, d+a+b+c, d+c+b+a]:
                    if not second_shell:
                        final_sites.append(site)
                    else:
                        neighbor_indices = []
                        for i in site['indices']:
                            cutoff = natural_cutoffs(atoms)
                            nl = NeighborList(cutoff, self_interaction=False, bothways=True)
                            nl.update(atoms)
                            indices, offsets = nl.get_neighbors(i)
                            for inb in indices:
                                if atoms[inb].symbol not in adsorbates:
                                    neighbor_indices.append(inb)
                        second_shell_index = [key for key, count in Counter(neighbor_indices).items() 
                                              if count == 4][0]
                        second_shell_element = atoms[second_shell_index].symbol
                        if second_shell == second_shell_element:
                            final_sites.append(site)
        else:
            raise ValueError('{0} sites do not have second shell'.format(site))
        #print(final_sites)
        if not final_sites:
            print('No such adsorption site found on this nanoparticle')
        elif adsorbate == 'CO':
            if (nsite == 'all') or (nsite > len(final_sites)):
                for site in final_sites:                                            
                    add_adsorbate(atoms, molecule(adsorbate)[::-1], site)
            else:
                final_sites = random.sample(final_sites, nsite)
                for site in final_sites:
                    add_adsorbate(atoms, molecule(adsorbate)[::-1], site)
        else:
            if (nsite == 'all') or (nsite > len(final_sites)):
                for site in final_sites:
                    add_adsorbate(atoms, molecule(adsorbate), site)
            else:
                final_sites = random.sample(final_sites, nsite)
                for site in final_sites:
                    add_adsorbate(atoms, molecule(adsorbate), site)
    return atoms


def get_bimetallic_sites(atoms, site, surface, composition, second_shell=False):
    system = 'site {0}, surface {1}, composition {2}, second shell {3}'.format(site, surface, composition, second_shell)
    atoms.info['data'] = {}
    cutoff = natural_cutoffs(atoms)
    nl = NeighborList(cutoff, self_interaction=False, bothways=True)
    nl.update(atoms)
    ads = AdsorptionSites(atoms)
    sites = ads.get_sites_from_surface(site, surface)
    if sites:
        final_sites = []
        if sites[0]['site'] == 'ontop' and not second_shell:
            final_sites += [site for site in sites if atoms[site['indices'][0]].symbol == composition]
        elif sites[0]['site'] == 'bridge' and not second_shell:
            for site in sites:
                a = atoms[site['indices'][0]].symbol
                b = atoms[site['indices'][1]].symbol
                if composition in [a+b, b+a]:
                    final_sites.append(site)
        elif sites[0]['site'] == 'fcc' and not second_shell:
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
                    if not second_shell:
                        final_sites.append(site)
                    else:
                        neighbor_indices = []
                        for i in site['indices']: 
                            indices, offsets = nl.get_neighbors(i)
                            for inb in indices:
                                if atoms[inb].symbol not in adsorbates:
                                    neighbor_indices.append(inb)                        
                        second_shell_index = [key for key, count in Counter(neighbor_indices).items() 
                                              if count == 3][0]
                        second_shell_element = atoms[second_shell_index].symbol
                        if second_shell == second_shell_element:
                            site['indices'] += (second_shell_index,)
                            final_sites.append(site)
        elif sites[0]['site'] == 'hollow':
            for site in sites:
                a = atoms[site['indices'][0]].symbol
                b = atoms[site['indices'][1]].symbol
                c = atoms[site['indices'][2]].symbol
                d = atoms[site['indices'][3]].symbol
                comp = re.findall('[A-Z][^A-Z]*', composition)
                if (comp[0] != comp[1]) and (comp[0]+comp[1] == comp[2]+comp[3]):
                    if (a != b) and (a+b == c+d):
                        if not second_shell:
                            final_sites.append(site)
                        else:
                            neighbor_indices = []
                            for i in site['indices']:
                                indices, offsets = nl.get_neighbors(i)
                                for inb in indices:
                                    if atoms[inb].symbol not in adsorbates:
                                        neighbor_indices.append(inb)
                            second_shell_index = [key for key, count in Counter(neighbor_indices).items() 
                                                  if count == 4][0] 
                            second_shell_element = atoms[second_shell_index].symbol
                            if second_shell == second_shell_element:
                                site['indices'] += (second_shell_index,)
                                final_sites.append(site)
                elif composition in [a+b+c+d, a+d+c+b, b+a+d+c, b+c+d+a, c+b+a+d, c+d+a+b, d+a+b+c, d+c+b+a]:
                    if not second_shell:
                        final_sites.append(site)
                    else:
                        neighbor_indices = []
                        for i in site['indices']:
                            indices, offsets = nl.get_neighbors(i)
                            for inb in indices:
                                if atoms[inb].symbol not in adsorbates:
                                    neighbor_indices.append(inb)
                        second_shell_index = [key for key, count in Counter(neighbor_indices).items() 
                                              if count == 4][0]
                        second_shell_element = atoms[second_shell_index].symbol
                        if second_shell == second_shell_element:
                            site['indices'] += (second_shell_index,)
                            final_sites.append(site)
        else:
            raise ValueError('{0} sites do not have second shell'.format(site))
        if final_sites:
            for site in final_sites:
                site['system'] = system

        return final_sites


def enumerate_bimetallic_sites(atoms):
    all_sites = []
    elements = list(set(atoms.symbols))
    metals = [element for element in elements if element not in adsorbates]
    for surface in ['vertex', 'edge', 'fcc100', 'fcc111']:
        for composition in metals:
            ontop_sites = get_bimetallic_sites(atoms, 'ontop', surface, composition, second_shell=False)
            if ontop_sites:
                all_sites += ontop_sites
    for surface in ['edge', 'fcc100', 'fcc111']:
        for composition in [metals[0]+metals[0], metals[0]+metals[1], metals[1]+metals[1]]:
            bridge_sites = get_bimetallic_sites(atoms, 'bridge', surface, composition, second_shell=False)
            if bridge_sites:
                all_sites += bridge_sites
    for composition in [metals[0]+metals[0]+metals[0], metals[0]+metals[0]+metals[1], 
                        metals[0]+metals[1]+metals[1], metals[1]+metals[1]+metals[1]]:
        fcc_sites = get_bimetallic_sites(atoms, 'fcc', 'fcc111', composition, second_shell=False)
        if fcc_sites:
            all_sites += fcc_sites
    for composition in [metals[0]+metals[0]+metals[0], metals[0]+metals[0]+metals[1], 
                        metals[0]+metals[1]+metals[1], metals[1]+metals[1]+metals[1]]:
        for second_shell in metals:
            hcp_sites = get_bimetallic_sites(atoms, 'hcp', 'fcc111', composition, second_shell)
            if hcp_sites:
                all_sites += hcp_sites
    for composition in [metals[0]+metals[0]+metals[0]+metals[0], metals[0]+metals[0]+metals[0]+metals[1], 
                        metals[0]+metals[0]+metals[1]+metals[1], metals[0]+metals[1]+metals[0]+metals[1], 
                        metals[0]+metals[1]+metals[1]+metals[1], metals[1]+metals[1]+metals[1]+metals[1]]:
        for second_shell in metals:
            hollow_sites = get_bimetallic_sites(atoms, 'hollow', 'fcc100', composition, second_shell)
            if hollow_sites:
                all_sites += hollow_sites

    return all_sites


def label_occupied_sites(atoms, adsorbate):
    elements = list(set(atoms.symbols))
    metals = [element for element in elements if element not in adsorbates]
    mA = metals[0]
    mB = metals[1]
    if Atom(metals[0]).number > Atom(metals[1]).number:
        mA = metals[1]
        mB = metals[0]
    sites = enumerate_monometallic_sites(atoms) 
    ao = AdsorbateOperator(adsorbate, sites)
    n_found_occupied_sites = 0
    for site in sites:
        if ao.is_site_occupied(atoms, site, min_adsorbate_distance=0.1):
            site['occupied'] = 1
            indices = site['indices']
            label = site['label']
            for index in indices:                
                if atoms[index].tag == 0:
                    atoms[index].tag = label
                elif label not in str(atoms[index].tag):
                    atoms[index].tag = str(atoms[index].tag) + label
                if atoms[index].symbol == mA:
                    atoms[index].symbol = 'Nb'
                elif atoms[index].symbol == mB:
                    atoms[index].symbol = 'Pb'
            n_found_occupied_sites += 1
    n_occupied_sites = ao.count_occupied_sites(atoms)
    if n_found_occupied_sites != n_occupied_sites: 
        raise ValueError('Could not find all {0} occupied adsorption sites, found {1} instead'.format(
                          n_occupied_sites, n_found_occupied_sites))

    return atoms


def label_encoder(atoms, adsorbate):
    labeled_atoms = label_occupied_sites(atoms, adsorbate)
    np_indices = [a.index for a in labeled_atoms if a.symbol not in adsorbates]  
    np_atoms = labeled_atoms[np_indices]
    output = []
    for i, atom in enumerate(np_atoms):
        if atom.tag == 0:
            output.append((i, atom.symbol, np.zeros(10)))
        else:
            line = str(atom.tag)
            strings = [line[k:k+2] for k in range(0, len(line), 2)]
            zeros = list(np.zeros(10))
            for idx in [int(s[1]) for s in strings]:
                zeros[idx] = 1
            output.append((i, atom.symbol, np.asarray(zeros)))

    return output
