from .adsorption_sites import * 
from .utilities import *
from .settings import *
from ase.data import covalent_radii, atomic_numbers
from ase.formula import Formula
from ase import Atom, Atoms
from operator import itemgetter
import numpy as np
import re


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
        height = site_heights[site]
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
        if not is_list_or_tuple(indices):
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


def add_adsorbate_to_site(atoms, adsorbate, site, height=None, 
                          orientation=None):            
    '''orientation: vector that the adsorbate is algined to'''
    
    if height is None:
        height = site_heights[site['site']]

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
        if not ads:
            print('Molecule {} does not exist'.format(adsorbate),
                  'in the database. Nothing is added.')
            return 
        if len(ads) == 2 or adsorbate == 'COH':
            ads.rotate(ads[1].position - ads[0].position, normal)
            #pvec = np.cross(np.random.rand(3) - ads[0].position, normal)
            #ads.rotate(-45, pvec, center=ads[0].position)

    if adsorbate not in adsorbate_list:
        # Always sort the indices the same order as the input symbol.
        # This is a naive sorting which might cause H in wrong order.
        # Please sort your own adsorbate atoms by reindexing as has
        # been done in the adsorbate_molecule function in settings.py
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


def remove_adsorbate_from_site(atoms, site, remove_fragment=False):

    if not remove_fragment:
        si = list(site['adsorbate_indices'])
    else:
        si = list(site['fragment_indices'])
    del atoms[si]


def remove_adsorbates_too_close(atoms, adsorbate_coverage,
                                min_adsorbate_distance=0.5):

    dups = get_close_atoms(atoms, cutoff=min_adsorbate_distance,
                           mic=(True in atoms.pbc))
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

