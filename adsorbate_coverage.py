from .adsorption_sites import AdsorptionSites, get_mic_distance, add_adsorbate, identify_surface
from .adsorption_sites import get_monometallic_sites, enumerate_monometallic_sites
from .adsorbate_operators import AdsorbateOperator 
from ase.io import read, write
from ase.build import molecule
from ase.neighborlist import NeighborList
import networkx as nx
from collections import defaultdict
import numpy as np
import random
import copy


adsorbates = 'SCHON'
heights_dict = {'ontop': 2.0, 'bridge': 1.8, 'fcc': 1.8, 'hcp': 1.8, 'hollow': 1.7}


def get_coverage(atoms, adsorbate, nfullsite=None):
    """Get the coverage of a nanoparticle / surface with adsorbates.
       Provide the number of adsorption sites under 1 ML coverage will
       significantly accelerate the calculation."""

    sites = enumerate_monometallic_sites(atoms, second_shell=False)
    ao = AdsorbateOperator(adsorbate, sites)
    if nfullsite is None:
        pattern = pattern_generator(atoms, 'O', ['fcc111', 'fcc100'], coverage=1)
        nfullsite = len([a for a in pattern if a.symbol == 'O'])
    symbol = list(adsorbate)[0]
    nadsorbate = 0
    for a in atoms:
        if a.symbol == symbol:
            nadsorbate += 1

    return nadsorbate / nfullsite


def group_sites(atoms, sites):
    """A function that uses networkx to group one type of sites 
       by geometrical facets of the nanoparticle"""

    # Find all indices of vertex and edge sites
    vertex_sites = get_monometallic_sites(atoms, 'ontop', surface='vertex', second_shell=False) 
    edge_sites = get_monometallic_sites(atoms, 'ontop', surface='edge', second_shell=False)
    special_indices_tuples = [s['indices'] for s in vertex_sites] + [s['indices'] for s in edge_sites]
    special_indices = set(list(sum(special_indices_tuples, ())))
    
    G=nx.Graph()
    for site in sites:
        indices = site['indices']
        reduced_indices = tuple(i for i in indices if i not in special_indices)
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


def symmetric_pattern_generator(atoms, adsorbate, surface=['fcc100','fcc111'], coverage=1., height=None, min_adsorbate_distance=0.6):
    """A function for generating certain well-defined symmetric adsorbate coverage patterns.
       Parameters
       ----------
       atoms: The nanoparticle or surface slab onto which the adsorbate should be added.
           
       adsorbate: The adsorbate. Must be one of the following three types:
           A string containing the chemical symbol for a single atom.
           An atom object.
           An atoms object (for a molecular adsorbate).                                                                                                         
       surface: Support 2 typical surfaces for fcc crystal where the adsorbate is attached:  
           'fcc100', 
           'fcc111'.
           Can either specify a string or a list of strings

       coverage: The coverage (ML) of the adsorbate.
           Note that for small nanoparticles, the function might give results that do not correspond to the coverage.
           This is normal since the surface area can be too small to encompass the coverage pattern properly.
           We expect this function to work especially well on large nanoparticles and extended surfaces.
                                                                                                         
       height: The height from the adsorbate to the surface.
           Default is 'ontop': 2.0, 'bridge': 1.8, 'fcc': 1.8, 'hcp': 1.8, 'hollow': 1.7 for nanoparticles 
           and 2.0 for all sites on surface slabs.

       min_adsorbate_distance: The minimum distance between two adsorbate atoms.
           Default value 0.2 is good for adsorbate coverage patterns. Play around to find the best value.
       
       Example
       ------- 
       pattern_generator(atoms, adsorbate='CO', surface=['fcc100','fcc111'], coverage=3/4)"""

    rmin = min_adsorbate_distance/2.9
    ads_indices = [a.index for a in atoms if a.symbol in adsorbates]
    ads_atoms = None
    if ads_indices:
        ads_atoms = atoms[ads_indices]
        atoms = atoms[[a.index for a in atoms if a.symbol not in adsorbates]]
    ads = molecule(adsorbate)[::-1]
    if str(ads.symbols) != 'CO':
        ads.set_chemical_symbols(ads.get_chemical_symbols()[::-1])
    if True in atoms.get_pbc():
        surface = identify_surface(atoms) 
    if not isinstance(surface, list):
        surface = [surface]     
    final_sites = []
    positions = []
    if 'fcc111' in surface: 
        if coverage == 1:
            sites = get_monometallic_sites(atoms, site='fcc', surface='fcc111', height=height, second_shell=False)
            if sites:
                final_sites += sites
                positions += [s['adsorbate_position'] for s in sites]

        elif coverage == 3/4:
            # Kagome pattern
            all_sites = get_monometallic_sites(atoms, site='fcc', surface='fcc111', height=height, second_shell=False)
            if True not in atoms.get_pbc():
                grouped_sites = group_sites(atoms, all_sites)
            else:
                grouped_sites = {'pbc_sites': all_sites}
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
                                for i in sitej['indices']:
                                    common_site_indices.append(i)
                            else:
                                non_common_sites.append(sitej)
                        for sitej in non_common_sites:
                            overlap = sum([common_site_indices.count(i) for i in sitej['indices']])
                            if overlap == 1 and sitej['indices'] not in [s['indices'] for s in sites_to_delete]:
                                sites_to_delete.append(sitej)                
                    for s in sites:
                        if s['indices'] not in [st['indices'] for st in sites_to_delete]:
                            final_sites.append(s)
                            positions.append(s['adsorbate_position'])

        elif coverage == 2/4:
            # Honeycomb pattern
            fcc_sites = get_monometallic_sites(atoms, site='fcc', surface='fcc111', height=height, second_shell=False) 
            hcp_sites = get_monometallic_sites(atoms, site='hcp', surface='fcc111', height=height, second_shell=False) 
            all_sites = fcc_sites + hcp_sites
            if True not in atoms.get_pbc():
                grouped_sites = group_sites(atoms, all_sites)
            else:                
                grouped_sites = {'pbc_sites': all_sites}
            for sites in grouped_sites.values():
                if sites:                    
                    sites_to_remain = [sites[0]]
                    for sitei in sites_to_remain:
                        for sitej in sites:
                            if sitej['indices'] == sitei['indices']:
                                pass
                            elif len(set(sitej['indices']) & set(sitei['indices'])) == 1 \
                                 and sitej['site'] != sitei['site'] \
                                 and sitej['indices'] not in [s['indices'] for s in sites_to_remain]:
                                sites_to_remain.append(sitej)
                    final_sites += sites_to_remain                                         
                    positions += [s['adsorbate_position'] for s in sites_to_remain]
            if True not in atoms.get_pbc():                                                                       
                bad_sites = []
                for sti in final_sites:
                    if sti['site'] == 'hcp':
                        count = 0
                        for stj in final_sites:
                            if stj['site'] == 'fcc':
                                if len(set(stj['indices']) & set(sti['indices'])) == 2:
                                    count += 1
                        if count != 0:
                            bad_sites.append(sti)
                final_sites = [s for s in final_sites if s['indices'] not in [st['indices'] for st in bad_sites]]

        elif coverage == 1/4:
            # Kagome pattern                                                                 
            all_sites = get_monometallic_sites(atoms, site='fcc', surface='fcc111', height=height, second_shell=False)
            if True not in atoms.get_pbc():
                grouped_sites = group_sites(atoms, all_sites)
            else:
                grouped_sites = {'pbc_sites': all_sites}
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
                                for i in sitej['indices']:
                                    common_site_indices.append(i)
                            else:
                                non_common_sites.append(sitej)
                        for sitej in non_common_sites:
                            overlap = sum([common_site_indices.count(i) for i in sitej['indices']])
                            if overlap == 1 and sitej['indices'] not in [s['indices'] for s in sites_to_remain]:
                                sites_to_remain.append(sitej)               
                    final_sites += sites_to_remain
                    positions += [s['adsorbate_position'] for s in sites if s['indices'] in [st['indices'] for st in sites_to_remain]]

    if 'fcc100' in surface:
        if coverage == 1:
            sites = get_monometallic_sites(atoms, site='hollow', surface='fcc100', height=height, second_shell=False)
            if sites:
                final_sites += sites
                positions += [s['adsorbate_position'] for s in sites]

        elif coverage == 3/4:
            all_sites = get_monometallic_sites(atoms, site='hollow', surface='fcc100', height=height, second_shell=False)            
            if True not in atoms.get_pbc():
                grouped_sites = group_sites(atoms, all_sites)
            else:
                grouped_sites = {'pbc_sites': all_sites}
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
                                for i in sitej['indices']:
                                    common_site_indices.append(i)
                            else:
                                non_common_sites.append(sitej)
                        for sitej in non_common_sites:                        
                            overlap = sum([common_site_indices.count(i) for i in sitej['indices']])                        
                            if overlap in [1, 4] and sitej['indices'] not in [s['indices'] for s in sites_to_delete]:  
                                sites_to_delete.append(sitej)
                    for s in sites:
                        if s['indices'] not in [st['indices'] for st in sites_to_delete]:
                            final_sites.append(s)
                            positions.append(s['adsorbate_position'])

        elif coverage == 2/4:
            #c(2x2) pattern
            all_sites = get_monometallic_sites(atoms, site='hollow', surface='fcc100', height=height, second_shell=False)
            original_sites = copy.deepcopy(all_sites)
            if True not in atoms.get_pbc():
                grouped_sites = group_sites(atoms, all_sites)
            else:
                grouped_sites = {'pbc_sites': all_sites}
            for sites in grouped_sites.values():
                if sites:
                    sites_to_remain = [sites[0]]
                    for sitei in sites_to_remain:
                        for sitej in sites:
                            if (len(set(sitej['indices']) & set(sitei['indices'])) == 1) and \
                            (sitej['indices'] not in [s['indices'] for s in sites_to_remain]):
                                sites_to_remain.append(sitej)
                    for s in original_sites:
                        if s['indices'] in [st['indices'] for st in sites_to_remain]:
                            final_sites.append(s)
                            positions.append(s['adsorbate_position'])

        elif coverage == 1/4:
            #p(2x2) pattern
            all_sites = get_monometallic_sites(atoms, site='hollow', surface='fcc100', height=height, second_shell=False)
            if True not in atoms.get_pbc():
                grouped_sites = group_sites(atoms, all_sites)
            else:
                grouped_sites = {'pbc_sites': all_sites}
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
                                for i in sitej['indices']:
                                    common_site_indices.append(i)
                            else:
                                non_common_sites.append(sitej)
                        for sitej in non_common_sites:              
                            overlap = sum([common_site_indices.count(i) for i in sitej['indices']])                        
                            if overlap in [1, 4] and sitej['indices'] not in [s['indices'] for s in sites_to_remain]:  
                                sites_to_remain.append(sitej)
                    final_sites += sites_to_remain
                    positions += [s['adsorbate_position'] for s in sites if s['indices'] in [st['indices'] for st in sites_to_remain]]

    if True not in atoms.get_pbc():
        if coverage == 1:
            edge_sites = get_monometallic_sites(atoms, site='bridge', surface='edge', height=height, second_shell=False)
            vertex_sites = get_monometallic_sites(atoms, site='ontop', surface='vertex', height=height, second_shell=False)
            vertex_indices = [s['indices'][0] for s in vertex_sites]
            ve_common_indices = set()
            for esite in edge_sites:
                if set(esite['indices']) & set(vertex_indices):
                    for i in esite['indices']:
                        if i not in vertex_indices:
                            ve_common_indices.add(i)
            for esite in edge_sites:
                if not set(esite['indices']).issubset(ve_common_indices):
                    final_sites.append(esite)
                    positions.append(esite['adsorbate_position'])

        if coverage == 3/4:
            occupied_sites = final_sites.copy()
            hcp_sites = get_monometallic_sites(atoms, site='hcp', surface='fcc111', height=height, second_shell=False)
            edge_sites = get_monometallic_sites(atoms, site='bridge', surface='edge', height=height, second_shell=False)
            vertex_sites = get_monometallic_sites(atoms, site='ontop', surface='vertex', height=height, second_shell=False)
            vertex_indices = [s['indices'][0] for s in vertex_sites]
            ve_common_indices = set()
            for esite in edge_sites:
                if set(esite['indices']) & set(vertex_indices):
                    for i in esite['indices']:
                        if i not in vertex_indices:
                            ve_common_indices.add(i)                
            for esite in edge_sites:
                if not set(esite['indices']).issubset(ve_common_indices):
                    intermediate_indices = []
                    for hsite in hcp_sites:
                        if len(set(esite['indices']) & set(hsite['indices'])) == 2:
                            intermediate_indices.append(min(set(esite['indices']) ^ set(hsite['indices'])))
                    too_close = 0
                    for s in occupied_sites:
                        if len(set(esite['indices']) & set(s['indices'])) == 2:
                            too_close += 1
                    share = [0]
                    for interi in intermediate_indices:
                        share.append(len([s for s in occupied_sites if interi in s['indices']]))
                    if max(share) <= 2 and too_close == 0:
                        final_sites.append(esite)
                        positions.append(esite['adsorbate_position'])

        if coverage == 2/4:            
            occupied_sites = final_sites.copy()
            edge_sites = get_monometallic_sites(atoms, site='bridge', surface='edge', height=height, second_shell=False)
            vertex_sites = get_monometallic_sites(atoms, site='ontop', surface='vertex', height=height, second_shell=False)
            vertex_indices = [s['indices'][0] for s in vertex_sites]
            ve_common_indices = set()
            for esite in edge_sites:
                if set(esite['indices']) & set(vertex_indices):
                    for i in esite['indices']:
                        if i not in vertex_indices:
                            ve_common_indices.add(i)                
            for esite in edge_sites:
                if not set(esite['indices']).issubset(ve_common_indices):
                    intermediate_indices = []
                    for hsite in hcp_sites:
                        if len(set(esite['indices']) & set(hsite['indices'])) == 2:
                            intermediate_indices.append(min(set(esite['indices']) ^ set(hsite['indices'])))
                    share = [0]
                    for interi in intermediate_indices:
                        share.append(len([s for s in occupied_sites if interi in s['indices']]))
                    too_close = 0
                    for s in occupied_sites:
                        if len(set(esite['indices']) & set(s['indices'])) == 2:
                            too_close += 1
                    if max(share) <= 1 and too_close == 0:
                        final_sites.append(esite)
                        positions.append(esite['adsorbate_position'])

        if coverage == 1/4:
            occupied_sites = final_sites.copy()
            hcp_sites = get_monometallic_sites(atoms, site='hcp', surface='fcc111', height=height, second_shell=False)
            edge_sites = get_monometallic_sites(atoms, site='bridge', surface='edge', height=height, second_shell=False)
            vertex_sites = get_monometallic_sites(atoms, site='ontop', surface='vertex', height=height, second_shell=False)
            vertex_indices = [s['indices'][0] for s in vertex_sites]
            ve_common_indices = set()
            for esite in edge_sites:
                if set(esite['indices']) & set(vertex_indices):
                    for i in esite['indices']:
                        if i not in vertex_indices:
                            ve_common_indices.add(i)                
            for esite in edge_sites:
                if not set(esite['indices']).issubset(ve_common_indices):
                    intermediate_indices = []
                    for hsite in hcp_sites:
                        if len(set(esite['indices']) & set(hsite['indices'])) == 2:
                            intermediate_indices.append(min(set(esite['indices']) ^ set(hsite['indices'])))
                    share = [0]
                    for interi in intermediate_indices:
                        share.append(len([s for s in occupied_sites if interi in s['indices']]))
                    too_close = 0
                    for s in occupied_sites:
                        if len(set(esite['indices']) & set(s['indices'])) > 0:
                            too_close += 1
                    if max(share) == 0 and too_close == 0:
                        final_sites.append(esite)
                        positions.append(esite['adsorbate_position'])

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
                    if (atoms[i].symbol in adsorbates) and (i not in overlap_atoms_indices):
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
                    if (atoms[i].symbol in adsorbates) and (i not in overlap_atoms_indices):                       
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


def full_coverage_pattern_generator(atoms, adsorbate, site, height=None, min_adsorbate_distance=0.6):
    '''A function to generate different 1ML coverage patterns'''

    rmin = min_adsorbate_distance/2.9
    ads_indices = [a.index for a in atoms if a.symbol in adsorbates]
    ads_atoms = None
    if ads_indices:
        ads_atoms = atoms[ads_indices]
        atoms = atoms[[a.index for a in atoms if a.symbol not in adsorbates]]
    ads = molecule(adsorbate)[::-1]
    if str(ads.symbols) != 'CO':
        ads.set_chemical_symbols(ads.get_chemical_symbols()[::-1])
    final_sites = []
    positions = []
    if site == 'fcc':
        return symmetric_pattern_generator(atoms, adsorbate, coverage=1, height=height, min_adsorbate_distance=min_adsorbate_distance)
    elif site == 'ontop':
        sites = get_monometallic_sites(atoms, site='ontop', surface='fcc100', second_shell=False) +\
                get_monometallic_sites(atoms, site='ontop', surface='fcc111', second_shell=False) +\
                get_monometallic_sites(atoms, site='ontop', surface='edge', second_shell=False) +\
                get_monometallic_sites(atoms, site='ontop', surface='vertex', second_shell=False)
        if sites:
            final_sites += sites
            positions += [s['adsorbate_position'] for s in sites]
    elif site in ['hcp', 'hollow']:
        sites = get_monometallic_sites(atoms, site='hcp', surface='fcc111', height=height, second_shell=False) +\
                get_monometallic_sites(atoms, site='hollow', surface='fcc100', height=height, second_shell=False)
        if sites:
            final_sites += sites
            positions += [s['adsorbate_position'] for s in sites]

    if True not in atoms.get_pbc():
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
                    if (atoms[i].symbol in adsorbates) and (i not in overlap_atoms_indices):
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
                    if (atoms[i].symbol in adsorbates) and (i not in overlap_atoms_indices):                       
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


def random_pattern_generator(atoms, adsorbate, min_adsorbate_distance=2., heights=heights_dict):
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
    ads_indices = [a.index for a in atoms if a.symbol in adsorbates]
    ads_atoms = None
    if ads_indices:
        ads_atoms = atoms[ads_indices]
        atoms = atoms[[a.index for a in atoms if a.symbol not in adsorbates]]
    all_sites = enumerate_monometallic_sites(atoms, heights, second_shell=False)
    random.shuffle(all_sites)    
 
    if True not in atoms.get_pbc():
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
            if (atoms[i].symbol in adsorbates) and (i not in overlap_atoms_indices):                     
                overlap += 1                                                                      
        if overlap > 0:                                                                           
            overlap_atoms_indices += list(set([idx-n_ads_atoms+1, idx]))                                
    del atoms[overlap_atoms_indices]

    return atoms
