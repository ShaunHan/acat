Building things
===============

Operate adsorbate
-----------------

.. automodule:: acat.build.actions
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: add_adsorbate, add_adsorbate_to_site, add_adsorbate_to_label, remove_adsorbate_from_site, remove_adsorbates_from_sites, remove_adsorbates_too_close

    .. autofunction:: add_adsorbate
               
    **Example**

    To add a NO molecule to a bridge site consists of one Pt and 
    one Ni on the fcc111 surface of a truncated octahedron:

        >>> from acat.build.actions import add_adsorbate 
        >>> from ase.cluster import Octahedron
        >>> from ase.visualize import view
        >>> atoms = Octahedron('Ni', length=7, cutoff=2)
        >>> for atom in atoms:
        ...     if atom.index % 2 == 0:
        ...         atom.symbol = 'Pt' 
        >>> add_adsorbate(atoms, adsorbate='NO', site='bridge',
        ...               surface='fcc111', composition='NiPt')
        >>> view(atoms)

    Output: 
    
    .. image:: ../images/add_adsorbate.png 
       :scale: 70 %
       :align: center

    .. autofunction:: add_adsorbate_to_site

    **Example**

    To add CO to all fcc sites of an icosahedral nanoparticle:

        >>> from acat.adsorption_sites import ClusterAdsorptionSites
        >>> from acat.build.actions import add_adsorbate_to_site
        >>> from ase.cluster import Icosahedron
        >>> from ase.visualize import view
        >>> atoms = Icosahedron('Pt', noshells=5)
        >>> atoms.center(vacuum=5.)
        >>> cas = ClusterAdsorptionSites(atoms)
        >>> fcc_sites = cas.get_sites(site='fcc')
        >>> for site in fcc_sites:
        ...     add_adsorbate_to_site(atoms, adsorbate='CO', site=site)
        >>> view(atoms)

    Output:

    .. image:: ../images/add_adsorbate_to_site_1.png
       :align: center

    To add a bidentate CH3OH to the (54, 57, 58) site on a Pt fcc111 
    surface slab and rotate the orientation to a neighbor site:

        >>> from acat.adsorption_sites import SlabAdsorptionSites
        >>> from acat.adsorption_sites import get_adsorption_site
        >>> from acat.build.actions import add_adsorbate_to_site 
        >>> from acat.utilities import get_mic
        >>> from ase.build import fcc111
        >>> from ase.visualize import view
        >>> atoms = fcc111('Pt', (4, 4, 4), vacuum=5.)
        >>> i, site = get_adsorption_site(atoms, indices=(54, 57, 58),
        ...                               surface='fcc111',
        ...                               return_index=True)
        >>> sas = SlabAdsorptionSites(atoms, surface='fcc111')
        >>> sites = sas.get_sites()
        >>> nbsites = sas.get_neighbor_site_list(neighbor_number=1)
        >>> nbsite = sites[nbsites[i][0]] # Choose the first neighbor site
        >>> ori = get_mic(site['position'], nbsite['position'], atoms.cell)
        >>> add_adsorbate_to_site(atoms, adsorbate='CH3OH', site=site, 
        ...                       orientation=ori)
        >>> view(atoms)

    Output:

    .. image:: ../images/add_adsorbate_to_site_2.png
       :scale: 70 %
       :align: center

    .. autofunction:: add_adsorbate_to_label

    **Example**

    To add a NH molecule to a site with bimetallic label 14 (an hcp 
    CuCuAu site) on a fcc110 surface slab:

        >>> from acat.build.actions import add_adsorbate_to_label 
        >>> from ase.build import fcc110
        >>> from ase.visualize import view
        >>> atoms = fcc110('Cu', (3, 3, 8), vacuum=5.)
        >>> for atom in atoms:
        ...     if atom.index % 2 == 0:
        ...         atom.symbol = 'Au'
        ... atoms.center()
        >>> add_adsorbate_to_label(atoms, adsorbate='NH', label=14,
        ...                        surface='fcc110', composition_effect=True)
        >>> view(atoms)

    Output:

    .. image:: ../images/add_adsorbate_to_label.png
       :scale: 70 %
       :align: center

    .. autofunction:: remove_adsorbate_from_site

    **Example**

    To remove a CO molecule from a fcc111 surface slab with one 
    CO and one OH:

        >>> from acat.adsorption_sites import SlabAdsorptionSites
        >>> from acat.adsorbate_coverage import SlabAdsorbateCoverage
        >>> from acat.build.actions import add_adsorbate_to_site 
        >>> from acat.build.actions import remove_adsorbate_from_site
        >>> from ase.build import fcc111
        >>> from ase.visualize import view
        >>> atoms = fcc111('Pt', (6, 6, 4), 4, vacuum=5.)
        >>> atoms.center() 
        >>> sas = SlabAdsorptionSites(atoms, surface='fcc111')
        >>> sites = sas.get_sites()
        >>> add_adsorbate_to_site(atoms, adsorbate='OH', site=sites[0])
        >>> add_adsorbate_to_site(atoms, adsorbate='CO', site=sites[-1])
        >>> sac = SlabAdsorbateCoverage(atoms, sas)
        >>> occupied_sites = sac.get_sites(occupied_only=True)
        >>> CO_site = next((s for s in occupied_sites if s['adsorbate'] == 'CO'))
        >>> remove_adsorbate_from_site(atoms, site=CO_site)
        >>> view(atoms)

    .. image:: ../images/remove_adsorbate_from_site.png
       :scale: 60 %

    .. autofunction:: remove_adsorbates_from_sites

    **Example**

    To remove all CO species from a fcc111 surface slab covered 
    with both CO and OH:

        >>> from acat.adsorption_sites import SlabAdsorptionSites
        >>> from acat.adsorbate_coverage import SlabAdsorbateCoverage
        >>> from acat.build.patterns import random_coverage_pattern
        >>> from acat.build.actions import remove_adsorbates_from_sites
        >>> from ase.build import fcc111
        >>> from ase.visualize import view
        >>> slab = fcc111('Pt', (6, 6, 4), 4, vacuum=5.)
        >>> slab.center()
        >>> atoms = random_coverage_pattern(slab, adsorbate_species=['OH','CO'],
        ...                                 surface='fcc111',
        ...                                 min_adsorbate_distance=5.)
        >>> sas = SlabAdsorptionSites(atoms, surface='fcc111')
        >>> sac = SlabAdsorbateCoverage(atoms, sas)
        >>> occupied_sites = sac.get_sites(occupied_only=True)
        >>> CO_sites = [s for s in occupied_sites if s['adsorbate'] == 'CO']
        >>> remove_adsorbates_from_sites(atoms, sites=CO_sites)
        >>> view(atoms)

    Output:

    .. image:: ../images/remove_adsorbates_from_sites.png
       :scale: 60 %

    .. autofunction:: remove_adsorbates_too_close

    **Example**

    To remove unphysically close adsorbates on the edges of a Marks 
    decahedron with 0.75 ML symmetric CO coverage:

        >>> from acat.build.patterns import symmetric_coverage_pattern
        >>> from acat.build.actions import remove_adsorbates_too_close
        >>> from ase.cluster import Decahedron
        >>> from ase.visualize import view
        >>> atoms = Decahedron('Pt', p=4, q=3, r=1)
        >>> atoms.center(vacuum=5.)
        >>> pattern = symmetric_coverage_pattern(atoms, adsorbate='CO', 
        ...                                      coverage=0.75)
        >>> remove_adsorbates_too_close(pattern, min_adsorbate_distance=1.)
        >>> view(pattern)

    Output:

    .. image:: ../images/remove_adsorbates_too_close.png

Generate adsorbate coverage patterns
------------------------------------

.. automodule:: acat.build.patterns
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: StochasticPatternGenerator, SystematicPatternGenerator, symmetric_coverage_pattern, full_coverage_pattern, random_coverage_pattern

    .. autoclass:: StochasticPatternGenerator

        .. automethod:: run

    **Example**

    The following example illustrates how to generate 100 stochastic
    adsorbate coverage patterns with CO, OH, CH3 and CHO, based on 
    10 Pt fcc111 surface slabs with random C and O coverages, where 
    CH3 is forbidden to be added to ontop and bridge sites:

        >>> from acat.build.patterns import StochasticPatternGenerator as SPG
        >>> from acat.build.patterns import random_coverage_pattern
        >>> from ase.build import fcc111
        >>> from ase.io import read
        >>> from ase.visualize import view
        >>> slab = fcc111('Pt', (6, 6, 4), 4, vacuum=5.)
        >>> slab.center()
        >>> images = []
        >>> for _ in range(10):
        ...     atoms = slab.copy()
        ...     image = random_coverage_pattern(atoms, adsorbate_species=['C','O'],
        ...                                     surface='fcc111',
        ...                                     min_adsorbate_distance=5.)
        ...     images.append(image)
        >>> spg = SPG(images, adsorbate_species=['CO','OH','CH3','CHO'],
        ...           species_probabilities={'CO':0.3, 'OH': 0.3, 
        ...                                  'CH3': 0.2, 'CHO': 0.2},
        ...           min_adsorbate_distance=1.5, 
        ...           surface='fcc111',
        ...           composition_effect=False, 
        ...           species_forbidden_sites={'CH3': ['ontop','bridge']})
        >>> spg.run(num_gen=100, actions='add')
        >>> images = read('patterns.traj', index=':') 
        >>> view(images)

    Output:

    .. image:: ../images/StochasticPatternGenerator.gif
       :scale: 60 %
       :align: center

    .. autoclass:: SystematicPatternGenerator

        .. automethod:: run

    **Example**

    The following example illustrates how to add CO to all unique sites on 
    a cuboctahedral bimetallic nanoparticle:

        >>> from acat.adsorption_sites import ClusterAdsorptionSites
        >>> from acat.build.patterns import SystematicPatternGenerator as SPG
        >>> from ase.cluster import Octahedron
        >>> from ase.io import read
        >>> from ase.visualize import view
        >>> atoms = Octahedron('Cu', length=7, cutoff=3)
        >>> for atom in atoms:
        ...     if atom.index % 2 == 0:
        ...         atom.symbol = 'Au' 
        >>> atoms.center(vacuum=5.)
        >>> cas = ClusterAdsorptionSites(atoms, composition_effect=True)
        >>> spg = SPG(atoms, adsorbate_species='CO',
        ...           min_adsorbate_distance=2., 
        ...           adsorption_sites=cas,
        ...           composition_effect=True) 
        >>> spg.run(action='add')
        >>> images = read('patterns.traj', index=':') 
        >>> view(images)

    Output:

    .. image:: ../images/SystematicPatternGenerator.gif
 
    .. autofunction:: symmetric_coverage_pattern

    **Example**

    To add a 0.5 ML CO coverage pattern on a cuboctahedron:

        >>> from acat.build.patterns import symmetric_coverage_pattern
        >>> from ase.cluster import Octahedron
        >>> from ase.visualize import view
        >>> atoms = Octahedron('Au', length=9, cutoff=4)
        >>> atoms.center(vacuum=5.)
        >>> pattern = symmetric_coverage_pattern(atoms, adsorbate='CO', 
        ...                                      coverage=0.5)
        >>> view(pattern)

    Output:

    .. image:: ../images/symmetric_coverage_pattern_1.png

    To add a 0.75 ML CO coverage pattern on a fcc111 surface slab:

        >>> from acat.build.patterns import symmetric_coverage_pattern
        >>> from ase.build import fcc111
        >>> from ase.visualize import view
        >>> atoms = fcc111('Cu', (8, 8, 4), vacuum=5.)
        >>> atoms.center()
        >>> pattern = symmetric_coverage_pattern(atoms, adsorbate='CO',
        ...                                      coverage=0.5, 
        ...                                      surface='fcc111')
        >>> view(pattern)

    Output:

    .. image:: ../images/symmetric_coverage_pattern_2.png

    .. autofunction:: full_coverage_pattern

    **Example**

    To add CO to all hcp sites on a icosahedron:

        >>> from acat.build.patterns import full_coverage_pattern
        >>> from ase.cluster import Icosahedron
        >>> from ase.visualize import view
        >>> atoms = Icosahedron('Au', noshells=5)
        >>> atoms.center(vacuum=5.)
        >>> pattern = full_coverage_pattern(atoms, adsorbate='CO', site='hcp')
        >>> view(pattern)

    Output:

    .. image:: ../images/full_coverage_pattern_1.png

    To add CO to all 3fold sites on a bcc110 surface slab:

        >>> from acat.build.patterns import full_coverage_pattern
        >>> from ase.build import bcc110
        >>> from ase.visualize import view
        >>> atoms = bcc110('Mo', (8, 8, 4), vacuum=5.)
        >>> atoms.center()
        >>> pattern = full_coverage_pattern(atoms, adsorbate='CO',
        ...                                 surface='bcc110', site='3fold')
        >>> view(pattern)

    Output:

    .. image:: ../images/full_coverage_pattern_2.png

    .. autofunction:: random_coverage_pattern

    **Example**

    To add CO randomly onto a cuboctahedron with a minimum adsorbate 
    distance of 5 Angstrom:

        >>> from acat.build.patterns import random_coverage_pattern
        >>> from ase.cluster import Octahedron
        >>> from ase.visualize import view
        >>> atoms = Octahedron('Au', length=9, cutoff=4)
        >>> atoms.center(vacuum=5.)
        >>> pattern = random_coverage_pattern(atoms, adsorbate_species='CO', 
        ...                                   min_adsorbate_distance=5.)
        >>> view(pattern)

    Output:

    .. image:: ../images/random_coverage_pattern_1.png

    To add C, N, O randomly onto a hcp0001 surface slab with probabilities 
    of 0.25, 0.25, 0.5, respectively, and a minimum adsorbate distance of 
    2 Angstrom:

        >>> from acat.build.patterns import random_coverage_pattern
        >>> from ase.build import hcp0001
        >>> from ase.visualize import view
        >>> atoms = hcp0001('Ru', (8, 8, 4), vacuum=5.)
        >>> atoms.center()
        >>> pattern = random_coverage_pattern(atoms, adsorbate_species=['C','N','O'],
        ...                                   species_probabilities={'C': 0.25, 
        ...                                                          'N': 0.25, 
        ...                                                          'O': 0.5},
        ...                                   surface='hcp0001',
        ...                                   min_adsorbate_distance=2.)
        >>> view(pattern)

    Output:

    .. image:: ../images/random_coverage_pattern_2.png


Generate chemical orderings
---------------------------

.. automodule:: acat.build.orderings
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: SymmetricOrderingGenerator, RandomOrderingGenerator

    .. autoclass:: SymmetricOrderingGenerator

        .. automethod:: get_nblist_from_center_atom

        .. automethod:: get_shells

        .. automethod:: run

    **Example**

    To generate 100 symmetric chemical orderings of a truncated
    octahedral NiPt nanoalloy with spherical symmetry:

        >>> from acat.build.orderings import SymmetricOrderingGenerator as SOG
        >>> from ase.cluster import Octahedron
        >>> from ase.io import read
        >>> from ase.visualize import view
        >>> atoms = Octahedron('Ni', length=8, cutoff=3)
        >>> sog = SOG(atoms, elements=['Ni', 'Pt'], symmetry='spherical')
        >>> sog.run(max_gen=100, verbose=True)
        >>> images = read('orderings.traj', index=':') 
        >>> view(images)

    Output:

        10 layers classified

        100 symmetric chemical orderings generated 

    .. image:: ../images/SymmetricOrderingGenerator.gif
       :scale: 60 %
       :align: center
               
    .. autoclass:: RandomOrderingGenerator       

        .. automethod:: randint_with_sum

        .. automethod:: random_split_indices

        .. automethod:: run

    **Example**

    To generate 50 random chemical orderings of a icosahedral Ni0.5Pt0.2Au0.3
    nanoalloy:

        >>> from acat.build.orderings import RandomOrderingGenerator as ROG
        >>> from ase.cluster import Icosahedron
        >>> from ase.io import read
        >>> from ase.visualize import view
        >>> atoms = Icosahedron('Ni', noshells=4)
        >>> rog = ROG(atoms, elements=['Ni', 'Pt', 'Au'], 
        ...           composition={'Ni': 0.5, 'Pt': 0.2, 'Au': 0.3})
        >>> rog.run(num_gen=50)
        >>> images = read('orderings.traj', index=':') 
        >>> view(images)

    Output:

    .. image:: ../images/RandomOrderingGenerator.gif
       :scale: 60 %
       :align: center
