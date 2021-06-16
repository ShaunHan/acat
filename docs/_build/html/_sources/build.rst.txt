Building things
===============

Operate adsorbate
-----------------

.. automodule:: acat.build.action
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: add_adsorbate, add_adsorbate_to_site, add_adsorbate_to_label, remove_adsorbate_from_site, remove_adsorbates_from_sites, remove_adsorbates_too_close

The add_adsorbate function
~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: add_adsorbate
               
    **Example**

    To add a NO molecule to a bridge site consists of one Pt and one 
    Ni on the fcc111 surface of a bimetallic truncated octahedron:

        >>> from acat.build.action import add_adsorbate 
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

The add_adsorbate_to_site function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: add_adsorbate_to_site

    **Example1**

    To add CO to all fcc sites of an icosahedral nanoparticle:

        >>> from acat.adsorption_sites import ClusterAdsorptionSites
        >>> from acat.build.action import add_adsorbate_to_site
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

    **Example2**

    To add a bidentate CH3OH to the (54, 57, 58) site on a Pt fcc111 
    surface slab and rotate to the orientation of a neighbor site:

        >>> from acat.adsorption_sites import SlabAdsorptionSites
        >>> from acat.adsorption_sites import get_adsorption_site
        >>> from acat.build.action import add_adsorbate_to_site 
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

The add_adsorbate_to_label function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: add_adsorbate_to_label

    **Example**

    To add a NH molecule to a site with bimetallic label 14 (an hcp 
    CuCuAu site) on a bimetallic fcc110 surface slab:

        >>> from acat.build.action import add_adsorbate_to_label 
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

The remove_adsorbate_from_site function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: remove_adsorbate_from_site

    **Example1**

    To remove a CO molecule from a fcc111 surface slab with one 
    CO and one OH:

        >>> from acat.adsorption_sites import SlabAdsorptionSites
        >>> from acat.adsorbate_coverage import SlabAdsorbateCoverage
        >>> from acat.build.action import add_adsorbate_to_site 
        >>> from acat.build.action import remove_adsorbate_from_site
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

The remove_adsorbates_from_sites function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: remove_adsorbates_from_sites

    **Example**

    To remove all CO species from a fcc111 surface slab covered 
    with both CO and OH:

        >>> from acat.adsorption_sites import SlabAdsorptionSites
        >>> from acat.adsorbate_coverage import SlabAdsorbateCoverage
        >>> from acat.build.overlayer import random_coverage_pattern
        >>> from acat.build.action import remove_adsorbates_from_sites
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

The remove_adsorbates_too_close function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: remove_adsorbates_too_close

    **Example**

    To remove unphysically close adsorbates on the edges of a Marks 
    decahedron with 0.75 ML ordered CO coverage:

        >>> from acat.build.overlayer import ordered_coverage_pattern
        >>> from acat.build.action import remove_adsorbates_too_close
        >>> from ase.cluster import Decahedron
        >>> from ase.visualize import view
        >>> atoms = Decahedron('Pt', p=4, q=3, r=1)
        >>> atoms.center(vacuum=5.)
        >>> pattern = ordered_coverage_pattern(atoms, adsorbate='CO', 
        ...                                    coverage=0.75)
        >>> remove_adsorbates_too_close(pattern, min_adsorbate_distance=1.)
        >>> view(pattern)

    Output:

    .. image:: ../images/remove_adsorbates_too_close.png

Generate adsorbate overlayer patterns
-------------------------------------

.. automodule:: acat.build.overlayer
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: StochasticPatternGenerator, SystematicPatternGenerator, ordered_coverage_pattern, full_coverage_pattern, random_coverage_pattern

The StochasticPatternGenerator class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: StochasticPatternGenerator

        .. automethod:: run

    **Example1**

    The following example illustrates how to generate 100 stochastic
    adsorbate overlayer patterns with one more adsorbate chosen from 
    CO, OH, CH3 and CHO, based on 10 Pt fcc111 surface slabs with 
    random C and O coverages, where CH3 is forbidden to be added to 
    ontop and bridge sites:

        >>> from acat.build.overlayer import StochasticPatternGenerator as SPG
        >>> from acat.build.overlayer import random_coverage_pattern
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
        ...           species_probabilities={'CO': 0.3, 'OH': 0.3, 
        ...                                  'CH3': 0.2, 'CHO': 0.2},
        ...           min_adsorbate_distance=1.5, 
        ...           surface='fcc111',
        ...           composition_effect=False, 
        ...           species_forbidden_sites={'CH3': ['ontop','bridge']})
        >>> spg.run(num_gen=100, action='add')
        >>> images = read('patterns.traj', index=':') 
        >>> view(images)

    Output:

    .. image:: ../images/StochasticPatternGenerator1.gif
       :scale: 60 %
       :align: center

    **Example2**

    The following example illustrates how to generate 20 unique coverage 
    patterns, each adding 4 adsorbates (randomly chosen from H, OH and 
    H2O) onto a fcc100 Ni2Cu surface slab on both top and bottom 
    interfaces (:download:`bulk water <../tests/water_bulk.xyz>` in 
    between) with probabilities of 0.25, 0.25, 0.5, respectively, and a 
    minimum adsorbate distance of 2.5 Angstrom:

        >>> from acat.build.overlayer import StochasticPatternGenerator as SPG
        >>> from acat.adsorption_sites import SlabAdsorptionSites
        >>> from ase.io import read
        >>> from ase.build import fcc100
        >>> from ase.visualize import view
        >>> water_bulk = read('water_bulk.xyz')
        >>> water_bulk.center(vacuum=11., axis=2)
        >>> slab = fcc100('Ni', (4, 4, 8), vacuum=9.5, periodic=True)
        >>> for atom in slab:
        ...     if atom.index % 3 == 0:
        ...         atom.symbol = 'Cu'
        >>> slab.translate(-slab.cell[2] / 2)
        >>> slab.wrap()
        >>> atoms = slab + water_bulk
        >>> sas = SlabAdsorptionSites(atoms, surface='fcc100',
        ...                           composition_effect=True,
        ...                           both_sides=True)
        >>> spg = SPG(atoms, adsorbate_species=['H','OH','OH2'],
        ...           species_probabilities={'H': 0.25, 'OH': 0.25, 'OH2': 0.5},
        ...           min_adsorbate_distance=2.5,
        ...           adsorption_sites=sas,
        ...           surface='fcc100')
        >>> spg.run(num_gen=20, action='add', num_act=4)
        >>> images = read('patterns.traj', index=':')
        >>> view(images) 

    Output:

    .. image:: ../images/StochasticPatternGenerator2.gif
       :scale: 60 %
       :align: center

    **Example3**

    The following example illustrates how to generate 20 stochastic
    adsorbate overlayer patterns with 5 adsorbates chosen from CO, OH 
    and N, on 10 quaternary cuboctahedral nanoalloys with random 
    chemical orderings. The minimum adsorbate distance is set to 3 
    Angstrom and duplicate patterns are allowed (very unlikely for 
    nanoparticles):

        >>> from acat.adsorption_sites import ClusterAdsorptionSites
        >>> from acat.build.overlayer import StochasticPatternGenerator as SPG
        >>> from acat.build.ordering import RandomOrderingGenerator as ROG
        >>> from ase.cluster import Octahedron
        >>> from ase.io import read
        >>> from ase.visualize import view
        >>> atoms = Octahedron('Ni', length=5, cutoff=2)
        >>> atoms.center(vacuum=5.)
        >>> rog = ROG(atoms, elements=['Ni', 'Cu', 'Pt', 'Au'])
        >>> rog.run(num_gen=10)
        >>> particles = read('orderings.traj', index=':')
        >>> spg = SPG(particles, adsorbate_species=['CO','OH','N'],
        ...           min_adsorbate_distance=3.,
        ...           composition_effect=True)
        >>> spg.run(num_gen=20, action='add', num_act=5, unique=False)
        >>> images = read('patterns.traj', index=':')
        >>> view(images)

    Output:

    .. image:: ../images/StochasticPatternGenerator3.gif


The SystematicPatternGenerator class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: SystematicPatternGenerator

        .. automethod:: run

    **Example1**

    The following example illustrates how to add CO to all unique sites on 
    a cuboctahedral bimetallic nanoparticle with a minimum adsorbate
    distance of 2 Angstrom:

        >>> from acat.adsorption_sites import ClusterAdsorptionSites
        >>> from acat.build.overlayer import SystematicPatternGenerator as SPG
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

    .. image:: ../images/SystematicPatternGenerator1.gif

    **Example2**

    The following example illustrates how to enumerate all unique coverage
    patterns consists of 3 adsorbates chosen from C, N and O on a bimetallic 
    bcc111 surface slab with a minimum adsorbate distance of 2 Angstrom 
    (here only generate a maximum of 100 unique patterns):

        >>> from acat.build.overlayer import SystematicPatternGenerator as SPG
        >>> from acat.adsorption_sites import SlabAdsorptionSites
        >>> from ase.io import read
        >>> from ase.build import bcc111
        >>> from ase.visualize import view
        >>> atoms = bcc111('Fe', (2, 2, 12), vacuum=5.)
        >>> for atom in atoms:
        ...     if atom.index % 2 == 0:
        ...         atom.symbol = 'Mo'
        >>> atoms.center()
        >>> sas = SlabAdsorptionSites(atoms, surface='bcc111',
        ...                           composition_effect=True)
        >>> spg = SPG(atoms, adsorbate_species=['C','N','O'],
        ...           min_adsorbate_distance=2.,
        ...           adsorption_sites=sas,
        ...           surface='bcc111',
        ...           composition_effect=True)
        >>> spg.run(max_gen_per_image=100, action='add', num_act=3)
        >>> images = read('patterns.traj', index=':')
        >>> view(images)

    Output:

    .. image:: ../images/SystematicPatternGenerator2.gif
       :scale: 40 %
       :align: center
 
The ordered_coverage_pattern function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: ordered_coverage_pattern

    **Example1**

    To add a 0.5 ML CO overlayer pattern on a cuboctahedron:

        >>> from acat.build.overlayer import ordered_coverage_pattern
        >>> from ase.cluster import Octahedron
        >>> from ase.visualize import view
        >>> atoms = Octahedron('Au', length=9, cutoff=4)
        >>> atoms.center(vacuum=5.)
        >>> pattern = ordered_coverage_pattern(atoms, adsorbate='CO', 
        ...                                    coverage=0.5)
        >>> view(pattern)

    Output:

    .. image:: ../images/ordered_coverage_pattern_1.png

    **Example2**

    To add a 0.75 ML CO overlayer pattern on a fcc111 surface slab:

        >>> from acat.build.overlayer import ordered_coverage_pattern
        >>> from ase.build import fcc111
        >>> from ase.visualize import view
        >>> atoms = fcc111('Cu', (8, 8, 4), vacuum=5.)
        >>> atoms.center()
        >>> pattern = ordered_coverage_pattern(atoms, adsorbate='CO',
        ...                                    coverage=0.5, 
        ...                                    surface='fcc111')
        >>> view(pattern)

    Output:

    .. image:: ../images/ordered_coverage_pattern_2.png

The full_coverage_pattern function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: full_coverage_pattern

    **Example1**

    To add CO to all hcp sites on a icosahedron:

        >>> from acat.build.overlayer import full_coverage_pattern
        >>> from ase.cluster import Icosahedron
        >>> from ase.visualize import view
        >>> atoms = Icosahedron('Au', noshells=5)
        >>> atoms.center(vacuum=5.)
        >>> pattern = full_coverage_pattern(atoms, adsorbate='CO', site='hcp')
        >>> view(pattern)

    Output:

    .. image:: ../images/full_coverage_pattern_1.png

    **Example2**

    To add CO to all 3fold sites on a bcc110 surface slab:

        >>> from acat.build.overlayer import full_coverage_pattern
        >>> from ase.build import bcc110
        >>> from ase.visualize import view
        >>> atoms = bcc110('Mo', (8, 8, 4), vacuum=5.)
        >>> atoms.center()
        >>> pattern = full_coverage_pattern(atoms, adsorbate='CO',
        ...                                 surface='bcc110', site='3fold')
        >>> view(pattern)

    Output:

    .. image:: ../images/full_coverage_pattern_2.png

The random_coverage_pattern function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: random_coverage_pattern

    **Example1**

    To add CO randomly onto a cuboctahedron with a minimum adsorbate 
    distance of 5 Angstrom:

        >>> from acat.build.overlayer import random_coverage_pattern
        >>> from ase.cluster import Octahedron
        >>> from ase.visualize import view
        >>> atoms = Octahedron('Au', length=9, cutoff=4)
        >>> atoms.center(vacuum=5.)
        >>> pattern = random_coverage_pattern(atoms, adsorbate_species='CO', 
        ...                                   min_adsorbate_distance=5.)
        >>> view(pattern)

    Output:

    .. image:: ../images/random_coverage_pattern_1.png

    **Example2**

    To add C, N, O randomly onto a hcp0001 surface slab with probabilities 
    of 0.25, 0.25, 0.5, respectively, and a minimum adsorbate distance of 
    2 Angstrom:

        >>> from acat.build.overlayer import random_coverage_pattern
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

    **Example3**

    To add H, OH and H2O randomly onto a fcc100 Ni2Cu surface slab on both top 
    and bottom interfaces ( :download:`bulk water <../tests/water_bulk.xyz>` 
    in between) with probabilities of 0.25, 0.25, 0.5, respectively, and a 
    minimum adsorbate distance of 2 Angstrom:

        >>> from acat.build.overlayer import random_coverage_pattern                        
        >>> from ase.io import read
        >>> from ase.build import fcc100 
        >>> from ase.visualize import view
        >>> water_bulk = read('water_bulk.xyz')
        >>> water_bulk.center(vacuum=11., axis=2)
        >>> slab = fcc100('Ni', (4, 4, 8), vacuum=9.5, periodic=True)
        >>> for atom in slab:
        ...     if atom.index % 3 == 0:
        ...         atom.symbol = 'Cu'
        >>> slab.translate(-slab.cell[2] / 2)
        >>> slab.wrap()
        >>> atoms = slab + water_bulk
        >>> pattern = random_coverage_pattern(atoms, adsorbate_species=['H','OH','OH2'],
        ...                                   species_probabilities={'H': 0.25,
        ...                                                          'OH': 0.25,
        ...                                                          'OH2': 0.5},
        ...                                   surface='fcc100',
        ...                                   min_adsorbate_distance=2.,
        ...                                   both_sides=True)
        >>> view(pattern) 

    Output:

    .. image:: ../images/random_coverage_pattern_3.png

Generate alloy chemical orderings
---------------------------------

.. automodule:: acat.build.ordering
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: SymmetricClusterOrderingGenerator, RandomOrderingGenerator

The SymmetricClusterOrderingGenerator class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: SymmetricClusterOrderingGenerator

        .. automethod:: get_sorted_indices

        .. automethod:: get_groups

        .. automethod:: run

    **Example1**

    To generate 100 symmetric chemical orderings for truncated
    octahedral NiPt nanoalloys with spherical symmetry:

        >>> from acat.build.ordering import SymmetricClusterOrderingGenerator as SCOG
        >>> from ase.cluster import Octahedron
        >>> from ase.io import read
        >>> from ase.visualize import view
        >>> atoms = Octahedron('Ni', length=8, cutoff=3)
        >>> scog = SCOG(atoms, elements=['Ni', 'Pt'], symmetry='spherical')
        >>> scog.run(max_gen=100, verbose=True)
        >>> images = read('orderings.traj', index=':') 
        >>> view(images)

    Output:

        | 10 symmetry-equivalent groups classified
        | 100 symmetric chemical orderings generated 

    .. image:: ../images/SymmetricClusterOrderingGenerator1.gif
       :scale: 60 %
       :align: center

    **Example2**
               
    To systematically generate 50 symmetric chemical orderings for 
    quaternary truncated octahedral Ni0.4Cu0.3Pt0.2Au0.1 nanoalloys 
    with mirror circular symmetry:

        >>> from acat.build.ordering import SymmetricClusterOrderingGenerator as SCOG
        >>> from ase.cluster import Octahedron
        >>> from ase.io import read
        >>> from ase.visualize import view
        >>> atoms = Octahedron('Ni', 7, 2)
        >>> scog = SCOG(atoms, elements=['Ni', 'Cu', 'Pt', 'Au'],
        ...             symmetry='mirror_circular',
        ...             composition={'Ni': 0.4, 'Cu': 0.3, 'Pt': 0.2, 'Au': 0.1})
        >>> scog.run(max_gen=50, mode='systematic', verbose=True)
        >>> images = read('orderings.traj', index=':')
        >>> view(images)

    Output:

        | 25 symmetry-equivalent groups classified
        | 50 symmetric chemical orderings generated

    .. image:: ../images/SymmetricClusterOrderingGenerator2.gif
       :scale: 60 %
       :align: center

    **Example3**

    To stochastically generate 50 symmetric chemical orderings for 
    quaternary truncated octahedral Ni0.4Cu0.3Pt0.2Au0.1 nanoalloys 
    with mirror circular symmetry:

        >>> from acat.build.ordering import SymmetricClusterOrderingGenerator as SCOG
        >>> from ase.cluster import Octahedron
        >>> from ase.io import read
        >>> from ase.visualize import view
        >>> atoms = Octahedron('Ni', 7, 2)
        >>> scog = SCOG(atoms, elements=['Ni', 'Cu', 'Pt', 'Au'],
        ...             symmetry='mirror_circular',
        ...             composition={'Ni': 0.4, 'Cu': 0.3, 'Pt': 0.2, 'Au': 0.1})
        >>> scog.run(max_gen=50, mode='stochastic', verbose=True)
        >>> images = read('orderings.traj', index=':')
        >>> view(images)

    Output:

        | 25 symmetry-equivalent groups classified
        | 50 symmetric chemical orderings generated

    .. image:: ../images/SymmetricClusterOrderingGenerator3.gif
       :scale: 60 %
       :align: center

    **Example4**

    Sometimes it is also useful to get the structure of each group. 
    For instance, to visualize the concentric shells of a truncated 
    octahedral nanoparticle:

        >>> from acat.build.ordering import SymmetricClusterOrderingGenerator as SCOG
        >>> from ase.cluster import Octahedron
        >>> from ase.visualize import view
        >>> atoms = Octahedron('Ni', 12, 5)
        >>> scog = SCOG(atoms, elements=['Ni', 'Pt'], symmetry='concentric')
        >>> groups = scog.get_groups()
        >>> images = [atoms[g] for g in groups]
        >>> view(images)

    Output:

    .. image:: ../images/SymmetricClusterOrderingGenerator4.gif
       :scale: 60 %
       :align: center

The OrderedSlabOrderingGenerator class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: OrderedSlabOrderingGenerator

        .. automethod:: get_groups

        .. automethod:: run

    **Example1**

    To stochastically generate 50 ordered chemical orderings for
    ternary NixPtyAu1-x-y fcc111 surface slabs:

        >>> from acat.build.ordering import OrderedSlabOrderingGenerator as OSOG 
        >>> from ase.build import fcc111
        >>> from ase.io import read
        >>> from ase.visualize import view
        >>> atoms = fcc111('Ni', (6, 6, 4), vacuum=5.)
        >>> atoms.center()
        >>> osog = OSOG(atoms, elements=['Ni', 'Pt', 'Au'],
        ...             repeating_size=(3, 3)) 
        >>> osog.run(max_gen=50, mode='stochastic', verbose=True)
        >>> images = read('orderings.traj', index=':')
        >>> view(images)

    Output:

        | 16 symmetry-equivalent groups classified
        | 50 ordered chemical orderings generated 

    .. image:: ../images/OrderedSlabOrderingGenerator1.gif
       :scale: 60 %
       :align: center

    **Example2**
               
    To systematically generate 50 ordered chemical orderings for 
    Ni0.75Pt0.25 fcc110 surface slabs:

        >>> from acat.build.ordering import OrderedSlabOrderingGenerator as OSOG 
        >>> from ase.build import fcc110
        >>> from ase.io import read
        >>> from ase.visualize import view
        >>> atoms = fcc110('Ni', (4, 4, 4), vacuum=5.)
        >>> atoms.center()
        >>> osog = OSOG(atoms, elements=['Ni', 'Pt'],
        ...             composition={'Ni': 0.75, 'Pt': 0.25},
        ...             repeating_size=(2, 2)) 
        >>> osog.run(max_gen=50, mode='systematic', verbose=True)
        >>> images = read('orderings.traj', index=':')
        >>> view(images)

    Output:

        | 16 symmetry-equivalent groups classified
        | 50 ordered chemical orderings generated

    .. image:: ../images/OrderedSlabOrderingGenerator2.gif
       :scale: 60 %
       :align: center

    **Example3**

    To stochastically generate 50 ordered chemical orderings for 
    Ni0.75Pt0.25 fcc110 surface slabs:

        >>> from acat.build.ordering import OrderedSlabOrderingGenerator as OSOG 
        >>> from ase.build import fcc110
        >>> from ase.io import read
        >>> from ase.visualize import view
        >>> atoms = fcc110('Ni', (4, 4, 4), vacuum=5.)
        >>> atoms.center()
        >>> osog = OSOG(atoms, elements=['Ni', 'Pt'],
        ...             composition={'Ni': 0.75, 'Pt': 0.25},
        ...             repeating_size=(2, 2)) 
        >>> osog.run(max_gen=50, mode='stochastic', verbose=True)
        >>> images = read('orderings.traj', index=':')
        >>> view(images)

    Output:

        | 16 symmetry-equivalent groups classified
        | 50 ordered chemical orderings generated

    .. image:: ../images/OrderedSlabOrderingGenerator3.gif
       :scale: 60 %
       :align: center

The RandomOrderingGenerator class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: RandomOrderingGenerator       

        .. automethod:: randint_with_sum

        .. automethod:: random_split_indices

        .. automethod:: run

    **Example1**

    To generate 50 random chemical orderings for icosahedral 
    Ni0.5Pt0.2Au0.3 nanoalloys:

        >>> from acat.build.ordering import RandomOrderingGenerator as ROG
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

    .. image:: ../images/RandomOrderingGenerator1.gif
       :scale: 60 %
       :align: center

    **Example2**

    To generate 50 random chemical orderings for Pt0.5Au0.5 
    fcc111 surface slabs:

        >>> from acat.build.ordering import RandomOrderingGenerator as ROG 
        >>> from ase.build import fcc111
        >>> from ase.io import read
        >>> from ase.visualize import view
        >>> atoms = fcc111('Ni', (4, 4, 4), vacuum=5.)
        >>> atoms.center()
        >>> rog = ROG(atoms, elements=['Pt', 'Au'],
        ...           composition={'Pt': 0.5, 'Au': 0.5})
        >>> rog.run(num_gen=50)
        >>> images = read('orderings.traj', index=':')
        >>> view(images)

    Output:

    .. image:: ../images/RandomOrderingGenerator2.gif
       :scale: 60 %
       :align: center
