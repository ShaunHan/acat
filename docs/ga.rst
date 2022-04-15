Genetic algorithm
=================

Symmetry-constrained genetic algorithm for nanoalloys
-----------------------------------------------------

.. automodule:: acat.ga.group_operators
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: AdsorbateGroupSubstitute, AdsorbateGroupPermutation

.. automodule:: acat.ga.group_comparators
    :members:
    :undoc-members:
    :show-inheritance:

**Example1**

All the group operators and comparators can be easily used with other indexing-preserved operators and comparators. All operators can be used for both non-periodic nanoparticles and periodic surface slabs. To accelerate the GA, provide the symmetry-equivalent group (or all possible groups).

As an example we will find the convex hull of ternary NixPtyAu405-x-y truncated octahedral nanoalloys using the ASAP EMT calculator. **Note that we must first align the symmetry axis of interest to the z direction.** Here we want to study the 3-fold mirror circular symmetry around the C3 axis of the particle.

The script for a parallel symmetry-constrained genetic algorithm (SCGA) looks as follows:

.. code-block:: python

    from acat.build.ordering import SymmetricClusterOrderingGenerator as SCOG
    from acat.ga.group_operators import (GroupSubstitute,
                                         GroupPermutation,
                                         GroupCrossover)
    from acat.ga.group_comparators import (GroupSizeComparator,
                                           GroupCompositionComparator)
    from ase.ga.particle_comparator import NNMatComparator
    from ase.ga.standard_comparators import SequentialComparator
    from ase.ga.offspring_creator import OperationSelector
    from ase.ga.population import Population, RankFitnessPopulation
    from ase.ga.convergence import GenerationRepetitionConvergence
    from ase.ga.data import DataConnection, PrepareDB
    from ase.ga.utilities import get_nnmat
    from ase.io import read, write
    from ase.cluster import Octahedron
    from ase.optimize import BFGS
    from asap3 import EMT as asapEMT
    from multiprocessing import Pool
    from random import randint
    import numpy as np
    import os
    
    # Define population. 
    # Recommend to choose a number that is a multiple of the number of cpu
    pop_size = 100
        
    # Build and rotate the particle so that C3 axis is aligned to z direction
    particle = Octahedron('Ni', length=9, cutoff=3)
    particle.rotate(45, 'x')     
    particle.rotate(-35.29, 'y') 
    particle.center(vacuum=5.)                                              

    # Generate 100 truncated ocatahedral NixPtyAu405-x-y nanoalloys with
    # mirror circular symmetry. Get the groups at the same time.
    scog = SCOG(particle, elements=['Ni', 'Pt', 'Au'],
                symmetry='mirror_circular',
                trajectory='starting_generation.traj')
    scog.run(max_gen=pop_size, mode='stochastic', verbose=True)
    groups = scog.get_groups()
    images = read('starting_generation.traj', index=':')
    
    # Instantiate the db
    db_name = 'ridge_mirror_circular_NiPtAu_TO405_C3.db'
    
    db = PrepareDB(db_name, cell=particle.cell, population_size=pop_size)
    
    for atoms in images:
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
    
    # Connect to the db
    db = DataConnection(db_name)
    
    # Define operators
    soclist = ([2, 2, 3], 
               [GroupSubstitute(groups, elements=['Ni', 'Pt', 'Au'], num_muts=3),
                GroupPermutation(groups, elements=['Ni', 'Pt', 'Au'], num_muts=1),                              
                GroupCrossover(groups, elements=['Ni', 'Pt', 'Au']),]) 
    op_selector = OperationSelector(*soclist)
    
    # Define comparators
    comp = SequentialComparator([GroupSizeComparator(groups, ['Ni', 'Pt', 'Au']),
                                 NNMatComparator(0.2, ['Ni', 'Pt', 'Au'])],
                                [0.5, 0.5])
    
    def vf(atoms):
        """Returns the descriptor that distinguishes candidates in the 
        niched population."""
        return atoms.get_chemical_formula(mode='hill')
    
    # Give fittest candidates at different compositions equal fitness.
    # Use this to find global minimum at each composition
    pop = RankFitnessPopulation(data_connection=db,
                                population_size=pop_size,
                                comparator=comp,
                                variable_function=vf,
                                exp_function=True,
                                logfile='log.txt')
    
    # Normal fitness ranking irrespective of composition
    #pop = Population(data_connection=db, 
    #                 population_size=pop_size, 
    #                 comparator=comp, 
    #                 logfile='log.txt')
    
    # Set convergence criteria
    cc = GenerationRepetitionConvergence(pop, 5)
    
    # Calculate the relaxed energies for pure Ni405, Pt405 and Au405
    pure_pots = {'Ni': 147.532, 'Pt':  86.892, 'Au': 63.566}
    
    # Define the relax function
    def relax(atoms, single_point=False):    
        atoms.center(vacuum=5.)   
        atoms.calc = asapEMT()
        if not single_point:
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=0.1)
        Epot = atoms.get_potential_energy()
        atoms.info['key_value_pairs']['potential_energy'] = Epot
    
        # There is a known issue of asapEMT in GA. You can either detach 
        # the calculator or re-assign to a SinglePointCalculator
        atoms.calc = None
    
        # Calculate mixing energy 
        syms = atoms.get_chemical_symbols()
        for a in set(syms):
            Epot -= (pure_pots[a] / len(atoms)) * syms.count(a)
        atoms.info['key_value_pairs']['raw_score'] = -Epot

        # Parallelize nnmat calculations to accelerate NNMatComparator
        atoms.info['data']['nnmat'] = get_nnmat(atoms)
    
        return atoms

    # Relax starting generation
    def relax_an_unrelaxed_candidate(atoms):
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        nncomp = atoms.get_chemical_formula(mode='hill')
        print('Relaxing ' + nncomp)

        return relax(atoms)
    
    # Create a multiprocessing Pool
    pool = Pool(os.cpu_count())
    # Perform relaxations in parallel. Especially
    # useful when running GA on large nanoparticles  
    relaxed_candidates = pool.map(relax_an_unrelaxed_candidate, 
                                  db.get_all_unrelaxed_candidates())  
    pool.close()
    pool.join()
    db.add_more_relaxed_candidates(relaxed_candidates)
    pop.update()

    # Number of generations
    num_gens = 1000
    
    # Below is the iterative part of the algorithm
    gen_num = db.get_generation_number()
    for i in range(num_gens):
        # Check if converged
        if cc.converged():
            print('Converged')
            break             
        print('Creating and evaluating generation {0}'.format(gen_num + i))
    
        # Performing procreations in parallel
        def procreation(x):
            # Select an operator and use it
            op = op_selector.get_operator()
            while True:
                # Assign rng with a random seed
                np.random.seed(randint(1, 10000))
                pop.rng = np.random 
                # Select parents for a new candidate
                p1, p2 = pop.get_two_candidates()
                # Pure and binary candidates are not considered
                if len(set(p1.numbers)) < 3:
                    continue 
                parents = [p1, p2]
                offspring, desc = op.get_new_individual(parents)
                # An operator could return None if an offspring cannot be formed
                # by the chosen parents
                if offspring is not None:
                    break
            nncomp = offspring.get_chemical_formula(mode='hill')
            print('Relaxing ' + nncomp)        
            if 'data' not in offspring.info:
                offspring.info['data'] = {'tag': None}
            
            return relax(offspring)
            
        # Create a multiprocessing Pool
        pool = Pool(os.cpu_count())
        # Perform procreations in parallel. Especially
        # useful when running GA on large nanoparticles  
        relaxed_candidates = pool.map(procreation, range(pop_size))  
        pool.close()
        pool.join()
        db.add_more_relaxed_candidates(relaxed_candidates)    

        # Update the population to allow new candidates to enter
        pop.update()

**Example2**

If we want to study the same system but target 3 compositions: Ni0.5Pt0.25Au0.25, Ni0.25Pt0.5Au0.25 and Ni0.25Pt0.25Au0.5, we should not use ``GroupSubstitute`` operator and set keep_composition to True in ``GroupPermutation`` and ``GroupCrossover`` operators. The tolerance of the intitial compositions can be controlled by the eps parameter in ``SymmetricClusterOrderingGenerator.run``.

The script for a fixed-composition parallel genetic algorithm now looks as follows:

.. code-block:: python
                                                                                                                    
    from acat.build.ordering import SymmetricClusterOrderingGenerator as SCOG
    from acat.ga.group_operators import (GroupSubstitute,
                                         GroupPermutation,
                                         GroupCrossover)
    from acat.ga.group_comparators import (GroupSizeComparator,
                                           GroupCompositionComparator)
    from ase.ga.particle_comparator import NNMatComparator
    from ase.ga.standard_comparators import SequentialComparator
    from ase.ga.offspring_creator import OperationSelector
    from ase.ga.population import Population, RankFitnessPopulation
    from ase.ga.convergence import GenerationRepetitionConvergence
    from ase.ga.data import DataConnection, PrepareDB
    from ase.ga.utilities import get_nnmat
    from ase.io import read, write
    from ase.cluster import Octahedron
    from ase.optimize import BFGS
    from asap3 import EMT as asapEMT
    from multiprocessing import Pool
    from random import randint
    import numpy as np
    import os
    
    # Define population. 
    # Recommend to choose a number that is a multiple of the number of cpu
    pop_size = 100
 
    # Build and rotate the particle so that C3 axis is aligned to z direction
    particle = Octahedron('Ni', length=9, cutoff=3)
    particle.rotate(45, 'x')     
    particle.rotate(-35.29, 'y') 
    particle.center(vacuum=5.)                                              

    # Generate 100 truncated ocatahedral NixPtyAu405-x-y nanoalloys with
    # mirror circular symmetry and concentrations of {x=0.5, y=0.25}, 
    # {x=0.25, y=0.5} and {x=y=0.25}. The concentrations are allowed to
    # vary by a range of 2*eps=0.1. Get the groups at the same time.
    scog1 = SCOG(particle, elements=['Ni', 'Pt'],
                 symmetry='mirror_circular',
                 cutoff=1.,
                 secondary_symmetry='chemical',
                 secondary_cutoff=.1,
                 composition={'Ni': 0.5, 'Pt': 0.25, 'Au': 0.25},
                 trajectory='starting_generation.traj')
    groups = scog1.get_groups()
    scog1.run(max_gen=33, mode='stochastic', eps=0.05, verbose=True)
    
    scog2 = SCOG(particle, elements=['Ni', 'Pt'],
                 symmetry='mirror_circular',
                 cutoff=1.,
                 secondary_symmetry='chemical',
                 secondary_cutoff=.1,
                 composition={'Ni': 0.25, 'Pt': 0.5, 'Au': 0.25},
                 trajectory='starting_generation.traj',
                 append_trajectory=True)
    scog2.run(max_gen=33, mode='stochastic', eps=0.05, verbose=True)
    
    scog3 = SCOG(particle, elements=['Ni', 'Pt'],
                 symmetry='mirror_circular',
                 cutoff=1.,
                 secondary_symmetry='chemical',
                 secondary_cutoff=.1,
                 composition={'Ni': 0.25, 'Pt': 0.25, 'Au': 0.5},
                 trajectory='starting_generation.traj',
                 append_trajectory=True)
    scog3.run(max_gen=34, mode='stochastic', eps=0.05, verbose=True)

    images = read('starting_generation.traj', index=':')
    
    # Instantiate the db
    db_name = 'ridge_mirror_circular_NiPtAu_TO405_C3.db'
    
    db = PrepareDB(db_name, cell=particle.cell, population_size=pop_size)
    
    for atoms in images:
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
    
    # Connect to the db
    db = DataConnection(db_name)
    
    # Define operators, now set keep_composition=True
    # GroupSubstitute cannot keep the composition so it's not used
    soclist = ([4, 6], 
               [GroupPermutation(groups, elements=['Ni', 'Pt', 'Au'], 
                                 keep_composition=True, num_muts=1),                              
                GroupCrossover(groups, elements=['Ni', 'Pt', 'Au'],
                               keep_composition=True),])    
    op_selector = OperationSelector(*soclist)
    
    # Define comparators
    comp = SequentialComparator([GroupSizeComparator(groups, ['Ni', 'Pt', 'Au']),
                                 NNMatComparator(0.2, ['Ni', 'Pt', 'Au'])],
                                [0.5, 0.5])
    
    def vf(atoms):
        """Returns the descriptor that distinguishes candidates in the 
        niched population."""
        return atoms.get_chemical_formula(mode='hill')
    
    # Give fittest candidates at different compositions equal fitness.
    # Use this to find global minimum at each composition
    pop = RankFitnessPopulation(data_connection=db,
                                population_size=pop_size,
                                comparator=comp,
                                variable_function=vf,
                                exp_function=True,
                                logfile='log.txt')
    
    # Normal fitness ranking irrespective of composition
    #pop = Population(data_connection=db, 
    #                 population_size=pop_size, 
    #                 comparator=comp, 
    #                 logfile='log.txt')
    
    # Set convergence criteria
    cc = GenerationRepetitionConvergence(pop, 5)
    
    # Calculate the relaxed energies for pure Ni405, Pt405 and Au405
    pure_pots = {'Ni': 147.532, 'Pt':  86.892, 'Au': 63.566}
    
    # Define the relax function
    def relax(atoms, single_point=False):    
        atoms.center(vacuum=5.)   
        atoms.calc = asapEMT()
        if not single_point:
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=0.1)
        Epot = atoms.get_potential_energy()
        atoms.info['key_value_pairs']['potential_energy'] = Epot
    
        # There is a known issue of asapEMT in GA. You can either detach 
        # the calculator or re-assign to a SinglePointCalculator
        atoms.calc = None
    
        # Calculate mixing energy 
        syms = atoms.get_chemical_symbols()
        for a in set(syms):
            Epot -= (pure_pots[a] / len(atoms)) * syms.count(a)
        atoms.info['key_value_pairs']['raw_score'] = -Epot

        # Parallelize nnmat calculations to accelerate NNMatComparator
        atoms.info['data']['nnmat'] = get_nnmat(atoms)
    
        return atoms

    # Relax starting generation
    def relax_an_unrelaxed_candidate(atoms):
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        nncomp = atoms.get_chemical_formula(mode='hill')
        print('Relaxing ' + nncomp)

        return relax(atoms)
    
    # Create a multiprocessing Pool
    pool = Pool(os.cpu_count())
    # Perform relaxations in parallel. Especially
    # useful when running GA on large nanoparticles  
    relaxed_candidates = pool.map(relax_an_unrelaxed_candidate, 
                                  db.get_all_unrelaxed_candidates())  
    pool.close()
    pool.join()
    db.add_more_relaxed_candidates(relaxed_candidates)
    pop.update()

    # Number of generations
    num_gens = 1000
    
    # Below is the iterative part of the algorithm
    gen_num = db.get_generation_number()
    for i in range(num_gens):
        # Check if converged
        if cc.converged():
            print('Converged')
            break             
        print('Creating and evaluating generation {0}'.format(gen_num + i))
    
        # Performing procreations in parallel
        def procreation(x):
            # Select an operator and use it
            op = op_selector.get_operator()
            while True:
                # Assign rng with a random seed
                np.random.seed(randint(1, 10000))
                pop.rng = np.random 
                # Select parents for a new candidate
                p1, p2 = pop.get_two_candidates()
                # Pure and binary candidates are not considered
                if len(set(p1.numbers)) < 3:
                    continue 
                parents = [p1, p2]
                offspring, desc = op.get_new_individual(parents)
                # An operator could return None if an offspring cannot be formed
                # by the chosen parents
                if offspring is not None:
                    break
            nncomp = offspring.get_chemical_formula(mode='hill')
            print('Relaxing ' + nncomp)        
            if 'data' not in offspring.info:
                offspring.info['data'] = {'tag': None}

            return relax(offspring)
            
        # Create a multiprocessing Pool
        pool = Pool(os.cpu_count())
        # Perform procreations in parallel. Especially
        # useful when running GA on large nanoparticles  
        relaxed_candidates = pool.map(procreation, range(pop_size))  
        pool.close()
        pool.join()
        db.add_more_relaxed_candidates(relaxed_candidates)    

        # Update the population to allow new candidates to enter
        pop.update()

Genetic algorithm for adlayer patterns                                                       
--------------------------------------

.. automodule:: acat.ga.adsorbate_operators
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: get_all_adsorbate_indices, get_numbers, get_atoms_without_adsorbates

.. automodule:: acat.ga.adsorbate_comparators
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: acat.ga.graph_comparators
    :members:
    :undoc-members:
    :show-inheritance:

**Example1**

All the adsorbate operators and comparators can be easily used with other operators and comparators. ``AddAdsorbate``, ``RemoveAdsorbate``, ``MoveAdsorbate``, ``ReplaceAdsorbate``, ``ReplaceAdsorbateSpecies`` and ``CatalystAdsorbateCrossover`` operators can be used for both non-periodic nanoparticles and periodic surface slabs. ``CutSpliceCrossoverWithAdsorbates`` and ``SimpleCutSpliceCrossoverWithAdsorbates`` operators only work for nanoparticles, and the latter is recommended. To strictly sort out duplicate structures, consider using ``AdsorptionGraphComparator`` or ``WLGraphComparator``. To accelerate the GA, provide adsorsption sites and use indexing-preserved operators implemented in ACAT.

As an example we will simultaneously optimize both the adsorbate overlayer pattern and the catalyst chemical ordering of a Ni110Pt37 icosahedral nanoalloy with adsorbate species of H, C, O, OH, CO, CH, CH2 and CH3 using the EMT calculator.

The script for a parallel genetic algorithm looks as follows:

.. code-block:: python
   
    from acat.settings import adsorbate_elements
    from acat.adsorption_sites import ClusterAdsorptionSites
    from acat.adsorbate_coverage import ClusterAdsorbateCoverage
    from acat.build.ordering import RandomOrderingGenerator as ROG
    from acat.build.adlayer import min_dist_coverage_pattern
    from acat.ga.adsorbate_operators import (AddAdsorbate, RemoveAdsorbate,
                                             MoveAdsorbate, ReplaceAdsorbate,
                                             SimpleCutSpliceCrossoverWithAdsorbates)
    # Import particle_mutations from acat instead of ase to get the indexing-preserved version
    from acat.ga.particle_mutations import (RandomPermutation, COM2surfPermutation,
                                            Rich2poorPermutation, Poor2richPermutation)
    from acat.ga.adsorbate_comparators import AdsorptionSitesComparator
    from ase.ga.particle_comparator import NNMatComparator
    from ase.ga.standard_comparators import SequentialComparator
    from ase.ga.offspring_creator import OperationSelector
    from ase.ga.population import Population, RankFitnessPopulation
    from ase.ga.convergence import GenerationRepetitionConvergence
    from ase.ga.utilities import closest_distances_generator, get_nnmat
    from ase.ga.data import DataConnection, PrepareDB
    from ase.io import read, write
    from ase.cluster import Icosahedron
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS
    from collections import defaultdict
    from random import uniform, randint
    from multiprocessing import Pool
    import numpy as np
    import time
    import os
    
    # Define population
    # Recommend to choose a number that is a multiple of the number of cpu
    pop_size = 50
        
    # Generate 50 icosahedral Ni110Pt37 nanoparticles with random orderings
    particle = Icosahedron('Ni', noshells=4)
    particle.center(vacuum=5.)
    rog = ROG(particle, elements=['Ni', 'Pt'],
              composition={'Ni': 0.75, 'Pt': 0.25},
              trajectory='starting_generation.traj')
    rog.run(num_gen=pop_size)
    
    # Generate random coverage on each nanoparticle
    species = ['H', 'C', 'O', 'OH', 'CO', 'CH', 'CH2', 'CH3']
    images = read('starting_generation.traj', index=':')
    patterns = []
    for atoms in images:
        dmin = uniform(3.5, 8.5)
        pattern = min_dist_coverage_pattern(atoms, adsorbate_species=species,
                                            min_adsorbate_distance=dmin)
        patterns.append(pattern)
    
    # Get the adsorption sites. Composition does not affect GA operations
    sas = ClusterAdsorptionSites(particle, composition_effect=False)
    
    # Instantiate the db
    db_name = 'ridge_Ni110Pt37_ads.db'
    
    db = PrepareDB(db_name, cell=particle.cell, population_size=pop_size)
    
    for atoms in patterns:
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
    
    # Connect to the db
    db = DataConnection(db_name)
    
    # Define operators
    soclist = ([1, 1, 2, 1, 1, 1, 1, 2],
               [Rich2poorPermutation(elements=['Ni', 'Pt'], num_muts=5),
                Poor2richPermutation(elements=['Ni', 'Pt'], num_muts=5),
                RandomPermutation(elements=['Ni', 'Pt'], num_muts=5),
                AddAdsorbate(species, adsorption_sites=sas, num_muts=5),
                RemoveAdsorbate(species, adsorption_sites=sas, num_muts=5),
                MoveAdsorbate(species, adsorption_sites=sas, num_muts=5),
                ReplaceAdsorbate(species, adsorption_sites=sas, num_muts=5),
                SimpleCutSpliceCrossoverWithAdsorbates(species, keep_composition=True,
                                                       adsorption_sites=sas),])
    op_selector = OperationSelector(*soclist)
    
    # Define comparators
    comp = SequentialComparator([AdsorptionSitesComparator(10),
                                 NNMatComparator(0.2, ['Ni', 'Pt'])],
                                [0.5, 0.5])
    
    def get_ads(atoms):
        """Returns a list of adsorbate names and corresponding indices."""
    
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        if 'adsorbates' in atoms.info['data']:
            adsorbates = atoms.info['data']['adsorbates']
        else:
            cac = ClusterAdsorbateCoverage(atoms)
            adsorbates = [t[0] for t in cac.get_adsorbates()]
    
        return adsorbates
    
    def vf(atoms):
        """Returns the descriptor that distinguishes candidates in the
        niched population."""
    
        return len(get_ads(atoms))
    
    # Give fittest candidates at different coverages equal fitness.
    # Use this to find global minimum at each adsorbate coverage
    pop = RankFitnessPopulation(data_connection=db,
                                population_size=pop_size,
                                comparator=comp,
                                variable_function=vf,
                                exp_function=True,
                                logfile='log.txt')
    
    # Normal fitness ranking irrespective of adsorabte coverage
    #pop = Population(data_connection=db,
    #                 population_size=pop_size,
    #                 comparator=comp,
    #                 logfile='log.txt')
    
    # Set convergence criteria
    cc = GenerationRepetitionConvergence(pop, 5)
    
    # Calculate chemical potentials
    chem_pots = {'CH4': -24.039, 'H2O': -14.169, 'H2': -6.989}
    
    # Define the relax function
    def relax(atoms, single_point=False):
        atoms.center(vacuum=5.)
        atoms.calc = EMT()
        if not single_point:
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=0.1)
        time.sleep(1) # Add delay (only for testing)
    
        Epot = atoms.get_potential_energy()
        num_H = len([s for s in atoms.symbols if s == 'H'])
        num_C = len([s for s in atoms.symbols if s == 'C'])
        num_O = len([s for s in atoms.symbols if s == 'O'])
        mutot = num_C * chem_pots['CH4'] + num_O * chem_pots['H2O'] + (
                num_H - 4 * num_C - 2 * num_O) * chem_pots['H2'] / 2
        f = -(Epot - mutot)

        atoms.info['key_value_pairs']['raw_score'] = f
        atoms.info['key_value_pairs']['potential_energy'] = Epot 

        # Parallelize nnmat calculations to accelerate NNMatComparator
        atoms.info['data']['nnmat'] = get_nnmat(atoms)

        return atoms
    
    # Relax starting generation
    def relax_an_unrelaxed_candidate(atoms):
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        nncomp = atoms.get_chemical_formula(mode='hill')
        print('Relaxing ' + nncomp)

        return relax(atoms, single_point=True) # Single point only for testing
        
    # Create a multiprocessing Pool
    pool = Pool(os.cpu_count())
    # Perform relaxations in parallel. Especially
    # useful when running GA on large nanoparticles
    relaxed_candidates = pool.map(relax_an_unrelaxed_candidate, 
                                  db.get_all_unrelaxed_candidates())
    pool.close()
    pool.join()
    db.add_more_relaxed_candidates(relaxed_candidates)
    pop.update()
    
    # Number of generations
    num_gens = 1000
    
    # Below is the iterative part of the algorithm
    gen_num = db.get_generation_number()
    for i in range(num_gens):
        # Check if converged
        if cc.converged():
            print('Converged')
            break
        print('Creating and evaluating generation {0}'.format(gen_num + i))
    
        def procreation(x):
            # Select an operator and use it
            op = op_selector.get_operator()
            while True:
                # Assign rng with a random seed
                np.random.seed(randint(1, 10000))
                pop.rng = np.random 
                # Select parents for a new candidate            
                p1, p2 = pop.get_two_candidates()
                parents = [p1, p2]
                # Pure or bare nanoparticles are not considered
                if len(set(p1.numbers)) < 3:
                    continue
                offspring, desc = op.get_new_individual(parents)
                # An operator could return None if an offspring cannot be formed
                # by the chosen parents
                if offspring is not None:
                    break
            nncomp = offspring.get_chemical_formula(mode='hill')
            print('Relaxing ' + nncomp)
            if 'data' not in offspring.info:
                offspring.info['data'] = {'tag': None}

            return relax(offspring, single_point=True) # Single point only for testing
            
        # Create a multiprocessing Pool
        pool = Pool(os.cpu_count())
        # Perform procreations in parallel. Especially useful when
        # using adsorbate operators which requires site identification
        relaxed_candidates = pool.map(procreation, range(pop_size))
        pool.close()
        pool.join()
        db.add_more_relaxed_candidates(relaxed_candidates)
    
        # Update the population to allow new candidates to enter
        pop.update()

**Example2**

If we want to study the same system but fix the adsorbate coverage to be exactly 20 adsorbates, we should only use ``MoveAdsorbate``, ``ReplaceAdsorbate`` and ``SimpleCutSpliceCrossoverWithAdsorbates`` operators with same particle operators. Remember to set fix_coverage to True in ``SimpleCutSpliceCrossoverWithAdsorbates``.

The script for a fixed-coverage parallel genetic algorithm now looks as follows:

.. code-block:: python

    from acat.settings import adsorbate_elements
    from acat.adsorption_sites import ClusterAdsorptionSites
    from acat.adsorbate_coverage import ClusterAdsorbateCoverage
    from acat.build.ordering import RandomOrderingGenerator as ROG
    from acat.build.adlayer import RandomPatternGenerator as RPG
    from acat.ga.adsorbate_operators import (AddAdsorbate, RemoveAdsorbate,
                                             MoveAdsorbate, ReplaceAdsorbate,
                                             SimpleCutSpliceCrossoverWithAdsorbates)
    # Import particle_mutations from acat instead of ase to get the indexing-preserved version
    from acat.ga.particle_mutations import (RandomPermutation, COM2surfPermutation,
                                            Rich2poorPermutation, Poor2richPermutation)
    from acat.ga.adsorbate_comparators import AdsorptionSitesComparator
    from ase.ga.particle_comparator import NNMatComparator
    from ase.ga.standard_comparators import SequentialComparator
    from ase.ga.offspring_creator import OperationSelector
    from ase.ga.population import Population, RankFitnessPopulation
    from ase.ga.convergence import GenerationRepetitionConvergence
    from ase.ga.utilities import closest_distances_generator, get_nnmat
    from ase.ga.data import DataConnection, PrepareDB
    from ase.io import read, write
    from ase.cluster import Icosahedron
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS
    from collections import defaultdict
    from random import randint
    from multiprocessing import Pool
    import numpy as np
    import time
    import os
    
    # Define population
    # Recommand to choose a number that is a multiple of the number of cpu
    pop_size = 50
       
    # Generate 50 icosahedral Ni110Pt37 nanoparticles with random orderings
    particle = Icosahedron('Ni', noshells=4)
    particle.center(vacuum=5.)
    rog = ROG(particle, elements=['Ni', 'Pt'],
              composition={'Ni': 0.75, 'Pt': 0.25},
              trajectory='starting_generation.traj')
    rog.run(num_gen=pop_size)
    
    # Generate random coverage patterns of 20 adsorbates on nanoparticles
    species = ['H', 'C', 'O', 'OH', 'CO', 'CH', 'CH2', 'CH3']
    images = read('starting_generation.traj', index=':')
    num_ads = 20 # Number of adsorbates, fix at this coverage throughout GA
    for _ in range(num_ads): # This will take quite some time
        rpg = RPG(images, adsorbate_species=['CO','OH','N'],
                  min_adsorbate_distance=3.,
                  composition_effect=True)
        rpg.run(num_gen=pop_size, action='add', unique=False)
        images = read('patterns.traj')
    patterns = read('patterns.traj', index=':')
    
    # Get the adsorption sites. Composition does not affect GA operations
    sas = ClusterAdsorptionSites(particle, composition_effect=False)
    
    # Instantiate the db
    db_name = 'ridge_Ni110Pt37_ads20.db'
    
    db = PrepareDB(db_name, cell=particle.cell, population_size=pop_size)
    
    for atoms in patterns:
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
    
    # Connect to the db
    db = DataConnection(db_name)
    
    # Define operators
    soclist = ([1, 1, 2, 1, 1, 2],
               [Rich2poorPermutation(elements=['Ni', 'Pt'], num_muts=5),
                Poor2richPermutation(elements=['Ni', 'Pt'], num_muts=5),
                RandomPermutation(elements=['Ni', 'Pt'], num_muts=5),
                MoveAdsorbate(species, adsorption_sites=sas, num_muts=5),
                ReplaceAdsorbate(species, adsorption_sites=sas, num_muts=5),
                SimpleCutSpliceCrossoverWithAdsorbates(species, keep_composition=True,
                                                       fix_coverage=True, 
                                                       adsorption_sites=sas),])
    op_selector = OperationSelector(*soclist)
    
    # Define comparators
    comp = SequentialComparator([AdsorptionSitesComparator(10),
                                 NNMatComparator(0.2, ['Ni', 'Pt'])],
                                [0.5, 0.5])
    
    # Normal fitness ranking irrespective of adsorbate coverage
    pop = Population(data_connection=db,
                     population_size=pop_size,
                     comparator=comp,
                     logfile='log.txt')
    
    # Set convergence criteria
    cc = GenerationRepetitionConvergence(pop, 5)
    
    # Calculate chemical potentials
    chem_pots = {'CH4': -24.039, 'H2O': -14.169, 'H2': -6.989}
    
    # Define the relax function
    def relax(atoms, single_point=False):
        atoms.center(vacuum=5.)
        atoms.calc = EMT()
        if not single_point:
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=0.1)
        time.sleep(1) # Add delay (only for testing)
    
        Epot = atoms.get_potential_energy()
        num_H = len([s for s in atoms.symbols if s == 'H'])
        num_C = len([s for s in atoms.symbols if s == 'C'])
        num_O = len([s for s in atoms.symbols if s == 'O'])
        mutot = num_C * chem_pots['CH4'] + num_O * chem_pots['H2O'] + (
                num_H - 4 * num_C - 2 * num_O) * chem_pots['H2'] / 2
        f = -(Epot - mutot)
    
        atoms.info['key_value_pairs']['raw_score'] = f
        atoms.info['key_value_pairs']['potential_energy'] = Epot

        # Parallelize nnmat calculations to accelerate NNMatComparator
        atoms.info['data']['nnmat'] = get_nnmat(atoms)
    
        return atoms
    
    # Relax starting generation
    def relax_an_unrelaxed_candidate(atoms):
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        nncomp = atoms.get_chemical_formula(mode='hill')
        print('Relaxing ' + nncomp)

        return relax(atoms, single_point=True) # Single point only for testing
    
    # Create a multiprocessing Pool
    pool = Pool(os.cpu_count())
    # Perform relaxations in parallel. Especially
    # useful when running GA on large nanoparticles
    relaxed_candidates = pool.map(relax_an_unrelaxed_candidate, 
                                  db.get_all_unrelaxed_candidates())
    pool.close()
    pool.join()
    db.add_more_relaxed_candidates(relaxed_candidates)
    pop.update()
    
    # Number of generations
    num_gens = 1000
    
    # Below is the iterative part of the algorithm
    gen_num = db.get_generation_number()
    for i in range(num_gens):
        # Check if converged
        if cc.converged():
            print('Converged')
            break
        print('Creating and evaluating generation {0}'.format(gen_num + i))
    
        def procreation(x):
            # Select an operator and use it
            op = op_selector.get_operator()
            while True:
                # Assign rng with a random seed
                np.random.seed(randint(1, 10000))
                pop.rng = np.random 
                # Select parents for a new candidate
                p1, p2 = pop.get_two_candidates()
                parents = [p1, p2]
                # Pure or bare nanoparticles are not considered
                if len(set(p1.numbers)) < 3:
                    continue
                offspring, desc = op.get_new_individual(parents)
                # An operator could return None if an offspring cannot be formed
                # by the chosen parents
                if offspring is not None:
                    break
            nncomp = offspring.get_chemical_formula(mode='hill')
            print('Relaxing ' + nncomp)
            if 'data' not in offspring.info:
                offspring.info['data'] = {'tag': None}

            return relax(offspring, single_point=True) # Single point only for testing
            
        # Create a multiprocessing Pool
        pool = Pool(os.cpu_count())
        # Perform procreations in parallel. Especially useful when
        # using adsorbate operators which requires site identification
        relaxed_candidates = pool.map(procreation, range(pop_size))
        pool.close()
        pool.join()
        db.add_more_relaxed_candidates(relaxed_candidates) 
    
        # Update the population to allow new candidates to enter
        pop.update()

Symmetry-constrained genetic algorithm for adlayer patterns                                                   
-----------------------------------------------------------

.. automodule:: acat.ga.group_operators
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: GroupSubstitute, GroupPermutation, Crossover, GroupCrossover

**Example1**

All the symmetry-constrained adsorbate operators can be easily used with other indexing-preserved operators, e.g. All operators can be used for both non-periodic nanoparticles and periodic surface slabs. To accelerate the GA, provide adsorsption sites, site pairing (or all possible parings), and use indexing-preserved operators implemented in ACAT.

As an example we will simultaneously optimize both the adsorbate overlayer pattern and the catalyst chemical ordering of a Ni48Pt16 fcc111 slab with adsorbate species of H, C, O, OH, CO, CH using the EMT calculator.

The script for a parallel symmetry-constrained genetic algorithm (SCGA) looks as follows:

.. code-block:: python

    from acat.settings import adsorbate_elements                                               
    from acat.adsorption_sites import SlabAdsorptionSites
    from acat.adsorbate_coverage import SlabAdsorbateCoverage
    from acat.build.ordering import RandomOrderingGenerator as ROG
    from acat.build.adlayer import OrderedPatternGenerator as OPG
    from acat.ga.adsorbate_operators import (ReplaceAdsorbateSpecies,
                                             CatalystAdsorbateCrossover)
    from acat.ga.slab_operators import (CutSpliceSlabCrossover,
                                        RandomSlabPermutation,
                                        RandomCompositionMutation)
    from acat.ga.group_operators import (AdsorbateGroupSubstitute,
                                         AdsorbateGroupPermutation)
    from acat.ga.adsorbate_comparators import AdsorptionSitesComparator  
    from ase.ga.standard_comparators import SequentialComparator, StringComparator
    from ase.ga.offspring_creator import OperationSelector
    from ase.ga.population import Population, RankFitnessPopulation
    from ase.ga.convergence import GenerationRepetitionConvergence
    from ase.ga.utilities import closest_distances_generator, get_nnmat
    from ase.ga.data import DataConnection, PrepareDB
    from ase.io import read, write
    from ase.build import fcc111
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS
    from collections import defaultdict
    from multiprocessing import Pool
    from random import randint
    import numpy as np
    import time
    import os
    
    # Define population
    # Recommend to choose a number that is a multiple of the number of cpu
    pop_size = 50
    
    # Generate 50 Ni48Pt16 slabs with random orderings
    slab = fcc111('Ni', (4, 4, 4), vacuum=5., orthogonal=True, periodic=True)
    slab_ids = list(range(len(slab)))
    rog = ROG(slab, elements=['Ni', 'Pt'],
              composition={'Ni': 0.75, 'Pt': 0.25},
              trajectory='starting_generation.traj')
    rog.run(num_gen=pop_size)
    
    # Get the adsorption sites. Composition does not affect GA operations
    sas = SlabAdsorptionSites(slab, surface='fcc111', ignore_bridge_sites=True, 
                              composition_effect=False)
    
    # Generate random coverage on each slab and save the groupings
    species = ['H', 'C', 'O', 'OH', 'CO', 'CH']
    images = read('starting_generation.traj', index=':')
    opg = OPG(images, adsorbate_species=species, surface='fcc111', 
              adsorption_sites=sas, max_species=2, allow_odd=True,
              remove_site_shells=1, save_groups=True, 
              trajectory='patterns.traj', append_trajectory=True)
    opg.run(max_gen=pop_size, unique=True)
    patterns = read('patterns.traj', index=':')
    
    # Instantiate the db
    db_name = 'ridge_Ni48Pt16_ads.db'
    
    db = PrepareDB(db_name, cell=slab.cell, population_size=pop_size)
    
    for atoms in patterns:
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
    
    # Connect to the db
    db = DataConnection(db_name)
    
    # Define operators
    soclist = ([3, 3, 2, 3, 3, 3],
               [RandomSlabPermutation(allowed_indices=slab_ids),
                RandomCompositionMutation(allowed_indices=slab_ids), 
                ReplaceAdsorbateSpecies(species, vacancy_replacement=True, 
                                        adsorption_sites=sas),
                AdsorbateGroupSubstitute(species, max_species=2, 
                                         adsorption_sites=sas, 
                                         remove_site_shells=1),
                AdsorbateGroupPermutation(species, adsorption_sites=sas, 
                                          remove_site_shells=1),
                CatalystAdsorbateCrossover(species, adsorption_sites=sas),])               
    op_selector = OperationSelector(*soclist)
    
    # Define comparators
    comp = SequentialComparator([StringComparator('potential_energy'),
                                 AdsorptionSitesComparator(10)],
                                [0.5, 0.5])
    
    def get_ads(atoms):
        """Returns a list of adsorbate names and corresponding indices."""
    
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        if 'adsorbates' in atoms.info['data']:
            adsorbates = atoms.info['data']['adsorbates']
        else:
            cac = SlabAdsorbateCoverage(atoms, adsorption_sites=sas)
            adsorbates = [t[0] for t in cac.get_adsorbates()]
    
        return adsorbates
    
    def vf(atoms):
        """Returns the descriptor that distinguishes candidates in the
        niched population."""
    
        return len(get_ads(atoms))
    
    # Give fittest candidates at different coverages equal fitness.
    # Use this to find global minimum at each adsorbate coverage
    pop = RankFitnessPopulation(data_connection=db,
                                population_size=pop_size,
                                comparator=comp,
                                variable_function=vf,
                                exp_function=True,
                                logfile='log.txt')
    
    # Normal fitness ranking irrespective of adsorbate coverage
    #pop = Population(data_connection=db,
    #                 population_size=pop_size,
    #                 comparator=comp,
    #                 logfile='log.txt')
    
    # Set convergence criteria
    cc = GenerationRepetitionConvergence(pop, 5)
    
    # Calculate chemical potentials
    chem_pots = {'CH4': -24.039, 'H2O': -14.169, 'H2': -6.989}
    
    # Define the relax function
    def relax(atoms, single_point=False):
        atoms.calc = EMT()
        if not single_point:
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=0.1)
        time.sleep(1) # Add delay (only for testing)
    
        Epot = atoms.get_potential_energy()
        num_H = len([s for s in atoms.symbols if s == 'H'])
        num_C = len([s for s in atoms.symbols if s == 'C'])
        num_O = len([s for s in atoms.symbols if s == 'O'])
        mutot = num_C * chem_pots['CH4'] + num_O * chem_pots['H2O'] + (
                num_H - 4 * num_C - 2 * num_O) * chem_pots['H2'] / 2
        f = -(Epot - mutot)
    
        atoms.info['key_value_pairs']['raw_score'] = f
        atoms.info['key_value_pairs']['potential_energy'] = Epot
    
        return atoms
    
    # Relax starting generation
    def relax_an_unrelaxed_candidate(atoms):
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        nncomp = atoms.get_chemical_formula(mode='hill')
        print('Relaxing ' + nncomp)
    
        return relax(atoms, single_point=True) # Single point only for testing
    
    # Create a multiprocessing Pool
    pool = Pool(25)#os.cpu_count())
    # Perform relaxations in parallel.
    relaxed_candidates = pool.map(relax_an_unrelaxed_candidate, 
                                  db.get_all_unrelaxed_candidates())
    pool.close()
    pool.join()
    db.add_more_relaxed_candidates(relaxed_candidates)
    pop.update()
    
    # Number of generations
    num_gens = 1000
    
    # Below is the iterative part of the algorithm
    gen_num = db.get_generation_number()
    for i in range(num_gens):
    
        # Check if converged
        if cc.converged():
            print('Converged')
            break
        print('Creating and evaluating generation {0}'.format(gen_num + i))
    
        def procreation(x):
            # Select an operator and use it
            op = op_selector.get_operator()
            while True:
                # Assign rng with a random seed
                np.random.seed(randint(1, 10000))
                pop.rng = np.random
                # Select parents for a new candidate
                p1, p2 = pop.get_two_candidates()
                parents = [p1, p2]
                # Pure or bare slabs are not considered
                if len(set(p1.numbers)) < 3:
                    continue
                offspring, desc = op.get_new_individual(parents)
                # An operator could return None if an offspring cannot be formed
                # by the chosen parents
                if offspring is not None:
                    break
            nncomp = offspring.get_chemical_formula(mode='hill')
            print('Relaxing ' + nncomp)
            if 'data' not in offspring.info:
                offspring.info['data'] = {'tag': None}
    
            return relax(offspring, single_point=True) # Single point only for testing
    
        # Create a multiprocessing Pool
        pool = Pool(25)#os.cpu_count())
        # Perform procreations in parallel. Especially useful when
        # using adsorbate operators which requires site identification
        relaxed_candidates = pool.map(procreation, range(pop_size))
        pool.close()
        pool.join()
        db.add_more_relaxed_candidates(relaxed_candidates)
    
        # Update the population to allow new candidates to enter
        pop.update() 

Evolutionary Multitasking
-------------------------

.. automodule:: acat.ga.multitasking
    :members:
    :undoc-members:
    :show-inheritance:

**Example1**

We implement evolutionary multitasking in a Population instance and a Convergence instance in ACAT. The maximum-gain dynamic niching (MGDN) algrotim is implemented in the ``MultitaskPopulation``. 

As an example we will simultaneously optimize both the adsorbate overlayer pattern and the catalyst chemical ordering of a Ni48Pt16 fcc111 slab with adsorbate species of H, C, O, OH, CO, CH using the EMT calculator for 10 tasks: each with CH4 chemical potentials of 20, 21, 22, 23, 24, 25, 26, 27, 28 and 29 eV, respectively.

The script for a parallel multitasking symmetry-constrained genetic algorithm (SCGA) looks as follows:

.. code-block:: python

    from acat.settings import adsorbate_elements                                             
    from acat.adsorption_sites import SlabAdsorptionSites
    from acat.adsorbate_coverage import SlabAdsorbateCoverage
    from acat.build.ordering import RandomOrderingGenerator as ROG
    from acat.build.adlayer import OrderedPatternGenerator as OPG
    from acat.ga.adsorbate_operators import (ReplaceAdsorbateSpecies,
                                             CatalystAdsorbateCrossover)
    from acat.ga.slab_operators import (CutSpliceSlabCrossover,
                                        RandomSlabPermutation,
                                        RandomCompositionMutation)
    from acat.ga.group_operators import (AdsorbateGroupSubstitute,
                                         AdsorbateGroupPermutation)
    from acat.ga.adsorbate_comparators import AdsorptionSitesComparator  
    from acat.ga.multitasking import (MultitaskPopulation,
                                      MultitaskGenerationRepetitionConvergence)
    from ase.ga.standard_comparators import SequentialComparator, StringComparator
    from ase.ga.offspring_creator import OperationSelector
    from ase.ga.utilities import closest_distances_generator, get_nnmat
    from ase.ga.data import DataConnection, PrepareDB
    from ase.io import read, write
    from ase.build import fcc111
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS
    from collections import defaultdict
    from multiprocessing import Pool
    from random import randint
    import numpy as np
    import time
    import os
    
    # Define population
    # Recommend to choose a number that is a multiple of the number of cpu
    pop_size = 50
    
    # Define the tasks. In this case we use 10 different chemical potentials of CH4
    tasks = np.arange(20., 30., 1.)
    
    # Generate 50 Ni48Pt16 slabs with random orderings
    slab = fcc111('Ni', (4, 4, 4), vacuum=5., orthogonal=True, periodic=True)
    slab_ids = list(range(len(slab)))
    rog = ROG(slab, elements=['Ni', 'Pt'],
              composition={'Ni': 0.75, 'Pt': 0.25},
              trajectory='starting_generation.traj')
    rog.run(num_gen=pop_size)
    
    # Get the adsorption sites. Composition does not affect GA operations
    sas = SlabAdsorptionSites(slab, surface='fcc111', ignore_bridge_sites=True, 
                              composition_effect=False)
    
    # Generate random coverage on each slab and save the groupings
    species = ['H', 'C', 'O', 'OH', 'CO', 'CH']
    images = read('starting_generation.traj', index=':')
    opg = OPG(images, adsorbate_species=species, surface='fcc111', 
              adsorption_sites=sas, max_species=2, allow_odd=True,
              remove_site_shells=1, save_groups=True, 
              trajectory='patterns.traj', append_trajectory=True)
    opg.run(max_gen=pop_size, unique=True)
    patterns = read('patterns.traj', index=':')
    
    # Instantiate the db
    db_name = 'ridge_Ni48Pt16_ads_multitask.db'
    
    db = PrepareDB(db_name, cell=slab.cell, population_size=pop_size)
    
    for atoms in patterns:
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
    
    # Connect to the db
    db = DataConnection(db_name)
    
    # Define operators
    soclist = ([3, 3, 2, 3, 3, 3],
               [RandomSlabPermutation(allowed_indices=slab_ids),
                RandomCompositionMutation(allowed_indices=slab_ids), 
                ReplaceAdsorbateSpecies(species, vacancy_replacement=True, 
                                        adsorption_sites=sas),
                AdsorbateGroupSubstitute(species, max_species=2, 
                                         adsorption_sites=sas, 
                                         remove_site_shells=1),
                AdsorbateGroupPermutation(species, adsorption_sites=sas, 
                                          remove_site_shells=1),
                CatalystAdsorbateCrossover(species, adsorption_sites=sas),])               
    op_selector = OperationSelector(*soclist)
    
    # Define comparators
    comp = SequentialComparator([StringComparator('potential_energy'),
                                 AdsorptionSitesComparator(10)],
                                [0.5, 0.5])
    
    # Initialize the population and specify the number of tasks
    pop = Multitaskpopulation(data_connection=db,
                              population_size=pop_size,
                              num_tasks=len(tasks),
                              comparator=comp,
                              exp_function=True,
                              logfile='log.txt')
    
    # Set convergence criteria
    cc = MultitaskGenerationRepetitionConvergence(pop, 5)
    
    # Calculate chemical potentials
    chem_pots = {'CH4': tasks, 'H2O': -14.169, 'H2': -6.989}
    
    # Define the relax function
    def relax(atoms, single_point=False):
        atoms.calc = EMT()
        if not single_point:
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=0.1)
        time.sleep(1) # Add delay (only for testing)
    
        Epot = atoms.get_potential_energy()
        num_H = len([s for s in atoms.symbols if s == 'H'])
        num_C = len([s for s in atoms.symbols if s == 'C'])
        num_O = len([s for s in atoms.symbols if s == 'O'])
        mutots = num_C * chem_pots['CH4'] + num_O * chem_pots['H2O'] + (
                 num_H - 4 * num_C - 2 * num_O) * chem_pots['H2'] / 2
        fs = -(Epot - mutots)

        # Save the raw fitness of the configuration for all tasks
        # as a Numpy array in atoms.info['data']['raw_scores']
        atoms.info['data']['raw_scores'] = fs
        atoms.info['key_value_pairs']['potential_energy'] = Epot
    
        return atoms
    
    # Relax starting generation
    def relax_an_unrelaxed_candidate(atoms):
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        nncomp = atoms.get_chemical_formula(mode='hill')
        print('Relaxing ' + nncomp)
    
        return relax(atoms, single_point=True) # Single point only for testing
    
    # Create a multiprocessing Pool
    pool = Pool(os.cpu_count())
    # Perform relaxations in parallel.
    relaxed_candidates = pool.map(relax_an_unrelaxed_candidate, 
                                  db.get_all_unrelaxed_candidates())
    pool.close()
    pool.join()
    # Update the population with the newly relaxed candidates
    # (DO NOT add relaxed_candidates into db before this update)
    pop.update(new_cand=relaxed_candidates)
    
    # Number of generations
    num_gens = 1000
    
    # Below is the iterative part of the algorithm
    gen_num = db.get_generation_number()
    for i in range(num_gens):
    
        # Check if converged
        if cc.converged():
            print('Converged')
            break
        print('Creating and evaluating generation {0}'.format(gen_num + i))
    
        def procreation(x):
            # Select an operator and use it
            op = op_selector.get_operator()
            while True:
                # Assign rng with a random seed
                np.random.seed(randint(1, 10000))
                pop.rng = np.random
                # Select parents for a new candidate
                p1, p2 = pop.get_two_candidates()
                parents = [p1, p2]
                # Pure or bare slabs are not considered
                if len(set(p1.numbers)) < 3:
                    continue
                offspring, desc = op.get_new_individual(parents)
                # An operator could return None if an offspring cannot be formed
                # by the chosen parents
                if offspring is not None:
                    break
            nncomp = offspring.get_chemical_formula(mode='hill')
            print('Relaxing ' + nncomp)
            if 'data' not in offspring.info:
                offspring.info['data'] = {'tag': None}
    
            return relax(offspring, single_point=True) # Single point only for testing
    
        # Create a multiprocessing Pool
        pool = Pool(os.cpu_count())
        # Perform procreations in parallel. Especially useful when
        # using adsorbate operators which requires site identification
        relaxed_candidates = pool.map(procreation, range(pop_size))
        pool.close()
        pool.join()
    
        # Update the population to allow new candidates to enter
        # (DO NOT add relaxed_candidates into db before this update)
        pop.update(new_cand=relaxed_candidates) 

The fittest individuals covering all tasks (and which task is dominated by which individual) can be easily obtained by the following script:

.. code-block:: python

    from ase.db import connect
    from ase.io import write

    db = connect('ridge_Ni48Pt16_ads_multitask.db')
    fittest_images = []
    seen_dns = set()
    for row in db.select('relaxed=1'):
        atoms = row.toatoms()
        f_eff = row.raw_score
        dn = row.dominating_niche
        niches = row.data['niches']
        # Get the fittest individual with an effective
        # fitness of 0 in each niche (without duplicates)
        if (f_eff == 0) and (dn not in seen_dns):
            seen_dns.add(dn)
            fittest_images.append(atoms)
        # Get the niches where this structure is the fittest
        print('Fittest niches: {}'.format(niches))
    write('fittest_images.traj', fittest_images)
