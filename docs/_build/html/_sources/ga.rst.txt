Genetic algorithm
=================

Optimize adsorbate overlayer pattern
------------------------------------

.. automodule:: acat.ga.adsorbate_operators
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: get_all_adsorbate_indices, get_numbers, get_atoms_without_adsorbates

.. automodule:: acat.ga.adsorbate_comparators
    :members:
    :undoc-members:
    :show-inheritance:

**Example**

All the adsorbate operators can be easily used with other ASE operators. ``AddAdsorbate``, ``RemoveAdsorbate``, ``MoveAdsorbate`` and ``ReplaceAdsorbate`` operators can be used for both non-periodic nanoparticles and periodic surface slabs. ``CutSpliceCrossoverWithAdsorbates`` operator only works for nanoparticles, and it is not recommonded as it is not stable yet.

As an example we will simultaneously optimize both the adsorbate overlayer pattern and the catalyst chemical ordering of a Ni110Pt37 icosahedral nanoalloy with adsorbate species of H, C, O, OH, CO, CH, CH2 and CH3 using the EMT calculator.

The script for a parallel genetic algorithm looks as follows:

.. code-block:: python
   
    from acat.settings import adsorbate_elements                                                         
    from acat.adsorbate_coverage import ClusterAdsorbateCoverage
    from acat.build.ordering import RandomOrderingGenerator as ROG
    from acat.build.overlayer import random_coverage_pattern
    from acat.ga.adsorbate_operators import AddAdsorbate, RemoveAdsorbate 
    from acat.ga.adsorbate_operators import MoveAdsorbate, ReplaceAdsorbate 
    from acat.ga.adsorbate_operators import CutSpliceCrossoverWithAdsorbates
    from ase.ga.particle_mutations import RandomPermutation, COM2surfPermutation
    from ase.ga.particle_mutations import Rich2poorPermutation, Poor2richPermutation
    from acat.ga.adsorbate_comparators import AdsorptionSitesComparator
    from ase.ga.particle_comparator import NNMatComparator
    from ase.ga.standard_comparators import SequentialComparator
    from ase.ga.offspring_creator import OperationSelector
    from ase.ga.population import Population, RankFitnessPopulation
    from ase.ga.convergence import GenerationRepetitionConvergence
    from ase.ga.utilities import closest_distances_generator
    from ase.ga.data import DataConnection, PrepareDB
    from ase.io import read, write
    from ase.cluster import Icosahedron
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS
    from collections import defaultdict
    from random import choices, uniform
    from multiprocessing import Pool
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
    rog.run(n_gen=pop_size)
    
    # Generate random coverage on each nanoparticle
    species=['H', 'C', 'O', 'OH', 'CO', 'CH', 'CH2', 'CH3']
    images = read('starting_generation.traj', index=':')
    patterns = []
    for atoms in images:
        dmin = uniform(3.5, 8.5)
        pattern = random_coverage_pattern(atoms, adsorbate_species=species,
                                          min_adsorbate_distance=dmin)
        patterns.append(pattern)
    
    # Instantiate the db
    db_name = 'ridge_Ni110Pt37_ads.db'
    
    db = PrepareDB(db_name, cell=particle.cell, population_size=pop_size)
    
    for atoms in patterns:
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
        db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
    
    # Connect to the db
    db = DataConnection(db_name)
    
    # Define operators
    soclist = ([1, 1, 2, 1, 1, 1, 1], 
               [Rich2poorPermutation(elements=['Ni', 'Pt'], num_muts=5),
                Poor2richPermutation(elements=['Ni', 'Pt'], num_muts=5),                              
                RandomPermutation(num_muts=5),
                AddAdsorbate(species, num_muts=5),
                RemoveAdsorbate(species, num_muts=5),
                MoveAdsorbate(species, num_muts=5),
                ReplaceAdsorbate(species, num_muts=5),])
    
    op_selector = OperationSelector(*soclist)

    # Define comparators    
    comp = SequentialComparator([AdsorptionSitesComparator(10),
                                 NNMatComparator(0.2, ['Ni', 'Pt'])], 
                                [0.5, 0.5])
    
    def get_ads(atoms):
        """Returns a list of adsorbate names and corresponding indices."""
    
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
        if 'adsorbates' in atoms.info['data']:
            adsorbates = atoms.info['data']['adsorbates']
        else:
            cac = ClusterAdsorbateCoverage(atoms)
            adsorbates = cac.get_adsorbates()
    
        return adsorbates
    
    def vf(atoms):
        """Returns the descriptor that distinguishes candidates in the 
        niched population."""
    
        return len(get_ads(atoms))
    
    # Give fittest candidates at different coverages equal fitness
    pop = RankFitnessPopulation(data_connection=db,
                                population_size=pop_size,
                                comparator=comp,
                                variable_function=vf,
                                exp_function=True,
                                logfile='log.txt')
    
    # Normal fitness ranking regardless of coverage
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
    
        Epot = atoms.get_potential_energy()
        num_H = len([s for s in atoms.symbols if s == 'H'])
        num_C = len([s for s in atoms.symbols if s == 'C'])
        num_O = len([s for s in atoms.symbols if s == 'O'])
        mutot = num_C * chem_pots['CH4'] + num_O * chem_pots['H2O'] + (
                num_H - 4 * num_C - 2 * num_O) * chem_pots['H2'] / 2
        f = Epot - mutot
    
        atoms.info['key_value_pairs']['raw_score'] = -f
        atoms.info['key_value_pairs']['potential_energy'] = Epot
    
        return atoms
    
    # Relax starting generation
    def relax_an_unrelaxed_candidate(atoms):
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
        nncomp = atoms.get_chemical_formula(mode='hill')
        print('Relaxing ' + nncomp)
        relax(atoms, single_point=True) # Single point for simplification
        db.add_relaxed_step(atoms)
    
    # Create a multiprocessing Pool
    pool = Pool(os.cpu_count())
    # Perform relaxations in parallel. Especially
    # useful when running GA on large nanoparticles  
    pool.map(relax_an_unrelaxed_candidate, db.get_all_unrelaxed_candidates())  
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
            # Select parents for a new candidate
            p1, p2 = pop.get_two_candidates()
            parents = [p1, p2]
            # Pure or bare nanoparticles are not considered
            if len(set(p1.numbers)) < 3:
                continue 
            offspring, desc = op.get_new_individual(parents)
            # An operator could return None if an offspring cannot be formed
            # by the chosen parents
            if offspring is None:
                continue
            nncomp = offspring.get_chemical_formula(mode='hill')
            print('Relaxing ' + nncomp)        
            if 'data' not in offspring.info:
                offspring.info['data'] = {}
            relax(offspring, single_point=True) # Single point for simplification
            db.add_relaxed_candidate(offspring)
    
        # Create a multiprocessing Pool
        pool = Pool(os.cpu_count())
        # Perform procreations in parallel. Especially useful when
        # using adsorbate operators which requires site identification
        pool.map(procreation, range(pop_size))  
    
        # update the population to allow new candidates to enter
        pop.update()

Symmetry-constrained genetic algorithm for nanoalloys
-----------------------------------------------------

.. automodule:: acat.ga.symmetric_particle_operators
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: acat.ga.symmetric_particle_comparators
    :members:
    :undoc-members:
    :show-inheritance:

**Example**

All the symmetric particle operators can be easily used with other ASE operators.

As an example we will find the convex hull of ternary NixPtyAu405-x-y truncated octahedral nanoalloys using the ASAP EMT calculator. **Note that we must first align the symmetry axis of interest to the z direction.** Here we want to study the 3-fold mirror circular symmetry around the C3 axis of the particle.

The script for a parallel symmetry-constrained genetic algorithm (SCGA) looks as follows:

.. code-block:: python

    from acat.build.ordering import SymmetricOrderingGenerator as SOG
    from acat.ga.symmetric_particle_operators import SymmetricSubstitute
    from acat.ga.symmetric_particle_operators import SymmetricPermutation
    from acat.ga.symmetric_particle_operators import SymmetricCrossover
    from acat.ga.symmetric_particle_comparators import ShellSizeComparator
    from acat.ga.symmetric_particle_comparators import ShellCompositionComparator
    from ase.ga.particle_comparator import NNMatComparator
    from ase.ga.standard_comparators import SequentialComparator
    from ase.ga.offspring_creator import OperationSelector
    from ase.ga.population import Population, RankFitnessPopulation
    from ase.ga.convergence import GenerationRepetitionConvergence
    from ase.ga.data import DataConnection, PrepareDB
    from ase.io import read, write
    from ase.cluster import Octahedron
    from ase.optimize import BFGS
    from asap3 import EMT as asapEMT
    from multiprocessing import Pool
    import os
    
    # Define population. 
    # Recommand to choose a number that is a multiple of the number of cpu
    pop_size = 100
    
    # Build and rotate the particle so that C3 axis is aligned to z direction
    particle = Octahedron('Ni', length=9, cutoff=3)
    particle.rotate(45, 'x')     
    particle.rotate(-35.29, 'y') 
    particle.center(vacuum=5.)                                              

    # Generate 100 truncated ocatahedral NixPtyAu405-x-y nanoalloys with
    # mirror circular symmetry. Get the shells at the same time.
    sog = SOG(particle, elements=['Ni', 'Pt', 'Au'],
              symmetry='mirror_circular',
              trajectory='starting_generation.traj')
    sog.run(max_gen=pop_size, mode='stochastic', verbose=True)
    shells = sog.get_shells()
    images = read('starting_generation.traj', index=':')
    
    # Instantiate the db
    db_name = 'ridge_mirror_circular_NiPtAu_TO405_C3.db'
    
    db = PrepareDB(db_name, cell=particle.cell, population_size=pop_size)
    
    for atoms in images:
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
        db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
    
    # Connect to the db
    db = DataConnection(db_name)
    
    # Define operators
    soclist = ([2, 2, 3], 
               [SymmetricSubstitute(shells, elements=['Ni', 'Pt', 'Au'], num_muts=3),
                SymmetricPermutation(shells, elements=['Ni', 'Pt', 'Au'], num_muts=1),                              
                SymmetricCrossover(shells, elements=['Ni', 'Pt', 'Au']),])
    
    op_selector = OperationSelector(*soclist)
    
    # Define comparators
    comp = SequentialComparator([ShellSizeComparator(shells, ['Ni', 'Pt', 'Au']),
                                 NNMatComparator(0.2, ['Ni', 'Pt', 'Au'])],
                                [0.5, 0.5])
    
    def vf(atoms):
        """Returns the descriptor that distinguishes candidates in the 
        niched population."""
        return atoms.get_chemical_formula(mode='hill')
    
    # Give fittest candidates at different compositions equal fitness.
    # Use this to find global minima at each composition
    pop = RankFitnessPopulation(data_connection=db,
                                population_size=pop_size,
                                comparator=comp,
                                variable_function=vf,
                                exp_function=True,
                                logfile='log.txt')
    
    # Normal fitness ranking irrespective of coverage.
    # Use this to find global minimum irrespective of composition
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
    
        return atoms

    # Relax starting generation
    def relax_an_unrelaxed_candidate(atoms):
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
        nncomp = atoms.get_chemical_formula(mode='hill')
        print('Relaxing ' + nncomp)
        relax(atoms)
        db.add_relaxed_step(atoms)
    
    # Create a multiprocessing Pool
    pool = Pool(os.cpu_count())
    # Perform relaxations in parallel. Especially
    # useful when running GA on large nanoparticles  
    pool.map(relax_an_unrelaxed_candidate, db.get_all_unrelaxed_candidates())  
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
            # Select parents for a new candidate
            p1, p2 = pop.get_two_candidates()
            # Pure and binary candidates are not considered
            if len(set(p1.numbers)) < 3:
                return 
            parents = [p1, p2]
            offspring, desc = op.get_new_individual(parents)
            # An operator could return None if an offspring cannot be formed
            # by the chosen parents
            if offspring is None:
                return
            nncomp = offspring.get_chemical_formula(mode='hill')
            print('Relaxing ' + nncomp)        
            if 'data' not in offspring.info:
                offspring.info['data'] = {}
            relax(offspring)
            db.add_relaxed_candidate(offspring)
    
        # Create a multiprocessing Pool
        pool = Pool(os.cpu_count())
        # Perform procreations in parallel. Especially
        # useful when running GA on large nanoparticles  
        pool.map(procreation, range(pop_size))  
    
        # update the population to allow new candidates to enter
        pop.update()
