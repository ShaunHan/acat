Genetic algorithm
=================

Adsorbate procreation operators
-------------------------------

.. automodule:: acat.ga.adsorbate_operators
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: get_all_adsorbate_indices, get_numbers, get_atoms_without_adsorbates

Usage
-----
All the adsorbate operators can be easily used with other ASE operators. ``AddAdsorbate``, ``RemoveAdsorbate``, ``MoveAdsorbate`` and ``ReplaceAdsorbate`` operators can be used for both non-periodic nanoparticles and periodic surface slabs. ``CutSpliceCrossoverWithAdsorbates`` operator only works for nanoparticles, and it is not recommonded as it is not stable yet.

As an example we will find the stable structures of a Ni110Pt37 icosahedral nanoparticle with adsorbate species of H, C, O, OH, CO, CH, CH2 and CH3 using the EMT calculator.

The script for the genetic algorithm looks as follows:

.. code-block:: python

    from acat.settings import adsorbate_elements                                                       
    from acat.adsorbate_coverage import ClusterAdsorbateCoverage
    from acat.build.orderings import RandomOrderingGenerator as ROG
    from acat.build.patterns import random_coverage_pattern
    from acat.ga.adsorbate_operators import AddAdsorbate, RemoveAdsorbate 
    from acat.ga.adsorbate_operators import MoveAdsorbate, ReplaceAdsorbate 
    from acat.ga.adsorbate_operators import CutSpliceCrossoverWithAdsorbates
    from ase.ga.particle_mutations import RandomPermutation, COM2surfPermutation
    from ase.ga.particle_mutations import Rich2poorPermutation, Poor2richPermutation
    from ase.ga.particle_comparator import NNMatComparator
    from ase.ga.standard_comparators import SequentialComparator
    from ase.ga.adsorbate_comparators import AdsorptionSitesComparator
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
    
    # Generate 50 icosahedral Ni110Pt37 nanoparticles with random orderings
    pop_size = 50
    particle = Icosahedron('Ni', noshells=4)
    particle.center(vacuum=5.)
    rog = ROG(particle, species=['Ni', 'Pt'], 
              composition={'Ni': 0.75, 'Pt': 0.25})
    rog.run(n_gen=pop_size)
    
    # Generate random coverage on each nanoparticle
    species=['H', 'C', 'O', 'OH', 'CO', 'CH', 'CH2', 'CH3']
    images = read('orderings.traj', index=':')
    patterns = []
    for atoms in images:
        dmin = uniform(3.5, 8.5)
        pattern = random_coverage_pattern(atoms, adsorbate_species=species,
                                          min_adsorbate_distance=dmin)
        patterns.append(pattern)
    
    # Instantiate the db
    db_name = 'emt_ridge_Ni110Pt37_ads.db'
    
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
    
    comp = SequentialComparator([AdsorptionSitesComparator(10),
                                 NNMatComparator(0.2,['Ni', 'Pt'])], 
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
    
    # Define population
    pop_size = db.get_param('population_size')
    
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
    while db.get_number_of_unrelaxed_candidates() > 0:
        atoms = db.get_an_unrelaxed_candidate()    
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
        nncomp = atoms.get_chemical_formula(mode='hill')
        print('Relaxing ' + nncomp)
        relax(atoms, single_point=True) # Single point for simplification
        db.add_relaxed_step(atoms)
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
        new_generation = []
        for _ in range(pop_size):
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
            new_generation.append(offspring)
    
        # We add a full relaxed generation at once, this is faster than adding
        # one at a time
        db.add_more_relaxed_candidates(new_generation)
    
        # update the population to allow new candidates to enter
        pop.update()
