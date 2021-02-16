from ase.geometry import get_distances
from ase.io import Trajectory, read, write
from ase import Atoms
from ase.ga.utilities import get_nnmat
from itertools import product
import numpy as np
import random
import math


class SymmetricOrderingGenerator(object):
    """
    As for now, only support clusters.
    Please align the z-direction to the symmetry axis of the cluster.

    cutoff: Minimum distance (A) that the code can recognize between 
         two neighbor layers. If the structure is irregular, use
         a higher cutoff.
    composition: e.g. {'Ni': 0.75, 'Pt': 0.25}
        All compositions if not specified
    """

    def __init__(self, atoms, species,
                 symmetry='central', #'horizontal', 'vertical'
                 cutoff=.1,                 
                 composition=None,
                 trajectory='orderings.traj',
                 append_trajectory=False):

        assert len(species) == 2
        self.atoms = atoms
        self.species = species
        self.ma, self.mb = species[0], species[1]
        self.symmetry = symmetry
        self.cutoff = cutoff
        self.composition = composition
        if self.composition is not None:
            assert set(self.composition.keys()) == set(self.species)
            ca = self.composition[self.ma] / sum(self.composition.values())
            self.nma = int(round(len(self.atoms) * ca))
            self.nmb = len(self.atoms) - self.nma

        if isinstance(trajectory, str):
            self.trajectory = trajectory                        
        self.append_trajectory = append_trajectory

        self.layers = self.get_layers()

    def get_nblist_from_center_atom(self):
        atoms = self.atoms.copy()
        atoms.center()
        geo_mid = [(atoms.cell/2.)[0][0], (atoms.cell/2.)[1][1], 
                   (atoms.cell/2.)[2][2]]
        if self.symmetry == 'central':
            dists = get_distances(atoms.positions, [geo_mid])[1]
        elif self.symmetry == 'horizontal':
            dists = atoms.positions[:, 2] - geo_mid[2]
        elif self.symmetry == 'vertical':
            dists = np.asarray([math.sqrt((a.position[0] - geo_mid[0])**2 + 
                               (a.position[1] - geo_mid[1])**2) for a in atoms])
        sorted_indices = np.argsort(np.ravel(dists))
        return sorted_indices, dists[sorted_indices]    
    
    def get_layers(self):
        indices, dists = self.get_nblist_from_center_atom() 
        layers = []
        old_dist = -10.0
        for i, dist in zip(indices, dists):
            if abs(dist-old_dist) > self.cutoff:
                layers.append([i])
            else:
                layers[-1].append(i)
            old_dist = dist
    
        return layers

    def run(self, max_gen=None, mode='systematic', verbose=True):
        traj_mode = 'a' if self.append_trajectory else 'w'
        traj = Trajectory(self.trajectory, mode=traj_mode)
        atoms = self.atoms
        layers = self.layers
        nlayers = len(layers)
        if verbose:
            print('{} layers classified'.format(len(layers)))
        n_write = 0

        # When the number of layers is too large (> 20), systematic enumeration 
        # is not feasible. Stochastic sampling is the only option
        if mode == 'systematic':
            if nlayers > 20:
                if verbose:
                    print('{} layers is too large for systematic'.format(nlayers), 
                          'generator. Use stochastic generator instead')
                mode = 'stochastic'
            else:    
                combs = list(product(self.species, repeat=len(layers)))
                random.shuffle(combs)
                for comb in combs:
                    for j, specie in enumerate(comb):
                        atoms.symbols[layers[j]] = specie
                    if self.composition is not None:
                        nnma = len([a for a in atoms if a.symbol == self.ma])
                        if nnma != self.nma:
                            continue
                    traj.write(atoms)
                    n_write += 1
                    if max_gen is not None:
                        if n_write == max_gen:
                            break

        if mode == 'stochastic':                                              
            combs = set()
            going = True
            while going:
                comb = tuple(np.random.choice(self.species, size=nlayers))
                if comb not in combs: 
                    combs.add(comb)
                    for j, specie in enumerate(comb):
                        atoms.symbols[layers[j]] = specie
                    if self.composition is not None:
                        nnma = len([a for a in atoms if a.symbol == self.ma])
                        if nnma != self.nma:
                            continue
                    traj.write(atoms)
                    n_write += 1
                    if max_gen is not None:
                        if n_write == max_gen:
                            going = False
        if verbose:
            print('{} symmetric chemical orderings generated'.format(n_write))


class RandomOrderingGenerator(object):

    def __init__(self, atoms, species,
                 composition=None,
                 trajectory='orderings.traj',
                 append_trajectory=False):

        assert len(species) == 2
        self.atoms = atoms
        self.species = species
        self.ma, self.mb = species[0], species[1]
        self.composition = composition
        if self.composition is not None:
            assert set(self.composition.keys()) == set(self.species)
            ca = self.composition[self.ma] / sum(self.composition.values())
            self.nma = int(round(len(self.atoms) * ca))
            self.nmb = len(self.atoms) - self.nma

        if isinstance(trajectory, str):
            self.trajectory = trajectory                        
        self.append_trajectory = append_trajectory

    def run(self, n_gen):
        traj_mode = 'a' if self.append_trajectory else 'w'
        traj = Trajectory(self.trajectory, mode=traj_mode)
        atoms = self.atoms
        natoms = len(atoms)
        for a in atoms:
            a.symbol = self.ma

        for _ in range(n_gen):
            indi = atoms.copy()
            if self.composition is not None:
                nmb = self.nmb
            else:
                nmb = random.randint(1, natoms-1)
            to_mb = random.sample(range(natoms), nmb)
            indi.symbols[to_mb] = self.mb
            traj.write(indi)
