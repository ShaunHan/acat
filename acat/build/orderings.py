from ase.geometry import get_distances
from ase.io import Trajectory, read, write
from asap3 import EMT as asapEMT
from itertools import product
import numpy as np
import random
import math


class SymmetricOrderingGenerator(object):
    """`SymmetricOrderingGenerator` is a class for generating 
    symmetric chemical orderings for a bimetallic catalyst.
    As for now, only support clusters. Please align the z direction 
    to the symmetry axis of the cluster.
 
    Parameters
    ----------
    atoms : ase.Atoms object
        The nanoparticle to use as a template to generate symmetric
        chemical orderings. Accept any ase.Atoms object. No need to be 
        built-in.

    species : list of strs 
        The two metal species of the bimetallic catalyst.

    symmetry : str, default 'spherical'
        Support 4 symmetries: 
        'spherical' = centrosymmetry (layers defined by the distances 
        to the geometric center);
        'planar' = planar symmetry around z axis (layers defined by 
        the z coordinates);
        'cylindrical' = cylindrical symmetry around z axis (layers 
        defined by the distances to the z axis);
        'chemical' = symmetry w.r.t chemical environment (layers 
        defined by the atomic energy calculated by EMT).

    cutoff: float, default 0.1
        Minimum distance (in Angstrom) or energy difference (in eV) 
        that the code can recognize between two neighbor layers. 
        If the structure is irregular, use a higher cutoff.

    secondary_symmetry : str, default None
        Add a secondary symmetry check to define layers hierarchically. 
        For example, even if two atoms are classifed in one layer that 
        defined by the primary symmetry, they can still end up in 
        different layers if they fall into two different layers that 
        defined by the secondary symmetry. Support same 4 symmetries. 
        Note that secondary symmetry has the same importance as the 
        primary symmetry, so you can set either of the two symmetries 
        of interest as the secondary symmetry.

    secondary_cutoff : float, default 0.1
        Same as cutoff, except that it is for the secondary symmetry.

    composition : dict, default None
        Generate symmetric orderings only at a certain composition.
        The dictionary contains the two speices as keys and their 
        concentrations as values. Generate orderings at all 
        compositions if not specified.

    layer_threshold : int, default 20
        Number of layers to switch to stochastic mode automatically.

    trajectory : str, default 'orderings.traj'
        The name of the output ase trajectory file.

    append_trajectory : bool, default False
        Whether to append structures to the existing trajectory. 

    """

    def __init__(self, atoms, species,
                 symmetry='spherical', #'planar', 'cylindrical', 'chemical'
                 cutoff=.1,       
                 secondary_symmetry=None,
                 secondary_cutoff=.1,
                 composition=None,
                 layer_threshold=20,
                 trajectory='orderings.traj',
                 append_trajectory=False):

        assert len(species) == 2
        self.atoms = atoms
        self.species = species
        self.ma, self.mb = species[0], species[1]
        self.symmetry = symmetry
        self.cutoff = cutoff
        self.secondary_symmetry = secondary_symmetry
        self.secondary_cutoff = secondary_cutoff

        self.composition = composition
        if self.composition is not None:
            assert set(self.composition.keys()) == set(self.species)
            ca = self.composition[self.ma] / sum(self.composition.values())
            self.nma = int(round(len(self.atoms) * ca))
            self.nmb = len(self.atoms) - self.nma

        self.layer_threshold = layer_threshold
        if isinstance(trajectory, str):
            self.trajectory = trajectory                        
        self.append_trajectory = append_trajectory

        self.layers = self.get_layers()

    def get_nblist_from_center_atom(self, symmetry):
        """Returns the indices sorted by the distance to the center atom,
        together with their distances, given a specific symmetry.

        Parameters
        ----------
        symmetry : str
            Support the same 4 symmetries.

        """

        atoms = self.atoms.copy()
        atoms.center()
        geo_mid = [(atoms.cell/2.)[0][0], (atoms.cell/2.)[1][1], 
                   (atoms.cell/2.)[2][2]]
        if symmetry == 'spherical':
            dists = get_distances(atoms.positions, [geo_mid])[1]
        elif symmetry == 'planar':
            dists = atoms.positions[:, 2]
        elif symmetry == 'cylindrical':
            dists = np.asarray([math.sqrt((a.position[0] - geo_mid[0])**2 + 
                               (a.position[1] - geo_mid[1])**2) for a in atoms])
        elif symmetry == 'chemical':
            calc = asapEMT()
            ref_atoms = self.atoms.copy()
            ref_atoms.center(vacuum=5.)
            for a in atoms:
                a.symbol = 'Pt'
            ref_atoms.calc = calc
            dists = ref_atoms.get_potential_energies()

        sorted_indices = np.argsort(np.ravel(dists))
        return sorted_indices, dists[sorted_indices]    
    
    def get_layers(self):
        """Get the layers (a list of lists of atom indices) that divided by 
        the symmetry."""

        indices, dists = self.get_nblist_from_center_atom(symmetry=self.symmetry) 
        layers = []
        old_dist = -10.
        for i, dist in zip(indices, dists):
            if abs(dist-old_dist) > self.cutoff:
                layers.append([i])
            else:
                layers[-1].append(i)
            old_dist = dist

        if self.secondary_symmetry is not None:
            indices2, dists2 = self.get_nblist_from_center_atom(
                               symmetry=self.secondary_symmetry)
            layers2 = []
            old_dist2 = -10.
            for j, dist2 in zip(indices2, dists2):
                if abs(dist2-old_dist2) > self.secondary_cutoff:
                    layers2.append([j])
                else:
                    layers2[-1].append(j)
                old_dist2 = dist2

            res = []
            for layer in layers:
                res2 = []
                for layer2 in layers2:
                    match = [i for i in layer if i in layer2]
                    if match:
                        res2.append(match)
                res += res2
            layers = res
 
        return layers

    def run(self, max_gen=None, mode='systematic', verbose=False):
        """Run the chemical ordering generator.

        Parameters
        ----------
        max_gen : int, default None
            Maximum number of chemical orderings to generate. Enumerate
            all symetric patterns if not specified. 

        mode : str, default 'systematic'
            Mode 'systematic' = enumerate all possible chemical orderings.
            Mode 'stochastic' = sample chemical orderings stochastically.
            Stocahstic mode is recommended when there are many layers.

        verbose : bool, default False 
            Whether to print out information about number of layers and
            number of generated structures.

        """

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
            if nlayers > self.layer_threshold:
                if verbose:
                    print('{} layers is infeasible for systematic'.format(nlayers), 
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
            too_few = (2 ** nlayers * 0.95 <= max_gen)
            if verbose:
                print('Too few layers. The generated images are not all unique.')

            while True:
                comb = tuple(np.random.choice(self.species, size=nlayers))
                if comb not in combs or too_few: 
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
                            break
        if verbose:
            print('{} symmetric chemical orderings generated'.format(n_write))


class RandomOrderingGenerator(object):
    """`RandomOrderingGenerator` is a class for generating random 
    chemical orderings for a bimetallic catalyst. The function is 
    generalized for both periodic and non-periodic systems.
 
    Parameters
    ----------
    atoms : ase.Atoms object
        The nanoparticle or surface slab to use as a template to
        generate random chemical orderings. Accept any ase.Atoms 
        object. No need to be built-in.

    species : list of strs 
        The two metal species of the bimetallic catalyst.

    composition: dict, None
        Generate random orderings only at a certain composition.
        The dictionary contains the two speices as keys and their 
        concentrations as values. Generate orderings at all 
        compositions if not specified.

    trajectory : str, default 'patterns.traj'
        The name of the output ase trajectory file.

    append_trajectory : bool, default False
        Whether to append structures to the existing trajectory. 

    """

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
        """Run the chemical ordering generator.

        Parameters
        ----------
        n_gen : int
            Number of chemical orderings to generate.

        """

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
