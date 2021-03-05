from ase.geometry import get_distances
from ase.io import Trajectory, read, write
from asap3.analysis import FullCNA 
from collections import defaultdict
from itertools import product
import numpy as np
import random
import math


class SymmetricOrderingGenerator(object):
    """`SymmetricOrderingGenerator` is a class for generating 
    symmetric chemical orderings for a nanoalloy.
    As for now only support clusters without fixing the composition, 
    but there is no limitation of the number of metal components. 
    Please align the z direction to the symmetry axis of the cluster.
 
    Parameters
    ----------
    atoms : ase.Atoms object
        The nanoparticle to use as a template to generate symmetric
        chemical orderings. Accept any ase.Atoms object. No need to be 
        built-in.

    elements : list of strs 
        The metal elements of the nanoalloy.

    symmetry : str, default 'spherical'
        Support 5 symmetries: 
        'spherical' = centrosymmetry (shells defined by the distances 
        to the geometric center);
        'cylindrical' = cylindrical symmetry around z axis (shells 
        defined by the distances to the z axis);
        'planar' = planar symmetry around z axis (shells defined by 
        the z coordinates);
        'parallel_planar' = double planar symmetry around both z and 
        x (or y) axis (shells defined by the z vertical distance to
        the geometric center);
        'chemical' = symmetry w.r.t chemical environment (shells 
        defined by vertex / edge / fcc111 / fcc100 / bulk identified
        by CNA analysis).

    cutoff: float, default 0.1
        Minimum distance (in Angstrom) that the code can distinguish 
        between two neighbor shells. If symmetry='chemical', this is 
        the cutoff radius (in Angstrom) for CNA and will automatically 
        use a reasonable cutoff based on the lattice constant of the 
        material if cutoff < 1. If the structure is distorted, please 
        use a larger cutoff. 

    secondary_symmetry : str, default None
        Add a secondary symmetry check to define shells hierarchically. 
        For example, even if two atoms are classifed in one shell that 
        defined by the primary symmetry, they can still end up in 
        different shells if they fall into two different shells that 
        defined by the secondary symmetry. Support same 5 symmetries. 
        Note that secondary symmetry has the same importance as the 
        primary symmetry, so you can set either of the two symmetries 
        of interest as the secondary symmetry.

    secondary_cutoff : float, default 0.1
        Same as cutoff, except that it is for the secondary symmetry.

    shell_threshold : int, default 20
        Number of shells to switch to stochastic mode automatically.

    trajectory : str, default 'orderings.traj'
        The name of the output ase trajectory file.

    append_trajectory : bool, default False
        Whether to append structures to the existing trajectory. 

    """

    def __init__(self, atoms, elements,
                 symmetry='spherical', #'planar' / 'cylindrical' / 'chemical'
                 cutoff=.1,       
                 secondary_symmetry=None,
                 secondary_cutoff=.1,
                 shell_threshold=20,
                 trajectory='orderings.traj',
                 append_trajectory=False):

        self.atoms = atoms
        self.elements = elements
        self.symmetry = symmetry
        self.cutoff = cutoff
        self.secondary_symmetry = secondary_symmetry
        self.secondary_cutoff = secondary_cutoff

        self.shell_threshold = shell_threshold
        if isinstance(trajectory, str):
            self.trajectory = trajectory                        
        self.append_trajectory = append_trajectory

        self.shells = self.get_shells()

    def get_sorted_indices(self, symmetry):
        """Returns the indices sorted by the metric that defines different 
        shells, together with the corresponding vlues, given a specific 
        symmetry. Returns the indices sorted by chemical environment if 
        symmetry='chemical'.

        Parameters
        ----------
        symmetry : str
            Support 5 symmetries: spherical, cylindrical, planar, 
            parallel_planar, chemical.

        """

        atoms = self.atoms.copy()
        atoms.center()
        geo_mid = [(atoms.cell/2.)[0][0], (atoms.cell/2.)[1][1], 
                   (atoms.cell/2.)[2][2]]
        if symmetry == 'spherical':
            dists = get_distances(atoms.positions, [geo_mid])[1]
        elif symmetry == 'cylindrical':
            dists = np.asarray([math.sqrt((a.position[0] - geo_mid[0])**2 + 
                               (a.position[1] - geo_mid[1])**2) for a in atoms])
        elif symmetry == 'planar':
            dists = atoms.positions[:, 2]
        elif symmetry == 'parallel_planar':
            dists = abs(atoms.positions[:, 2] - geo_mid[2])
        elif symmetry == 'chemical':
            if self.symmetry == 'chemical':
                rCut = None if self.cutoff < 1. else self.cutoff
            elif self.secondary_symmetry == 'chemical':
                rCut = None if self.secondary_cutoff < 1. else self.secondary_cutoff
            atoms.center(vacuum=5.)
            fcna = FullCNA(atoms, rCut=rCut).get_normal_cna()
            d = defaultdict(list)
            for i, x in enumerate(fcna):
                if sum(x.values()) < 12:
                    d[str(x)].append(i)
                else:
                    d['bulk'].append(i)
            return list(d.values()), None 

        else:
            raise ValueError("Symmetry '{}' is not supported".format(symmetry))

        sorted_indices = np.argsort(np.ravel(dists))
        return sorted_indices, dists[sorted_indices]    
    
    def get_shells(self):
        """Get the shells (a list of lists of atom indices) that divided 
        by the symmetry."""

        indices, dists = self.get_sorted_indices(symmetry=self.symmetry) 

        if self.symmetry == 'chemical':
            shells = indices
        else:
            shells = []
            old_dist = -10.
            for i, dist in zip(indices, dists):
                if abs(dist-old_dist) > self.cutoff:
                    shells.append([i])
                else:
                    shells[-1].append(i)
                old_dist = dist

        if self.secondary_symmetry is not None:
            indices2, dists2 = self.get_sorted_indices(symmetry=
                                                       self.secondary_symmetry)
            if self.secondary_symmetry == 'chemical':
                shells2 = indices2
            else:
                shells2 = []
                old_dist2 = -10.
                for j, dist2 in zip(indices2, dists2):
                    if abs(dist2-old_dist2) > self.secondary_cutoff:
                        shells2.append([j])
                    else:
                        shells2[-1].append(j)
                    old_dist2 = dist2

            res = []
            for shell in shells:
                res2 = []
                for shell2 in shells2:
                    match = [i for i in shell if i in shell2]
                    if match:
                        res2.append(match)
                res += res2
            shells = res
 
        return shells

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
            Stocahstic mode is recommended when there are many shells.

        verbose : bool, default False 
            Whether to print out information about number of shells and
            number of generated structures.

        """

        traj_mode = 'a' if self.append_trajectory else 'w'
        traj = Trajectory(self.trajectory, mode=traj_mode)
        atoms = self.atoms
        shells = self.shells
        nshells = len(shells)
        if verbose:
            print('{} shells classified'.format(nshells))
        n_write = 0

        # When the number of shells is too large (> 20), systematic enumeration 
        # is not feasible. Stochastic sampling is the only option
        if mode == 'systematic':
            if nshells > self.shell_threshold:
                if verbose:
                    print('{} shells is infeasible for systematic'.format(nshells), 
                          'generator. Use stochastic generator instead')
                mode = 'stochastic'
            else:    
                combs = list(product(self.elements, repeat=len(shells)))
                random.shuffle(combs)
                for comb in combs:
                    for j, spec in enumerate(comb):
                        atoms.symbols[shells[j]] = spec
                    traj.write(atoms)
                    n_write += 1
                    if max_gen is not None:
                        if n_write == max_gen:
                            break

        if mode == 'stochastic':
            combs = set()
            too_few = (2 ** nshells * 0.95 <= max_gen)
            if too_few and verbose:
                print('Too few shells. The generated images are not all unique.')
            while True:
                comb = tuple(np.random.choice(self.elements, size=nshells))
                if comb not in combs or too_few: 
                    combs.add(comb)
                    for j, spec in enumerate(comb):
                        atoms.symbols[shells[j]] = spec
                    traj.write(atoms)
                    n_write += 1
                    if max_gen is not None:
                        if n_write == max_gen:
                            break
        if verbose:
            print('{} symmetric chemical orderings generated'.format(n_write))


class RandomOrderingGenerator(object):
    """`RandomOrderingGenerator` is a class for generating random 
    chemical orderings for an alloy catalyst. The function is 
    generalized for both periodic and non-periodic systems, and 
    there is no limitation of the number of metal components.
 
    Parameters
    ----------
    atoms : ase.Atoms object
        The nanoparticle or surface slab to use as a template to
        generate random chemical orderings. Accept any ase.Atoms 
        object. No need to be built-in.

    elements : list of strs 
        The metal elements of the alloy catalyst.

    composition: dict, None
        Generate random orderings only at a certain composition.
        The dictionary contains the metal elements as keys and 
        their concentrations as values. Generate orderings at all 
        compositions if not specified.

    trajectory : str, default 'patterns.traj'
        The name of the output ase trajectory file.

    append_trajectory : bool, default False
        Whether to append structures to the existing trajectory. 

    """

    def __init__(self, atoms, elements,
                 composition=None,
                 trajectory='orderings.traj',
                 append_trajectory=False):

        self.atoms = atoms
        self.elements = elements
        self.composition = composition
        if self.composition is not None:
            assert set(self.composition.keys()) == set(self.elements)
            tot = sum(self.composition.values())
            self.num_dict = {e: int(round(self.composition[e] * 
                             len(self.atoms) / tot)) for e in self.elements}

        if isinstance(trajectory, str):
            self.trajectory = trajectory                        
        self.append_trajectory = append_trajectory

    def randint_with_sum(self):
        """Return a randomly chosen list of N positive integers i
        summing to the number of atoms. N is the number of elements.
        Each such list is equally likely to occur."""

        N = len(self.elements)
        total = len(self.atoms)
        dividers = sorted(random.sample(range(1, total), N - 1))
        return [a - b for a, b in zip(dividers + [total], [0] + dividers)]

    def random_split_indices(self):
        """Generate random chunks of indices given sizes of each 
        chunk."""

        indices = list(range(len(self.atoms)))
        random.shuffle(indices)
        res = {}
        pointer = 0
        for k, v in self.num_dict.items():
            res[k] = indices[pointer:pointer+v]
            pointer += v

        return res

    def run(self, num_gen):
        """Run the chemical ordering generator.

        Parameters
        ----------
        num_gen : int
            Number of chemical orderings to generate.

        """

        traj_mode = 'a' if self.append_trajectory else 'w'
        traj = Trajectory(self.trajectory, mode=traj_mode)
        atoms = self.atoms
        natoms = len(atoms)

        for _ in range(num_gen):
            if self.composition is None:
                rands = self.randint_with_sum()
                self.num_dict = {e: rands[i] for i, e in 
                                 enumerate(self.elements)}
            chunks = self.random_split_indices()    
            indi = atoms.copy()
            for e, ids in chunks.items():
                indi.symbols[ids] = e
            traj.write(indi)
