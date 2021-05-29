from acat.utilities import numbers_from_ratio
from ase.geometry import get_distances
from ase.io import Trajectory, read, write
from asap3.analysis import FullCNA 
from asap3 import EMT as asapEMT
from asap3.Internal.BuiltinPotentials import Gupta
from collections import defaultdict
from itertools import product, combinations
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
        Support 9 symmetries: 

        **'spherical'**: centrosymmetry (shells defined by the distances 
        to the geometric center);

        **'cylindrical'**: cylindrical symmetry around z axis (shells 
        defined by the distances to the z axis);

        **'planar'**: planar symmetry around z axis (shells defined by 
        the z coordinates), common for phase-separated nanoalloys;

        **'mirror_planar'**: mirror planar symmetry around both
        z and xy plane (shells defined by the absolute z coordinate), 
        high symmetry subset of 'planar';
        'circular' = ring symmetry around z axis (shells defined by
        both z coordinate and distance to z axis);

        **'mirror_circular'**: mirror ring symmetry around both
        z and xy plane (shells defined by both absolute z coordinate 
        and distance to z axis);

        **'chemical'**: symmetry w.r.t chemical environment (shells 
        defined by the atomic energies given by a Gupta potential)

        **'geometrical'**: symmetry w.r.t geometrical environment (shells 
        defined by vertex / edge / fcc111 / fcc100 / bulk identified
        by CNA analysis);

        **'conventional'**: conventional definition of shells (surface /
        subsurface, subsubsurface, ..., core).

    cutoff : float, default 1.0
        Maximum thickness (in Angstrom) of a single shell. The thickness
        is calculated as the difference between the "distances" of the 
        closest-to-center atoms in two neighbor shells. Note that the
        criterion of "distance" depends on the symmetry. This parameter 
        works differently if the symmetry is 'chemical', 'geometrical' or 
        'conventional'. For 'chemical' it is defined as the maximum atomic
        energy difference (in eV) of a single shell predicted by a Gupta
        potential. For 'geometrical' and 'conventional' it is defined as 
        the cutoff radius (in Angstrom) for CNA, and a reasonable cutoff 
        based on the lattice constant of the material will automatically 
        be used if cutoff <= 1. Use a larger cutoff if the structure is 
        distorted. 

    secondary_symmetry : str, default None
        Add a secondary symmetry check to define shells hierarchically. 
        For example, even if two atoms are classifed in the same shell
        defined by the primary symmetry, they can still end up in 
        different shells if they fall into two different shells defined 
        by the secondary symmetry. Support 7 symmetries: 'spherical',
        'cylindrical', 'planar', 'mirror_planar', 'chemical', 'geometrical' 
        and 'conventional'. Note that secondary symmetry has the same 
        importance as the primary symmetry, so you can set either of the 
        two symmetries of interest as the secondary symmetry. Useful for 
        combining symmetries of different types (e.g. circular + chemical) 
        or combining symmetries with different cutoffs.

    secondary_cutoff : float, default 1.0
        Same as cutoff, except that it is for the secondary symmetry.

    composition : dict, None
        Generate symmetric orderings only at a certain composition.
        The dictionary contains the metal elements as keys and their 
        concentrations as values. Generate orderings at all compositions 
        if not specified. Note that the computational cost scales badly 
        with the number of shells for a fixed-composition search.

    shell_threshold : int, default 20
        Number of shells to switch to stochastic mode automatically.

    trajectory : str, default 'orderings.traj'
        The name of the output ase trajectory file.

    append_trajectory : bool, default False
        Whether to append structures to the existing trajectory. 

    """

    def __init__(self, atoms, elements,
                 symmetry='spherical',
                 cutoff=1.,       
                 secondary_symmetry=None,
                 secondary_cutoff=1.,
                 composition=None,
                 shell_threshold=20,
                 trajectory='orderings.traj',
                 append_trajectory=False):

        self.atoms = atoms
        self.elements = elements
        self.symmetry = symmetry
        self.cutoff = cutoff
        assert secondary_symmetry not in ['circular', 'mirror_circular']
        self.secondary_symmetry = secondary_symmetry
        self.secondary_cutoff = secondary_cutoff

        self.composition = composition
        if self.composition is not None:
            ks = list(self.composition.keys())
            assert set(ks) == set(self.elements)
            vs = list(self.composition.values())
            nums = numbers_from_ratio(len(self.atoms), vs)
            self.num_dict = {ks[i]: nums[i] for i in range(len(ks))}

        self.shell_threshold = shell_threshold
        if isinstance(trajectory, str):
            self.trajectory = trajectory                        
        self.append_trajectory = append_trajectory

        self.shells = self.get_shells()

    def get_sorted_indices(self, symmetry):
        """Returns the indices sorted by the metric that defines different 
        shells, together with the corresponding vlues, given a specific 
        symmetry. Returns the indices sorted by geometrical environment if 
        symmetry='geometrical'. Returns the indices sorted by surface, 
        subsurface, subsubsurface, ..., core if symmetry='conventional'.

        Parameters
        ----------
        symmetry : str
            Support 7 symmetries: spherical, cylindrical, planar, 
            mirror_planar, chemical, geometrical, conventional.

        """

        atoms = self.atoms.copy()
        atoms.center()
        geo_mid = [(atoms.cell/2.)[0][0], (atoms.cell/2.)[1][1], 
                   (atoms.cell/2.)[2][2]]
        if symmetry == 'spherical':
            dists = get_distances(atoms.positions, [geo_mid])[1][:,0]

        elif symmetry == 'cylindrical':
            dists = np.asarray([math.sqrt((a.position[0] - geo_mid[0])**2 + 
                               (a.position[1] - geo_mid[1])**2) for a in atoms])

        elif symmetry == 'planar':
            dists = atoms.positions[:, 2]

        elif symmetry == 'mirror_planar':
            dists = abs(atoms.positions[:, 2] - geo_mid[2])

        elif symmetry == 'chemical':
            gupta_parameters = {'Cu': [10.960, 2.2780, 0.0855, 1.224, 2.556]}
            calc = Gupta(gupta_parameters, cutoff=1000, debug=False)
            for a in atoms:
                a.symbol = 'Cu'
            atoms.center(vacuum=5.)
            atoms.calc = calc
            dists = atoms.get_potential_energies()
            atoms.calc = None

        elif symmetry == 'geometrical':
            if self.symmetry == 'geometrical':
                rCut = None if self.cutoff <= 1. else self.cutoff
            elif self.secondary_symmetry == 'geometrical':
                rCut = None if self.secondary_cutoff <= 1. else self.secondary_cutoff
            atoms.center(vacuum=5.)
            fcna = FullCNA(atoms, rCut=rCut).get_normal_cna()
            d = defaultdict(list)
            for i, x in enumerate(fcna):
                if sum(x.values()) < 12:
                    d[str(x)].append(i)
                else:
                    d['bulk'].append(i)
            return list(d.values()), None

        elif symmetry == 'conventional':
            if self.symmetry == 'conventional':
                rCut = None if self.cutoff <= 1. else self.cutoff
            elif self.secondary_symmetry == 'conventional':
                rCut = None if self.secondary_cutoff <= 1. else self.secondary_cutoff

            def view1D(a, b): # a, b are arrays
                a = np.ascontiguousarray(a)
                b = np.ascontiguousarray(b)
                void_dt = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
                return a.view(void_dt).ravel(),  b.view(void_dt).ravel()

            def argwhere_nd_searchsorted(a,b):
                A,B = view1D(a,b)
                sidxB = B.argsort()
                mask = np.isin(A,B)
                cm = A[mask]
                idx0 = np.flatnonzero(mask)
                idx1 = sidxB[np.searchsorted(B,cm, sorter=sidxB)]
                return idx0, idx1 # idx0 : indices in A, idx1 : indices in B

            def get_surf_ids(a):
                fcna = FullCNA(a, rCut=rCut).get_normal_cna() 
                surf_ids, bulk_ids = [], []
                for i in range(len(a)):
                    if sum(fcna[i].values()) < 12:
                        surf_ids.append(i)
                    else:
                        bulk_ids.append(i)
                shell_ids = list(argwhere_nd_searchsorted(atoms.positions, 
                                 a.positions[surf_ids])[0])
                conv_shells.append(shell_ids)
                if not bulk_ids:
                    return 
                get_surf_ids(a[bulk_ids])

            conv_shells = []
            atoms.center(vacuum=5.)
            get_surf_ids(atoms)
            return conv_shells, None

        else:
            raise NotImplementedError("Symmetry '{}' is not supported".format(symmetry))

        sorted_indices = np.argsort(np.ravel(dists))
        return sorted_indices, dists[sorted_indices]    
    
    def get_shells(self):
        """Get the shells (a list of lists of atom indices) that divided 
        by the symmetry."""

        if self.symmetry == 'circular':
            symmetry = 'planar'
        elif self.symmetry == 'mirror_circular':
            symmetry = 'mirror_planar'
        else:
            symmetry = self.symmetry
        indices, dists = self.get_sorted_indices(symmetry=symmetry) 

        if self.symmetry in ['geometrical', 'conventional']:
            shells = indices
        else:
            shells = []
            old_dist = -10.
            for i, dist in zip(indices, dists):
                if abs(dist - old_dist) > self.cutoff:
                    shells.append([i])
                    old_dist = dist
                else:
                    shells[-1].append(i)

        if self.symmetry in ['circular', 'mirror_circular']:
            indices0, dists0 = self.get_sorted_indices(symmetry='cylindrical')
            shells0 = []
            old_dist0 = -10.
            for j, dist0 in zip(indices0, dists0):
                if abs(dist0 - old_dist0) > self.cutoff:
                    shells0.append([j])
                    old_dist0 = dist0
                else:
                    shells0[-1].append(j)

            res = []
            for shell in shells:
                res0 = []
                for shell0 in shells0:
                    match = [i for i in shell if i in shell0]
                    if match:
                        res0.append(match)
                res += res0
            shells = res

        if self.secondary_symmetry is not None:
            indices2, dists2 = self.get_sorted_indices(symmetry=
                                                       self.secondary_symmetry)
            if self.secondary_symmetry in ['geometrical', 'conventional']:
                shells2 = indices2
            else:
                shells2 = []
                old_dist2 = -10.
                for j, dist2 in zip(indices2, dists2):
                    if abs(dist2 - old_dist2) > self.secondary_cutoff:
                        shells2.append([j])
                        old_dist2 = dist2
                    else:
                        shells2[-1].append(j)

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
        n_write = 0
        if verbose:
            print('{} shells classified'.format(nshells))

        if self.composition is not None:
            def bipartitions(shells, total):
                n = len(shells)
                for k in range(n + 1):
                    for combo in combinations(range(n), k):
                        if sum(len(shells[i]) for i in combo) == total:
                            set_combo = set(combo)
                            yield sorted(shells[i] for i in combo), sorted(
                            shells[i] for i in range(n) if i not in set_combo)

            def partitions_into_totals(shells, totals):
                assert totals
                if len(totals) == 1:
                    yield [shells]
                else:
                    for first, remaining_shells in bipartitions(shells, totals[0]):
                        for rest in partitions_into_totals(remaining_shells, totals[1:]):
                            yield [first] + rest

            keys = list(self.num_dict.keys())
            totals = list(self.num_dict.values())
            if max_gen is None:
                max_gen = -1             
 
            for part in partitions_into_totals(shells, totals):
                for j in range(len(totals)):
                    ids = [i for shell in part[j] for i in shell] 
                    atoms.symbols[ids] = len(ids) * keys[j]
                traj.write(atoms)
                n_write += 1
                if n_write == max_gen:
                    break

        else: 
            # When the number of shells is too large (> 20), systematic enumeration 
            # is not feasible. Stochastic sampling is the only option
            if mode == 'systematic':
                if nshells > self.shell_threshold:
                    if verbose:
                        print('{} shells is infeasible for systematic'.format(nshells), 
                              'generator. Use stochastic generator instead')
                    mode = 'stochastic'
                else:    
                    combos = list(product(self.elements, repeat=nshells))
                    random.shuffle(combos)
                    for combo in combos:
                        for j, spec in enumerate(combo):
                            atoms.symbols[shells[j]] = spec
                        traj.write(atoms)
                        n_write += 1
                        if max_gen is not None:
                            if n_write == max_gen:
                                break
 
            if mode == 'stochastic':
                combos = set()
                too_few = (2 ** nshells * 0.95 <= max_gen)
                if too_few and verbose:
                    print('Too few shells. The generated images are not all unique.')
                while True:
                    combo = tuple(np.random.choice(self.elements, size=nshells))
                    if combo not in combos or too_few: 
                        combos.add(combo)
                        for j, spec in enumerate(combo):
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

    composition : dict, None
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
            ks = list(self.composition.keys())
            assert set(ks) == set(self.elements)
            vs = list(self.composition.values())
            nums = numbers_from_ratio(len(self.atoms), vs)
            self.num_dict = {ks[i]: nums[i] for i in range(len(ks))}

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
