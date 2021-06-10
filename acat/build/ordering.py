from ..utilities import bipartitions, partitions_into_totals
from ..utilities import numbers_from_ratio, is_list_or_tuple
from ase.geometry import get_distances
from ase.io import Trajectory, read, write
from asap3.analysis import FullCNA 
from asap3 import EMT as asapEMT
from asap3.Internal.BuiltinPotentials import Gupta
from collections import defaultdict
from itertools import product, combinations
from networkx.algorithms.components.connected import connected_components
import networkx as nx
import numpy as np
import random
import math


class SymmetricClusterOrderingGenerator(object):
    """`SymmetricClusterOrderingGenerator` is a class for generating 
    symmetric chemical orderings for a **nanoalloy**. There is no 
    limitation of the number of metal components. Please always align 
    the z direction to the symmetry axis of the nanocluster.
 
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

        **'spherical'**: centrosymmetry (groups defined by the distances 
        to the geometric center);

        **'cylindrical'**: cylindrical symmetry around z axis (groups 
        defined by the distances to the z axis);

        **'planar'**: planar symmetry around z axis (groups defined by 
        the z coordinates), common for phase-separated nanoalloys;

        **'mirror_planar'**: mirror planar symmetry around both
        z and xy plane (groups defined by the absolute z coordinate), 
        high symmetry subset of 'planar';
        'circular' = ring symmetry around z axis (groups defined by
        both z coordinate and distance to z axis);

        **'mirror_circular'**: mirror ring symmetry around both
        z and xy plane (groups defined by both absolute z coordinate 
        and distance to z axis);

        **'chemical'**: symmetry w.r.t chemical environment (groups 
        defined by the atomic energies given by a Gupta potential)

        **'geometrical'**: symmetry w.r.t geometrical environment (groups 
        defined by vertex / edge / fcc111 / fcc100 / bulk identified
        by CNA analysis);

        **'concentric'**: conventional definition of the concentric 
        shells (surface / subsurface, subsubsurface, ..., core).

    cutoff : float, default 1.0
        Maximum thickness (in Angstrom) of a single group. The thickness
        is calculated as the difference between the "distances" of the 
        closest-to-center atoms in two neighbor groups. Note that the
        criterion of "distance" depends on the symmetry. This parameter 
        works differently if the symmetry is 'chemical', 'geometrical' or 
        'concentric'. For 'chemical' it is defined as the maximum atomic
        energy difference (in eV) of a single group predicted by a Gupta
        potential. For 'geometrical' and 'concentric' it is defined as 
        the cutoff radius (in Angstrom) for CNA, and a reasonable cutoff 
        based on the lattice constant of the material will automatically 
        be used if cutoff <= 1. Use a larger cutoff if the structure is 
        distorted. 

    secondary_symmetry : str, default None
        Add a secondary symmetry check to define groups hierarchically. 
        For example, even if two atoms are classifed in the same group
        defined by the primary symmetry, they can still end up in 
        different groups if they fall into two different groups defined 
        by the secondary symmetry. Support 7 symmetries: 'spherical',
        'cylindrical', 'planar', 'mirror_planar', 'chemical', 'geometrical' 
        and 'concentric'. Note that secondary symmetry has the same 
        importance as the primary symmetry, so you can set either of the 
        two symmetries of interest as the secondary symmetry. Useful for 
        combining symmetries of different types (e.g. circular + chemical) 
        or combining symmetries with different cutoffs.

    secondary_cutoff : float, default 1.0
        Same as cutoff, except that it is for the secondary symmetry.

    composition : dict, default None
        Generate symmetric orderings only at a certain composition.
        The dictionary contains the metal elements as keys and their 
        concentrations as values. Generate orderings at all compositions 
        if not specified. Note that the computational cost scales badly 
        with the number of groups for a fixed-composition search.

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

        if isinstance(trajectory, str):
            self.trajectory = trajectory                        
        self.append_trajectory = append_trajectory

        self.groups = self.get_groups()

    def get_sorted_indices(self, symmetry):
        """Returns the indices sorted by the metric that defines different 
        groups, together with the corresponding vlues, given a specific 
        symmetry. Returns the indices sorted by geometrical environment if 
        symmetry='geometrical'. Returns the indices sorted by surface, 
        subsurface, subsubsurface, ..., core if symmetry='concentric'.

        Parameters
        ----------
        symmetry : str
            Support 7 symmetries: spherical, cylindrical, planar, 
            mirror_planar, chemical, geometrical, concentric.

        """

        atoms = self.atoms
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

        elif symmetry == 'concentric':
            if self.symmetry == 'concentric':
                rCut = None if self.cutoff <= 1. else self.cutoff
            elif self.secondary_symmetry == 'concentric':
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
                group_ids = list(argwhere_nd_searchsorted(atoms.positions, 
                                 a.positions[surf_ids])[0])
                conv_groups.append(group_ids)
                if not bulk_ids:
                    return 
                get_surf_ids(a[bulk_ids])

            conv_groups = []
            atoms.center(vacuum=5.)
            get_surf_ids(atoms)
            return conv_groups, None

        else:
            raise NotImplementedError("Symmetry '{}' is not supported".format(symmetry))

        sorted_indices = np.argsort(np.ravel(dists))
        return sorted_indices, dists[sorted_indices]    
    
    def get_groups(self):
        """Get the groups (a list of lists of atom indices) of all
        symmetry-equivalent atoms."""

        if self.symmetry == 'circular':
            symmetry = 'planar'
        elif self.symmetry == 'mirror_circular':
            symmetry = 'mirror_planar'
        else:
            symmetry = self.symmetry
        indices, dists = self.get_sorted_indices(symmetry=symmetry) 

        if self.symmetry in ['geometrical', 'concentric']:
            groups = indices
        else:
            groups = []
            old_dist = -10.
            for i, dist in zip(indices, dists):
                if abs(dist - old_dist) > self.cutoff:
                    groups.append([i])
                    old_dist = dist
                else:
                    groups[-1].append(i)

        if self.symmetry in ['circular', 'mirror_circular']:
            indices0, dists0 = self.get_sorted_indices(symmetry='cylindrical')
            groups0 = []
            old_dist0 = -10.
            for j, dist0 in zip(indices0, dists0):
                if abs(dist0 - old_dist0) > self.cutoff:
                    groups0.append([j])
                    old_dist0 = dist0
                else:
                    groups0[-1].append(j)

            res = []
            for group in groups:
                res0 = []
                for group0 in groups0:
                    match = [i for i in group if i in group0]
                    if match:
                        res0.append(match)
                res += res0
            groups = res

        if self.secondary_symmetry is not None:
            indices2, dists2 = self.get_sorted_indices(symmetry=
                                                       self.secondary_symmetry)
            if self.secondary_symmetry in ['geometrical', 'concentric']:
                groups2 = indices2
            else:
                groups2 = []
                old_dist2 = -10.
                for j, dist2 in zip(indices2, dists2):
                    if abs(dist2 - old_dist2) > self.secondary_cutoff:
                        groups2.append([j])
                        old_dist2 = dist2
                    else:
                        groups2[-1].append(j)

            res = []
            for group in groups:
                res2 = []
                for group2 in groups2:
                    match = [i for i in group if i in group2]
                    if match:
                        res2.append(match)
                res += res2
            groups = res
 
        return groups

    def run(self, max_gen=None, mode='systematic', verbose=False):
        """Run the chemical ordering generator.

        Parameters
        ----------
        max_gen : int, default None
            Maximum number of chemical orderings to generate. Enumerate
            all symetric patterns if not specified. 

        mode : str, default 'systematic'
            **'systematic'**: enumerate all possible unique chemical 
            orderings. Recommended when there are not many groups. Switch
            to stochastic mode automatically if the number of groups is
            more than 20. This mode is the only option when the 
            composition is fixed.

            **'stochastic'**: sample chemical orderings stochastically.
            Duplicate structures can be generated. Recommended when there 
            are many groups. Switch to systematic mode automatically if 
            the composition is fixed.

        verbose : bool, default False 
            Whether to print out information about number of groups and
            number of generated structures.

        """

        traj_mode = 'a' if self.append_trajectory else 'w'
        traj = Trajectory(self.trajectory, mode=traj_mode)
        atoms = self.atoms.copy()
        groups = self.groups
        ngroups = len(groups)
        n_write = 0
        if verbose:
            print('{} groups classified'.format(ngroups))

        if self.composition is not None:
            keys = list(self.num_dict.keys())
            totals = list(self.num_dict.values())
            if max_gen is None:
                max_gen = -1             
 
            for part in partitions_into_totals(groups, totals):
                for j in range(len(totals)):
                    ids = [i for group in part[j] for i in group] 
                    atoms.symbols[ids] = len(ids) * keys[j]
                traj.write(atoms)
                n_write += 1
                if n_write == max_gen:
                    break

        else: 
            # When the number of groups is too large (> 20), systematic enumeration 
            # is not feasible. Stochastic sampling is the only option
            if mode == 'systematic':
                if ngroups > 20:
                    if verbose:
                        print('{} groups is infeasible for systematic'.format(ngroups), 
                              'generator. Use stochastic generator instead')
                    mode = 'stochastic'
                else:    
                    combos = list(product(self.elements, repeat=ngroups))
                    random.shuffle(combos)
                    for combo in combos:
                        for j, spec in enumerate(combo):
                            atoms.symbols[groups[j]] = spec
                        traj.write(atoms)
                        n_write += 1
                        if max_gen is not None:
                            if n_write == max_gen:
                                break
 
            if mode == 'stochastic':
                combos = set()
                too_few = (2 ** ngroups * 0.95 <= max_gen)
                if too_few and verbose:
                    print('Too few groups. The generated images are not all unique.')
                while True:
                    combo = tuple(np.random.choice(self.elements, size=ngroups))
                    if combo not in combos or too_few: 
                        combos.add(combo)
                        for j, spec in enumerate(combo):
                            atoms.symbols[groups[j]] = spec
                        traj.write(atoms)
                        n_write += 1
                        if max_gen is not None:
                            if n_write == max_gen:
                                break
        if verbose:
            print('{} symmetric chemical orderings generated'.format(n_write))


class OrderedSlabOrderingGenerator(object):

    """`OrderedSlabOrderingGenerator` is a class for generating 
    ordered chemical orderings for a **alloy surface slab**. 
    There is no limitation of the number of metal components.
 
    Parameters
    ----------
    atoms : ase.Atoms object
        The surface slab to use as a template to generate ordered 
        chemical orderings. Accept any ase.Atoms object. No need 
        to be built-in.

    elements : list of strs 
        The metal elements of the alloy catalyst.

    repeating_size : list of ints or tuple of ints, default (2, 2)
        The multiples that describe the size of the repeating pattern
        on the surface. Symmetry-equivalent atoms are grouped by 
        the multiples in the x and y directions. The x or y length of 
        the cell must be this multiple of the distance between each 
        pair of symmetry-equivalent atoms. Larger reducing size 
        generates fewer structures.

    composition : dict, default None
        Generate ordered orderings only at a certain composition. 
        The dictionary contains the metal elements as keys and their 
        concentrations as values. Generate orderings at all 
        compositions if not specified. Note that the computational 
        cost scales badly with the number of groups for a 
        fixed-composition search.

    dtol : float, default 0.01
        The distance tolerance (in Angstrom) when comparing with 
        (cell length / multiple). Use a larger value if the structure 
        is distorted.

    ztol : float, default 0.1
        The tolerance (in Angstrom) when comparing z values. Use a 
        larger ztol if the structure is distorted.

    trajectory : str, default 'orderings.traj'
        The name of the output ase trajectory file.

    append_trajectory : bool, default False
        Whether to append structures to the existing trajectory. 

    """

    def __init__(self, atoms, elements,
                 repeating_size=(2, 2),
                 composition=None,
                 dtol=0.01,
                 ztol=0.1,
                 trajectory='orderings.traj',
                 append_trajectory=False):

        self.atoms = atoms
        self.elements = elements

        assert (is_list_or_tuple(repeating_size)) and (len(repeating_size) == 2)
        self.repeating_size = repeating_size
        self.dtol = dtol
        self.ztol = ztol

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

        self.groups = self.get_groups()

    def get_groups(self):
        """Get the groups (a list of lists of atom indices) of all
        symmetry-equivalent atoms."""

        atoms = self.atoms
        ds = atoms.get_all_distances(mic=True)
        cell = atoms.cell
        z_positions = atoms.positions[:,2]
        x_cell = np.linalg.norm(cell[0])
        y_cell = np.linalg.norm(cell[1])

        ref_x_dist = x_cell / self.repeating_size[0]
        ref_y_dist = y_cell / self.repeating_size[1]

        x_pairs = np.column_stack(np.where(abs(ds - ref_x_dist) < self.dtol))
        y_pairs = np.column_stack(np.where(abs(ds - ref_y_dist) < self.dtol))
        pairs = x_pairs.tolist() + y_pairs.tolist()
        pairs = [p for p in pairs if abs(z_positions[p[0]] - 
                 z_positions[p[1]]) < self.ztol]

        def to_edges(lst):
            it = iter(lst)
            last = next(it) 
            for current in it:
                yield last, current
                last = current    

        G = nx.Graph()
        for p in pairs:
            # each sublist is a bunch of nodes
            G.add_nodes_from(p)
            # it also imlies a number of edges:
            G.add_edges_from(to_edges(p))

        groups = [list(cc) for cc in list(connected_components(G))]

        return groups

    def run(self, max_gen=None, mode='systematic', verbose=False):
        """Run the chemical ordering generator.

        Parameters
        ----------
        max_gen : int, default None
            Maximum number of chemical orderings to generate. Enumerate
            all symetric patterns if not specified. 

        mode : str, default 'systematic'
            **'systematic'**: enumerate all possible unique chemical 
            orderings. Recommended when there are not many groups. Switch
            to stochastic mode automatically if the number of groups is
            more than 20. This mode is the only option when the 
            composition is fixed.

            **'stochastic'**: sample chemical orderings stochastically.
            Duplicate structures can be generated. Recommended when there 
            are many groups. Switch to systematic mode automatically if 
            the composition is fixed.

        verbose : bool, default False 
            Whether to print out information about number of groups and
            number of generated structures.

        """

        traj_mode = 'a' if self.append_trajectory else 'w'
        traj = Trajectory(self.trajectory, mode=traj_mode)
        atoms = self.atoms.copy()
        groups = self.groups
        ngroups = len(groups)
        n_write = 0
        if verbose:
            print('{} groups classified'.format(ngroups))

        if self.composition is not None:
            keys = list(self.num_dict.keys())
            totals = list(self.num_dict.values())
            if max_gen is None:
                max_gen = -1             
 
            for part in partitions_into_totals(groups, totals):
                for j in range(len(totals)):
                    ids = [i for group in part[j] for i in group] 
                    atoms.symbols[ids] = len(ids) * keys[j]
                traj.write(atoms)
                n_write += 1
                if n_write == max_gen:
                    break

        else: 
            # When the number of groups is too large (> 20), systematic enumeration 
            # is not feasible. Stochastic sampling is the only option
            if mode == 'systematic':
                if ngroups > 20:
                    if verbose:
                        print('{} groups is infeasible for systematic'.format(ngroups), 
                              'generator. Use stochastic generator instead')
                    mode = 'stochastic'
                else:    
                    combos = list(product(self.elements, repeat=ngroups))
                    random.shuffle(combos)
                    for combo in combos:
                        for j, spec in enumerate(combo):
                            atoms.symbols[groups[j]] = spec
                        traj.write(atoms)
                        n_write += 1
                        if max_gen is not None:
                            if n_write == max_gen:
                                break
 
            if mode == 'stochastic':
                combos = set()
                too_few = (2 ** ngroups * 0.95 <= max_gen)
                if too_few and verbose:
                    print('Too few groups. The generated images are not all unique.')
                while True:
                    combo = tuple(np.random.choice(self.elements, size=ngroups))
                    if combo not in combos or too_few: 
                        combos.add(combo)
                        for j, spec in enumerate(combo):
                            atoms.symbols[groups[j]] = spec
                        traj.write(atoms)
                        n_write += 1
                        if max_gen is not None:
                            if n_write == max_gen:
                                break
        if verbose:
            print('{} ordered chemical orderings generated'.format(n_write))


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

    composition : dict, default None
        Generate random orderings only at a certain composition.
        The dictionary contains the metal elements as keys and 
        their concentrations as values. Generate orderings at all 
        compositions if not specified.

    trajectory : str, default 'orderings.traj'
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
