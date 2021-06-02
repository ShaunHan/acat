"""Procreation operators meant to be used in symmetry-constrained
genetic algorithm (SCGA)."""
from ase.ga.offspring_creator import OffspringCreator
from collections import defaultdict
import random


class Mutation(OffspringCreator):
    """Base class for all particle mutation type operators.
    Do not call this class directly."""

    def __init__(self, num_muts=1):
        OffspringCreator.__init__(self, num_muts=num_muts)
        self.descriptor = 'Mutation'
        self.min_inputs = 1


class GroupSubstitute(Mutation):
    """Substitute all the atoms in a group with a different element. 
    The elemental composition cannot be fixed.

    Parameters
    ----------
    groups : list of lists
        The atom indices in each user-divided group. Can be obtained 
        by `acat.build.ordering.SymmetricClusterOrderingGenerator` 
        or `acat.build.ordering.OrderedSlabOrderingGenerator`.

    elements : list of strs, default None
        Only take into account the elements specified in this list. 
        Default is to take all elements into account.

    num_muts : int, default 1
        The number of times to perform this operation.

    """

    def __init__(self, groups, 
                 elements=None,
                 num_muts=1):
        Mutation.__init__(self, num_muts=num_muts)

        assert len(elements) >= 2
        self.descriptor = 'GroupSubstitute'
        self.elements = elements
        self.groups = groups

    def substitute(self, atoms):
        """Does the actual substitution"""

        atoms = atoms.copy() 

        if self.elements is None:
            e = list(set(atoms.get_chemical_symbols()))
        else:
            e = self.elements

        sorted_elems = sorted(set(atoms.get_chemical_symbols()))
        if e is not None and sorted(e) != sorted_elems:
            for group in self.groups:
                torem = []
                for i in group:
                    if atoms[i].symbol not in e:
                        torem.append(i)
                for i in torem:
                    group.remove(i)

        itbms = random.sample(range(len(self.groups)), self.num_muts)
        
        for itbm in itbms:
            mut_group = self.groups[itbm]
            other_elements = [e for e in self.elements if 
                              e != atoms[mut_group[0]].symbol]
            to_element = random.choice(other_elements)
            atoms.symbols[mut_group] = len(mut_group) * to_element

        return atoms

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.substitute(f)
        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]

        return (self.finalize_individual(indi),
                self.descriptor + ':Parent {0}'.format(f.info['confid']))


class GroupPermutation(Mutation):
    """Permutes the elements in two random groups. The elemental 
    composition can be fixed.

    Parameters
    ----------
    groups : list of lists
        The atom indices in each user-divided group. Can be obtained 
        by `acat.build.ordering.SymmetricClusterOrderingGenerator` 
        or `acat.build.ordering.OrderedSlabOrderingGenerator`.

    elements : list of strs, default None
        Only take into account the elements specified in this list. 
        Default is to take all elements into account.

    keep_composition : bool, defulat False
        Whether the elemental composition should be the same as in
        the parents.

    num_muts : int, default 1
        The number of times to perform this operation.

    """

    def __init__(self, groups,
                 elements=None,
                 keep_composition=False, 
                 num_muts=1):
        Mutation.__init__(self, num_muts=num_muts)

        assert len(elements) >= 2
        self.descriptor = 'GroupPermutation'
        self.elements = elements
        self.keep_composition = keep_composition
        self.groups = groups

    def get_new_individual(self, parents):
        f = parents[0].copy()

        diffatoms = len(set(f.numbers))
        assert diffatoms > 1, 'Permutations with one atomic type is not valid'

        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]

        for _ in range(self.num_muts):
            GroupPermutation.mutate(f, self.groups, self.elements,
                                        self.keep_composition)

        for atom in f:
            indi.append(atom)

        return (self.finalize_individual(indi),
                self.descriptor + ':Parent {0}'.format(f.info['confid']))

    @classmethod
    def mutate(cls, atoms, groups, elements=None, keep_composition=False):
        """Do the actual permutation."""

        if elements is None:
            e = list(set(atoms.get_chemical_symbols()))
        else:
            e = elements

        sorted_elems = sorted(set(atoms.get_chemical_symbols()))
        if e is not None and sorted(e) != sorted_elems:
            for group in groups:
                torem = []
                for i in group:
                    if atoms[i].symbol not in e:
                        torem.append(i)
                for i in torem:
                    group.remove(i)

        if keep_composition:
            dd = defaultdict(list)
            for si, group in enumerate(groups):
                dd[len(group)].append(si)
            items = list(dd.items())
            random.shuffle(items)
            mut_sis = None
            for k, v in items:
                if len(v) > 1:
                    mut_sis = v
                    break
            if mut_sis is None:
                return
            random.shuffle(mut_sis)
            i1 = mut_sis[0]
            mut_group1 = groups[i1]           
            options = [i for i in mut_sis[1:] if atoms[mut_group1[0]].symbol 
                       != atoms[groups[i][0]].symbol]

        else:
            i1 = random.randint(0, len(groups) - 1)
            mut_group1 = groups[i1]
            options = [i for i in range(0, len(groups)) if atoms[mut_group1[0]].symbol 
                       != atoms[groups[i][0]].symbol]

        if not options:
            return
        i2 = random.choice(options)
        mut_group2 = groups[i2]
        atoms.symbols[mut_group1+mut_group2] = len(mut_group1) * atoms[
        mut_group2[0]].symbol + len(mut_group2) * atoms[mut_group1[0]].symbol


class Crossover(OffspringCreator):
    """Base class for all particle crossovers.
    Do not call this class directly."""
    def __init__(self):
        OffspringCreator.__init__(self)
        self.descriptor = 'Crossover'
        self.min_inputs = 2


class GroupCrossover(Crossover):
    """Merge the elemental distributions in two half groups from 
    different structures together. The elemental composition can be 
    fixed.

    Parameters
    ----------
    groups : list of lists
        The atom indices in each user-divided group. Can be obtained 
        by `acat.build.ordering.SymmetricClusterOrderingGenerator` 
        or `acat.build.ordering.OrderedSlabOrderingGenerator`.

    elements : list of strs, default None
        Only take into account the elements specified in this list. 
        Default is to take all elements into account.

    keep_composition : bool, defulat False
        Whether the elemental composition should be the same as in
        the parents.

    """

    def __init__(self, groups, elements=None, keep_composition=False):
        Crossover.__init__(self)
        self.groups = groups
        self.elements = elements
        self.keep_composition = keep_composition
        self.descriptor = 'GroupCrossover'
        
    def get_new_individual(self, parents):
        f, m = parents

        indi = f.copy()
        groups = self.groups.copy()
        if self.elements is None:
            e = list(set(f.get_chemical_symbols()))
        else:
            e = self.elements

        sorted_elems = sorted(set(f.get_chemical_symbols()))
        if e is not None and sorted(e) != sorted_elems:
            for group in groups:
                torem = []
                for i in group:
                    if f[i].symbol not in e:
                        torem.append(i)
                for i in torem:
                    group.remove(i)

        random.shuffle(groups)
        if self.keep_composition:
            divisors = list(range(2, len(groups)+1))
            random.shuffle(divisors)
            mids = None
            for divisor in divisors: 
                mgroups = groups[len(groups)//divisor:]
                tmpmids = [i for group in mgroups for i in group] 
                if sorted(indi.symbols[tmpmids]) == sorted(m.symbols[tmpmids]):
                    mids = tmpmids
                    break
            if mids is None:
                m = f.copy()

        else:
            mgroups = groups[len(groups)//2:]
            mids = [i for group in mgroups for i in group]

        for i in mids:
            indi[i].symbol = m[i].symbol
        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [i.info['confid'] for i in parents] 
        indi.info['data']['operation'] = 'crossover'
        parent_message = ':Parents {0} {1}'.format(f.info['confid'],
                                                   m.info['confid']) 

        return (self.finalize_individual(indi),
                self.descriptor + parent_message)
