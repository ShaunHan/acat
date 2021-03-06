from ase.ga.offspring_creator import OffspringCreator
import random


class Mutation(OffspringCreator):
    """Base class for all particle mutation type operators.
    Do not call this class directly."""

    def __init__(self, num_muts=1):
        OffspringCreator.__init__(self, num_muts=num_muts)
        self.descriptor = 'SymmetryMutation'
        self.min_inputs = 1


class SymmetricSubstitute(Mutation):
    """Substitute all the atoms in a shell with a different metal 
    element.

    Parameters
    ----------
    shells : list of lists
        The atom indices in each shell divided by symmetry. Can be 
        obtained by `acat.build.SymmetricOrderingGenerator`.

    elements : list of strs, default None
        The metal elements of the nanoalloy. Only take into account 
        the elements specified in this list. Default is to take all 
        elements into account.

    num_muts : int, default 1
        The number of times to perform this operation.
    """

    def __init__(self, shells, 
                 elements=None, 
                 num_muts=1):
        Mutation.__init__(self, num_muts=num_muts)

        assert len(elements) >= 2
        self.descriptor = 'SymmetricSubstitute'
        self.elements = elements
        self.shells = shells

    def substitute(self, atoms):
        """Does the actual substitution"""
        atoms = atoms.copy() 

        if self.elements is None:
            e = list(set(atoms.get_chemical_symbols()))
        else:
            e = self.elements

        sorted_elems = sorted(set(atoms.get_chemical_symbols()))
        if e is not None and sorted(e) != sorted_elems:
            for shell in self.shells:
                torem = []
                for i in shell:
                    if atoms[i].symbol not in e:
                        torem.append(i)
                for i in torem:
                    shell.remove(i)

        itbms = random.sample(range(len(self.shells)), self.num_muts)
        
        for itbm in itbms:
            mut_shell = self.shells[itbm]
            other_elements = [e for e in self.elements if 
                              e != atoms[mut_shell[0]].symbol]
            to_element = random.choice(other_elements)
            atoms.symbols[mut_shell] = len(mut_shell) * to_element

        return atoms

    def get_new_individual(self, parents):
        f = parents[0]

        indi = self.substitute(f)
        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]

        return (self.finalize_individual(indi),
                self.descriptor + ':Parent {0}'.format(f.info['confid']))


class SymmetricPermutation(Mutation):
    """Permutes the elements in two random shells.

    Parameters
    ----------
    shells : list of lists
        The atom indices in each shell divided by symmetry. Can be 
        obtained by `acat.build.SymmetricOrderingGenerator`.

    elements : list of strs, default None
        The metal elements of the nanoalloy. Only take into account 
        the elements specified in this list. Default is to take all 
        elements into account.

    num_muts : int, default 1
        The number of times to perform this operation.
    """

    def __init__(self, shells,
                 elements=None, 
                 num_muts=1):
        Mutation.__init__(self, num_muts=num_muts)

        assert len(elements) >= 2
        self.descriptor = 'SymmetricPermutation'
        self.elements = elements
        self.shells = shells

    def get_new_individual(self, parents):
        f = parents[0].copy()

        diffatoms = len(set(f.numbers))
        assert diffatoms > 1, 'Permutations with one atomic type is not valid'

        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]

        for _ in range(self.num_muts):
            RandomPermutation.mutate(f, self.elements)

        for atom in f:
            indi.append(atom)

        return (self.finalize_individual(indi),
                self.descriptor + ':Parent {0}'.format(f.info['confid']))

    @classmethod
    def mutate(cls, atoms, elements=None):
        """Do the actual permutation."""

        shells = cls.shells
        if self.elements is None:
            e = list(set(atoms.get_chemical_symbols()))
        else:
            e = self.elements

        sorted_elems = sorted(set(atoms.get_chemical_symbols()))
        if e is not None and sorted(e) != sorted_elems:
            for shell in self.shells:
                torem = []
                for i in shell:
                    if atoms[i].symbol not in e:
                        torem.append(i)
                for i in torem:
                    shell.remove(i)

        i1 = random.randint(0, len(shells) - 1)
        i2 = random.randint(0, len(shells) - 1)
        mut_shell1 = shells[i1]

        while atoms[mut_shell1[0]].symbol == atoms[shells[i2][0]].symbol:
            i2 = random.randint(0, len(shells) - 1)
        mut_shell2 = shells[i2]
        atoms.symbols[mut_shell1+mut_shell2] = len(mut_shell1) * atoms[
        mut_shell2[0]].symbol + len(mut_shell2) * atoms[mut_shell1[0]].symbol


class Crossover(OffspringCreator):
    """Base class for all particle crossovers.
    Do not call this class directly."""
    def __init__(self):
        OffspringCreator.__init__(self)
        self.descriptor = 'Crossover'
        self.min_inputs = 2


class SymmetricCrossover(Crossover):
    """Merge the elemental distributions in two half shells from 
    different particles together.

    Parameters
    ----------
    shells : list of lists
        The atom indices in each shell divided by symmetry. Can be 
        obtained by `acat.build.SymmetricOrderingGenerator`.

    elements : list of strs, default None
        The metal elements of the nanoalloy. Only take into account 
        the elements specified in this list. Default is to take all 
        elements into account.

    """
    def __init__(self, shells, elements=None):
        Crossover.__init__(self)
        self.shells = shells
        self.elements = elements
        self.descriptor = 'SymmetricCrossover'
        
    def get_new_individual(self, parents):
        f, m = parents
        
        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [i.info['confid'] for i in parents] 
        indi.info['data']['operation'] = 'crossover'
        parent_message = ':Parents {0} {1}'.format(f.info['confid'],
                                                   m.info['confid']) 
        shells = self.shells.copy()
        if self.elements is None:
            e = list(set(atoms.get_chemical_symbols()))
        else:
            e = self.elements

        sorted_elems = sorted(set(atoms.get_chemical_symbols()))
        if e is not None and sorted(e) != sorted_elems:
            for shell in shells:
                torem = []
                for i in shell:
                    if atoms[i].symbol not in e:
                        torem.append(i)
                for i in torem:
                    shell.remove(i)

        random.shuffle(shells)
        fshells, mshells = shells[:len(shells)//2], shells[len(shells)//2:]
        flat_ids = [i for shell in shells for i in shell]
        indi.symbols[flat_ids] = str(f.symbols[fshells]) + str(
                                     m.symbols[mshells])

        return (self.finalize_individual(indi),
                self.descriptor + parent_message)
