"""Comparators meant to be used with symmetric particles"""
from ase.ga.standard_comparators import Comparator


class ShellCompositionComparator(Comparator):
    """Compares the elemental compositions of all shells defined 
    by the symmetry of the particle. Returns True if the numbers 
    are the same, False otherwise.

    Parameters
    ----------
    shells : list of lists
        The atom indices in each shell divided by symmetry. Can be 
        obtained by `acat.build.orderings.SymmetricOrderingGenerator`.

    elements : list of strs, default None
        The metal elements of the nanoalloy. Only take into account 
        the elements specified in this list. Default is to take all 
        elements into account.

    """

    def __init__(self, shells, elements=None):
        self.shells = shells
        self.elements = elements

    def looks_like(self, a1, a2):
        """ Return if structure a1 or a2 are similar or not. """

        elements = self.elements
        if self.elements is None:
            e = list(set(a1.get_chemical_symbols()) |
                     set(a2.get_chemical_symbols()))
        else:
            e = self.elements

        shells = self.shells.copy()
        sorted_elems = sorted(set(a1.get_chemical_symbols()) |
                              set(a2.get_chemical_symbols()))
        if e is not None and sorted(e) != sorted_elems:
            for shell in shells:
                torem = []
                for i in shell:
                    if atoms[i].symbol not in e:
                        torem.append(i)
                for i in torem:
                    shell.remove(i)

        comp1 = ''.join([a1[s[0]].symbol for s in shells])
        comp2 = ''.join([a2[s[0]].symbol for s in shells])

        return comp1 == comp2
