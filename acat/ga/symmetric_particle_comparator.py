"""Comparators meant to be used with symmetric particles"""


class ShellCompositionComparator(object):
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

    tol : int, default 0
        The maximum number of shells with different elements that two 
        structures are still considered to be look alike.

    """

    def __init__(self, shells, elements=None, tol=0):
        self.shells = shells
        self.elements = elements
        self.tol = tol

    def looks_like(self, a1, a2):
        """ Return if structure a1 or a2 are similar or not. """

        elements = self.elements
        if self.elements is None:
            e = list(set(a1.get_chemical_symbols()))
        else:
            e = self.elements

        shells = self.shells.copy()
        sorted_elems = sorted(set(a1.get_chemical_symbols()))
        if e is not None and sorted(e) != sorted_elems:
            for shell in shells:
                torem = []
                for i in shell:
                    if a1[i].symbol not in e:
                        torem.append(i)
                for i in torem:
                    shell.remove(i)

        diff = [s for s in shells if a1[s[0]].symbol != a2[s[0]].symbol]

        return len(diff) <= self.tol