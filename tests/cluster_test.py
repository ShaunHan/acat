from acat.adsorption_sites import ClusterAdsorptionSites
from ase.cluster import Octahedron

atoms = Octahedron('Ni', length=7, cutoff=2)
for atom in atoms:
    if atom.index % 2 == 0:
        atom.symbol = 'Pt' 
atoms.center(vacuum=5.)
cas = ClusterAdsorptionSites(atoms, allow_6fold=False,
                             composition_effect=True,
                             label_sites=True,
                             surrogate_metal='Ni')
site = cas.get_sites()
assert len(site) == 674

