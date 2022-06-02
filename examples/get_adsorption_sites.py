from acat.adsorption_sites import SlabAdsorptionSites
from ase.build import fcc211

# Initialize a pure Cu(211) slab
atoms = fcc211('Cu', (3, 3, 4), vacuum=5.)

# Generate a CuAu(211) slab by changing some Cu to Au 
for atom in atoms:
    if atom.index % 2 == 0:
        atom.symbol = 'Au'

# Identify all adsorption sites on the alloy slab
sas = SlabAdsorptionSites(atoms, surface='fcc211',
                          composition_effect=True,
                          label_sites=True,
                          surrogate_metal='Cu')

# Print each site
sites = sas.get_sites()
for i, site in enumerate(sites):
    print('Site {0}: {1}'.format(i, site))

# Print each symmetry-inequivalent site
unique_sites = sas.get_unique_sites()
for i, unique_site in enumerate(unique_sites):
    print('Symmetry-inequivalent site {0}: {1}'.format(i, unique_site))
