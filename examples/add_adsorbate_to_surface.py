from acat.build import add_adsorbate
from ase.build import fcc211
from ase.visualize import view

# Initialize a pure Cu(211) slab                             
atoms = fcc211('Cu', (3, 3, 4), vacuum=5.)

# Generate a CuAu(211) slab by changing some Cu to Au 
for atom in atoms:
    if atom.index % 2 == 0:
        atom.symbol = 'Au'

# Add a CO adsorbate to one site that is on the step bridge site
# with a composition of 'AuCu' and a height of 1.8 Angstrom
# (Always remember to pass in the surrogate metal for robust
# asorption site identification)
add_adsorbate(atoms, adsorbate='CO', site='bridge', surface='fcc211',
              morphology='step', height=1.8, surrogate_metal='Cu')

# Visualize the structure
view(atoms)

# Add another CO adsorbate to the site with atomic indices of (5,7,8)
add_adsorbate(atoms, adsorbate='CO', surface='fcc211', indices=(5,7,8), 
              height=1.8, surrogate_metal='Cu')

# Visualize the structure
view(atoms)

