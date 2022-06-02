from acat.build.adlayer import OrderedPatternGenerator as OPG
from acat.adsorption_sites import SlabAdsorptionSites
from ase.io import read
from ase.build import fcc111
from ase.visualize import view

# Initialize a pure Ni(111) slab
atoms = fcc111('Ni', (4, 4, 4), vacuum=5.)
for atom in atoms:
    if atom.index % 2 == 0:
        atom.symbol = 'Cu'

# Identify the surface adsorption sites (ignoring bridge sites)
sas = SlabAdsorptionSites(atoms, surface='fcc111',
                          ignore_bridge_sites=True,
                          surrogate_metal='Ni')

# Generate 50 unique adsorbate-alloy configurations with ordered
# adlayers, each consisting of at most 2 of the C, N, O adsorbates,
# and save in a trajectory file 'patterns_1.traj'
opg = OPG(atoms, adsorbate_species=['C', 'N', 'O'], surface='fcc111',
          max_species=2, adsorption_sites=sas, trajectory='patterns_1.traj')
opg.run(max_gen=50, unique=True)

# Visualize the output structures
images1 = read('patterns_1.traj', index=':')
view(images1)

# If you think the patterns are not 'ordered' enough, try providing
# a specific repeating distance, for example 5.026 Angstrom
opg = OPG(atoms, adsorbate_species=['C', 'N', 'O'], surface='fcc111',
          repeating_distance=5.026, max_species=2, adsorption_sites=sas, 
          trajectory='patterns_2.traj')
opg.run(max_gen=50, unique=True)

# Visualize the output structures
images2 = read('patterns_2.traj', index=':')
view(images2)

# If you want to generate very ordered well-known adlayer patterns,
# try acat.build.adlyer.special_coverage_pattern
from acat.build.adlayer import special_coverage_pattern as scp

# Generate the ordered adlayer pattern at 0.25 ML CO coverage
pattern = scp(atoms, adsorbate_species='CO', coverage=0.25, surface='fcc111')
view(pattern)

# Generate the ordered adlayer pattern at 0.5 ML CO coverage
pattern = scp(atoms, adsorbate_species='CO', coverage=0.5, surface='fcc111')
view(pattern)

# Generate the ordered adlayer pattern at 0.75 ML CO coverage
pattern = scp(atoms, adsorbate_species='CO', coverage=0.75, surface='fcc111')
view(pattern)

# Generate the ordered adlayer pattern at 1 ML CO coverage
pattern = scp(atoms, adsorbate_species='CO', coverage=1., surface='fcc111')
view(pattern)

