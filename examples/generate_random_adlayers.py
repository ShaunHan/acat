from acat.adsorption_sites import ClusterAdsorptionSites
from acat.build.adlayer import RandomPatternGenerator as RPG
from acat.build.ordering import RandomOrderingGenerator as ROG
from ase.cluster import Octahedron
from ase.io import read
from ase.visualize import view

# Initialize a pure Ni truncated octahedral nanoparticle 
atoms = Octahedron('Ni', length=5, cutoff=2)
atoms.center(vacuum=5.)

# Generate 50 random adlayer patterns on the nanoparticle with 2 
# of C, O, N adsorbates and a minimum adsorbate-adsorbate distance 
# of 3 Angstrom and save in a trajectory file 'patterns.traj'
rpg = RPG(atoms, adsorbate_species=['C','O','N'], 
          min_adsorbate_distance=3.)
rpg.run(num_gen=50, action='add', num_act=2)

# Visualize the output structures
images = read('patterns.traj', index=':')
view(images)

# If you want to generate random adlayer pattern at a certain coverage,
# try using acat.build.adlayer.max_dist_coverage_pattern. The generated 
# patterns are random but tend to maximize the adsorbate-adsorbate 
# distances so that the generated patterns are more reasonable
from acat.build.adlayer import max_dist_coverage_pattern as maxdcp

# Generate random 0.33 ML CO coverage pattern
pattern = maxdcp(atoms, adsorbate_species='CO', coverage=0.33)
view(pattern)

# Generate random 0.66 ML CO coverage pattern
pattern = maxdcp(atoms, adsorbate_species='CO', coverage=0.66)
view(pattern)

# Another strategy is to provide the minimum adsorbate-adsorbate distance
# and try to maximize the density of the adsorbates
from acat.build.adlayer import min_dist_coverage_pattern as mindcp

# Generate random CO coverage pattern with minimum distance of 2 Angstrom
pattern = mindcp(atoms, adsorbate_species='CO', min_adsorbate_distance=2.)
view(pattern)

# Generate random CO coverage pattern with minimum distance of 5 Angstrom
pattern = mindcp(atoms, adsorbate_species='CO', min_adsorbate_distance=5.)
view(pattern)

