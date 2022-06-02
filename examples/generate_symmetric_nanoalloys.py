from acat.build.ordering import SymmetricClusterOrderingGenerator as SCOG
from ase.cluster import Octahedron
from ase.io import read
from ase.visualize import view

# Initialize a pure Ni truncated octahedral nanoparticle 
# as the template
atoms = Octahedron('Ni', 7, 2)

# Generate 50 symmetric nanoalloys with 3 metals with 
# "mirror circular" symmetry and save in a trajectory 
# file 'orderings_1.traj'
scog = SCOG(atoms, elements=['Ni', 'Cu', 'Pt'], 
            symmetry='mirror_circular',
            trajectory='orderings_1.traj')
scog.run(max_gen=50, mode='stochastic')

# Visualize the output structures
images1 = read('orderings_1.traj', index=':')
view(images1)

# Now generate 50 symmetric NiCuPt nanoalloys with 
# the composition of 1:1:1 and save in a trajectory 
# file 'orderings_2.traj'
scog = SCOG(atoms, elements=['Ni','Cu','Pt'], 
            symmetry='mirror_circular',
            composition={'Ni': 1,'Cu': 1,'Pt': 1}, 
            trajectory='orderings_2.traj')
scog.run(max_gen=50, mode='stochastic', eps=0.01)

# Visualize the output structures. 
images2 = read('orderings_2.traj', index=':')
view(images2)

