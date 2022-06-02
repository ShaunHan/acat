from acat.build.ordering import RandomOrderingGenerator as ROG
from ase.build import bulk
from ase.io import read
from ase.visualize import view

# Initialize a pure Ni bulk structure
atoms = bulk('Ni', 'fcc')
atoms *= (6, 6, 6)

# Generate 50 NiPtAu bulk alloys with random chemical 
# orderings and save in a trajectory file 'orderings_1.traj'
rog = ROG(atoms, elements=['Ni','Pt','Au'], 
          trajectory='orderings_1.traj')
rog.run(num_gen=50)

# Visualize the output structures 
images1 = read('orderings_1.traj', index=':')
view(images1)

# Now generate 50 unique NiPtAu bulk alloys with the composition 
# of 1:1:1 and save in a trajectory file 'orderings_2.traj'
rog = ROG(atoms, elements=['Ni','Pt','Au'], 
          composition={'Ni': 1,'Pt': 1,'Au': 1},          
          trajectory='orderings_2.traj')
rog.run(num_gen=50)

# Visualize the output structures
images2 = read('orderings_2.traj', index=':')
view(images2)

