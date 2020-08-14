# Nanopads: NANOParticle with ADSorbate
A Python code for generating and encoding adsorbate coverage patterns on surfaces and nanoparticles.

## Developers: 
Shuang Han (shuha@dtu.dk) - current maintainer

Steen Lysgaard (stly@dtu.dk)

## Dependencies
* Python3
* Numpy
* NetworkX
* ASE
* Pymatgen
* Asap3

## Installation
Clone this repository:

```git clone https://gitlab.com/shuanghan/nanopads.git```

Then edit ```~/.bashrc``` add
```export PYTHONPATH=~/path-to-nanopads/:$PYTHONPATH```
## Usage
### Add adsorbates
The code can automatically identify the shape and surfaces of nanoparticles, or the type of surface slabs.

![](images/color_facets.png)

To add adsorbate to a monometallic system (or if you want to ignore the elemental composition), see example:
```python
from nanopads.adsorption_sites import monometallic_add_adsorbate
from ase.io import read, write
from ase.visualize import view

atoms = read('random_NiPt_111_surface.traj')
system = monometallic_add_adsorbate(atoms, adsorbate='OH', site='ontop', nsite=5)
system = monometallic_add_adsorbate(system, adsorbate='OH', site='bridge', nsite=6)
system = monometallic_add_adsorbate(system, adsorbate='OH', site='fcc', nsite=7)
system = monometallic_add_adsorbate(system, adsorbate='OH', site='hcp', nsite=8)
view(system)
```
Out:

<img src="images/random_NiPt_111_surface_with_OH.png"  width="400" height="250">

To add adsorbate to a bimetallic system, see example:
```python
from nanopads.adsorption_sites import bimetallic_add_adsorbate
from ase.io import read, write
from ase.visualize import view

atoms = read('random_icosahedron_NiPt_309.traj')
system = bimetallic_add_adsorbate(atoms, adsorbate='OH', site='hcp', 
surface='fcc111', composition='NiNiPt', second_shell='Ni', nsite='all')
view(system)
```
Out:

<img src="images/random_icosahedron_NiPt_309_with_OH.png"  width="300" height="300">

### Enumerate sites 
To enumerate all possible adsorption of a nanoparticle or surface slab, see example:
```python
from nanopads.adsorption_sites import enumerate_monometallic_sites

all_sites = enumerate_monometallic_sites(atoms, second_shell=True)
```

### Label occupied sites
To get information of site occupation, see example:
```python
from nanopads.adsorption_sites import label_occupied_sites
from ase.io import read, write

atoms = read('cuboctahedron_NiPt_309_with_OH.traj')
labeled_atoms = label_occupied_sites(atoms, adsorbate='OH', second_shell=True)
```
![](images/tagged_sites.png)

If multiple species are present, please provide a list of the present adsorbates. Currently only support at most 2 species.
![](images/labeled_sites.png)

### Generate coverage pattern
A search algorithm is implemented to automatically generate adsorbate patterns with certain coverages. Example: to generate the adsorbate pattern on fcc111 surface with a 0.75 ML coverage, simply use the following code
```python
from nanopads.adsorbate_coverage import pattern_generator
from ase.io import read, write

atoms = read('random_surface_111.traj')
pattern = pattern_generator(atoms, adsorbate='O', coverage=3/4)
```
Out:

<img src="images/fcc111_0.75ml.png"  width="400" height="250">

The code can generate coverage patterns for various surfaces and nanoparticles. Below shows all well-defined patterns.
![](images/all_coverage_patterns.png)