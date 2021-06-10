# ACAT: **A**lloy **C**atalysis **A**utomated **T**oolkit
ACAT is a Python package for atomistic modelling of metal / alloy catalysts used in heterogeneous catalysis. The package is based on automatic identifications of adsorption sites and adsorbate coverages on surface slabs and nanoparticles. Synergized with ASE, ACAT provides useful tools for building atomistic models of alloy catalysts with and without adsorbates. The goal is to automate workflows so that the structure generations can be readily adapted in a high-throughput screening.

ACAT has been developed by Shuang Han at the Section of Atomic Scale Materials Modelling, Department of Energy Conversion and Storage, Technical University of Denmark (DTU) in Lyngby, Denmark.

To use ACAT, please read **[ACAT documentation](https://asm-dtu.gitlab.io/acat)**.

![](images/color_facets.png)

## Developers: 
Shuang Han (shuha@dtu.dk) - current maintainer

## Dependencies
* python>=3.6
* networkx>=2.4
* ase
* asap3

## Installation
Install via pip:

```pip3 install acat```

Clone the repository:

```git clone https://gitlab.com/asm-dtu/acat.git```

Go to the installed path and install all dependencies:

```pip3 install -r requirements.txt```

Install the main package:

```python3 setup.py install```
 
## Acknowledgements

I would like to highly appreciate the support from BIKE project, where fundings are received from the European Union’s Horizon 2020 Research and Innovation programme under the Marie Skłodowska-Curie Action – International Training Network (MSCA-ITN), grant agreement 813748.

<img src="images/eu_logo.png" width="250"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 
<img src="images/bike_logo.png" width="250">

I also want to thank Dr. Steen Lysgaard for the useful scripts and Dr. Giovanni Barcaro, Dr. Alessandro Fortunelli for the useful discussions.

## Known issues

1. **ACAT currently only support metal / alloy surface slabs and nanoparticles.** Therefore H, C, N, O, F, S and Cl atoms are treated as adsorbate molecules and metals are treated as catalyst by default. Please do not use ACAT for metal oxide catalysts.

2. Some functions distinguishes between nanoparticle and surface slabs based on periodic boundary condition (PBC). Therefore, before using the code, it is recommended (but not required) to **set all directions as non-periodic for nanoparticles and at least one direction periodic for surface slabs, and also add vacuum layers to all non-periodic directions. For periodic surface slabs the code will not work if the number of layers is less than 3 (which should be avoided anyways).** Each layer always has the same number of atoms as the surface atoms. For stepped surface slabs one layer will have atoms at different heights.

3. ACAT uses a regularized adsorbate string representation. In each adsorbate string, **the first element must set to the bonded atom. If the adsorbate is multi-dentate, the order follows the order of their atomic numbers. Hydrogen should always follow the element that it bonds to.** For example, water should be written as 'OH2', hydrogen peroxide should be written as 'OHOH', ethanol should be written as 'CH3CH2OH', formyl should be written as 'CHO', hydroxymethylidyne should be written as 'COH'. If the string is not supported by the code, it will return the ase.build.molecule instead, which could result in a weird orientation. If the string is not supported by this code nor ASE, you can make your own molecules in the adsorbate_molecule function in acat.settings.

4. There is a bug that causes ``get_neighbor_site_list()`` to not return the correct neighbor site indices with ASE version <= 3.18. This is most likely due to shuffling of indices in some ASE functions, which is solved after the release of ASE 3.19.0. 
