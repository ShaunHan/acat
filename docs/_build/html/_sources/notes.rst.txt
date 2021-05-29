Notes
=====

1. **ACAT currently only support metal / alloy surface slabs and nanoparticles.** Therefore H, C, N, O, F, S and Cl atoms are treated as adsorbate molecules and metals are treated as catalyst by default. Please do not use ACAT for metal oxide catalysts.

2. Some functions distinguishes between nanoparticle and surface slabs based on periodic boundary condition (PBC). Therefore, before using the code, it is recommended (but not required) to **set all directions as non-periodic for nanoparticles and at least one direction periodic for surface slabs, and also add vacuum layers to all non-periodic directions. For periodic surface slabs the code will not work if the number of layers is less than 3 (which should be avoided anyways).** Each layer always has the same number of atoms as the surface atoms. For stepped surface slabs one layer will have atoms at different heights.

3. ACAT uses a regularized adsorbate string representation. In each adsorbate string, **the first element must set to the bonded atom. If the adsorbate is multi-dentate, the order follows the order of their atomic numbers. Hydrogen should always follow the element that it bonds to.** For example, water should be written as 'OH2', hydrogen peroxide should be written as 'OHOH', ethanol should be written as 'CH3CH2OH', formyl should be written as 'CHO', hydroxymethylidyne should be written as 'COH'. If the string is not supported by the code, it will return the ase.build.molecule instead, which could result in a weird orientation. If the string is not supported by this code nor ASE, you can make your own molecules in the adsorbate_molecule function in acat.settings.

4. There is a bug that causes ``get_neighbor_site_list()`` to not return the correct neighbor site indices with ASE version <= 3.18. This is most likely due to shuffling of indices in some ASE functions, which is solved after the release of ASE 3.19.0. 
