from acat.adsorption_sites import SlabAdsorptionSites
from acat.adsorbate_coverage import SlabAdsorbateCoverage
from acat.build.action import add_adsorbate
from ase.build import fcc211

atoms = fcc211('Cu', (3, 3, 4), vacuum=5.)
for atom in atoms:
    if atom.index % 2 == 0:
        atom.symbol = 'Au'
atoms.center()
add_adsorbate(atoms, adsorbate='CH3OH', surface='fcc211',
              indices=(5, 7, 8), surrogate_metal='Cu')
sac = SlabAdsorbateCoverage(atoms, surface='fcc211', 
                            label_occupied_sites=True)
occupied_sites = sac.get_sites(occupied_only=True)
assert len(occupied_sites) == 2

