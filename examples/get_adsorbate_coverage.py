from acat.adsorption_sites import ClusterAdsorptionSites
from acat.adsorbate_coverage import ClusterAdsorbateCoverage
from acat.build import add_adsorbate_to_site
from ase.cluster import Octahedron
from ase.visualize import view

# Initialize a pure Ni truncated octahedral nanoparticle
atoms = Octahedron('Ni', length=7, cutoff=2)
atoms.center(vacuum=5.)

# Generate a NiPt nanoalloy by changing some Ni to Pt
for atom in atoms:
    if atom.index % 2 == 0:
        atom.symbol = 'Pt'

# Identify all adsorption sites on the nanoalloy
cas = ClusterAdsorptionSites(atoms, composition_effect=True,
                             surrogate_metal='Ni') 

# Generate a Ni-Pt nanoalloy covered by CO at all fcc and 4fold sites
# (as an example of the input adsorbate-alloy structure one would provide)
sites = cas.get_sites()
for site in sites:
    if site['site'] in ['fcc', '4fold']:
        add_adsorbate_to_site(atoms, adsorbate='CO', site=site)

# Visualize the structure
view(atoms)

# Get information about the surface adsorbate coverage
cac = ClusterAdsorbateCoverage(atoms, adsorption_sites=cas,
                               label_occupied_sites=True)

# Print all occupied sites
occupied_sites = cac.get_sites(occupied_only=True)
for i, site in enumerate(occupied_sites):
    print('Occupied site {0}: {1}'.format(i, site))

# Calculate the surface coverage (ML)
print('Surface coverage: {0}'.format(cac.get_coverage()))

# Get information of all adsorbates
adsorbates_info = cac.get_adsorbates()
for i, ads_info in enumerate(adsorbates_info):
    ads_species, ads_indices = ads_info
    print('Adsorbate {0}: {1} with atomic indices of {2}'.format(i, ads_species, ads_indices))

