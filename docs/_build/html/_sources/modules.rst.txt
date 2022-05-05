Base modules
============

Adsorption sites
----------------

All symmetry-inequivalent adsorption sites supported by ACAT can be found in :download:`Table of Adsorption Sites <../table_of_adsorption_sites.pdf>`. The table includes snapshots of each site and the corresponding numerical labels irrespective of composition (`Label 1`) or considering composition effect (`Label 2`) for monometallics and bimetallics.

The ClusterAdsorptionSites class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: acat.adsorption_sites.ClusterAdsorptionSites 
       :members:
       :undoc-members:
       :show-inheritance:
       :exclude-members: get_labels, new_site, get_two_vectors, is_eq, get_angle, make_fullCNA, get_site_dict, set_first_neighbor_distance_from_rdf, get_surface_designation, make_neighbor_list

The group_sites_by_facet function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: acat.adsorption_sites.group_sites_by_facet

The SlabAdsorptionSites class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: acat.adsorption_sites.SlabAdsorptionSites
       :members:
       :undoc-members:
       :show-inheritance:
       :exclude-members: get_labels, new_site, get_two_vectors, is_eq, get_angle, make_fullCNA, get_site_dict, set_first_neighbor_distance_from_rdf, get_surface_designation, make_neighbor_list

The get_adsorption_site function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: acat.adsorption_sites.get_adsorption_site

The enumerate_adsorption_sites function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: acat.adsorption_sites.enumerate_adsorption_sites

Adsorbate coverage
------------------

This class is implemented to detect which adsorbate species are bound to which adsorption sites from a given surface-adsorbate configuration. It is recommended to use this class with a ClusterAdsorptionSites or SlabAdsorptionSites object as input.

The ClusterAdsorbateCoverage class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: acat.adsorbate_coverage.ClusterAdsorbateCoverage
       :members:
       :undoc-members:
       :show-inheritance:
       :exclude-members: identify_adsorbates, make_ads_neighbor_list

The SlabAdsorbateCoverage class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: acat.adsorbate_coverage.SlabAdsorbateCoverage
       :members:
       :undoc-members:
       :show-inheritance:
       :exclude-members: identify_adsorbates, make_ads_neighbor_list

The enumerate_occupied_sites function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: acat.adsorbate_coverage.enumerate_occupied_sites
