Base modules
============

Adsorption sites
----------------

All symmetry-inequivalent adsorption sites supported by ACAT can be found in :download:`Table of Adsorption Sites <../table_of_adsorption_sites.pdf>`. The table includes snapshots of each site and the corresponding numerical labels irrespective of composition (`Label 1`) or considering composition effect (`Label 2`).

.. automodule:: acat.adsorption_sites
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: get_labels, new_site, get_two_vectors, is_eq, get_angle, make_fullCNA, get_site_dict, set_first_neighbor_distance_from_rdf, get_surface_designation, make_neighbor_list

ClusterAdsorptionSites class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: ClusterAdsorptionSites 

group_sites_by_facet function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: group_sites_by_facet

SlabAdsorptionSites class
~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: SlabAdsorptionSites

get_adsorption_site function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: get_adsorption_site

enumerate_adsorption_sites function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: enumerate_adsorption_sites

Adsorbate coverage
------------------

.. automodule:: acat.adsorbate_coverage
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: identify_adsorbates, make_ads_neighbor_list  

ClusterAdsorbateCoverage class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: ClusterAdsorbateCoverage

SlabAdsorbateCoverage class
~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: SlabAdsorbateCoverage

enumerate_occupied_sites function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: enumerate_occupied_sites
