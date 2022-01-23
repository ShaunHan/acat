Base modules
============

Adsorption sites
----------------

All symmetry-inequivalent adsorption sites supported by ACAT can be found in :download:`Table of Adsorption Sites <../table_of_adsorption_sites.pdf>`. The table includes snapshots of each site and the corresponding numerical labels irrespective of composition (`Label 1`) or considering composition effect (`Label 2`) for monometallics and bimetallics.

.. automodule:: acat.adsorption_sites
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: get_labels, new_site, get_two_vectors, is_eq, get_angle, make_fullCNA, get_site_dict, set_first_neighbor_distance_from_rdf, get_surface_designation, make_neighbor_list

The ClusterAdsorptionSites class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: ClusterAdsorptionSites 

The group_sites_by_facet function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: group_sites_by_facet

The SlabAdsorptionSites class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: SlabAdsorptionSites

The get_adsorption_site function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: get_adsorption_site

The enumerate_adsorption_sites function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: enumerate_adsorption_sites

Adsorbate coverage
------------------

.. automodule:: acat.adsorbate_coverage
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: identify_adsorbates, make_ads_neighbor_list  

The ClusterAdsorbateCoverage class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: ClusterAdsorbateCoverage

The SlabAdsorbateCoverage class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autoclass:: SlabAdsorbateCoverage

The enumerate_occupied_sites function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. autofunction:: enumerate_occupied_sites
