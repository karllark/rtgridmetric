Grid Metrics for Dust Radiative Transfer
========================================

What makes a good dust radiative transfer grid?  That is the question.

In Development!
---------------

Active development.
Data still in changing.
Use at your own risk.

Contributors
------------
Karl Gordon

License
-------

This code is licensed under a 3-clause BSD style license (see the
``LICENSE`` file).

Workflow
--------

Generate a model with a regular single level grid.

Visualize the differences between cells in the x,y,z directions using
plot_radfield_vis.py

Generate an AMR grid based on a specific fractional threshold (default=0.05) using
refinegrid.py

Run dirty on AMR grid.

Collapse the AMR grid to a single, higher resolution grid using
regrid_uniform.py

Analyze the uniform grid using analyze_radfield.py.
Can also visualize the x,y grid slices using the higher res grid and
the x, z grid slices using the transpose of the higher res grid.
