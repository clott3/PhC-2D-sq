# Predicting Bandstructures of 2D Square Photonic Crystals

## Introduction
This repo hosts the code used in [https://doi.org/10.1515/nanoph-2020-0197] where a simple neural network with CNN and fully connected layers was used to predict the bandstructures of 2D square lattice Photonic Crystals.

## Dataset
The datasets can be downloaded from this [link](https://www.dropbox.com/sh/uezvuoia7fe2w7e/AAAAAJfesWnJWyeU1K1xLwKVa?dl=0). A description of the datasets are provided here (also check the /metadata/ tag inside the .h5 files):

This .hdf5 file contains MPB calculations of square photonic crystals of either TE or TM polarization, as indicated in /metadata/ attribute 'runtype'. The computational resolution is indicated in /metadata/ attribute 'resolution'. Besides the /metadata/ group, this .hdf5 file contains a /shapes/ group. This group in turn contains 20 000 numerically labelled groups (1, 2, etc) which give input and results for distinct unitcell calculations. Each such numeric subgroup (e.g. /shapes/1/) contain several subfields:

- kvecs    : a flatted 23x23 grid of 2-dimensional k-vectors,
                      in units of reciprocal lattice vectors (i.e. k*2*pi/a)
- eigfreqs : eigenfrequencies at the corresponding kvecs,
                      in units of c/a (i.e. omega*a/c). Includes first 6 bands
- bandgap_1 : a 2-element array with elements:
  - 1st element: bandgap size between 1st and 2nd band, in
                           units of c/a. If negative, value is NaN
  - 2nd element: gap-to-midgap ratio (i.e. relative gap size),
                            in dimensionless units (i.e. a fraction)
- unitcell  : a group with subfields that describe the unitcell:
  - epsilon   : the dielectric function on the unit cell, as input to
                            MPB, in a 256x256 grid
  - epsilon_comput : the downsampled version of epsilon used internally
                           in MPB. Has size equal to /metadata/resolution/
  - epsilon_average : the average of /unitcell/epsilon/, for convenience
  - boundary  : [N,2] array giving the boundary between inside/outside
                            materials in the unitcell. It is assumed that the unit-
                           cell lies in the [-0.5, 0.5]^2 box (i.e. lengths are
                            given in units of lattice constants 'a')
  - epsin     : Value of dielectric function inside inclusion
  - epsout    : Value of dielectric function outside inclusion
  - areain    : Relative area of inclusion (to entire unitcell area)

## Usage
Upon cloning, the code can be run directly using ```run.py``` where the path to the directory containing the dataset should be specified using the ```--path_to_h5``` flag. The default hyperparameters defined in the parser will give the optimal results reported in the paper.  
