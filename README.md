# Stokes Variational Micro-Solver Plugin DOP for Houdini


## Building

Before building anything, create a file called EigenPathMakefile and add a
single line setting the EIGEN_INCLUDE_PATH variable:
```
EIGEN_INCLUDE_PATH = <path to eigen>
```
to point to where you have Eigen installed.

SIM_Stokes.C and SIM_Stokes.h are the C++ files defining the Houdini plugin DOP.
In order to build them, source a houdini enviroment with

```
$ pushd /opt/hfs##.#.#/; source houdini_setup; popd
```

Then build with
```
$ make install
```

Note: instructions will differ for Windows


## Usage

To use the micro-solver, simply dive into the **FLIP Solver** DOP node and change
the type of the second **Gas Project Non Divergent Variational** operator named
`projectnondivergent` to the new **Stokes** node.  The parameters should stay
mostly the same as in the original node. Note that **Min Viscosity**
is the *dynamic* viscosity and NOT the kinematic viscosity
from the **Gas Viscosity** node. You may want to increase **Error Tolerance** by an
order of magnitude, and set copy the **Samples Per Axis** parameter from the
**Gas Viscosity** node. Finally remove the lingering spare parameters and
disconnect the remaining **Gas Project Non Divergent Variational** and **Gas
Viscosity** nodes named `projectnondivergent_viscosity` and `gasviscosity`
respectively. Once disconnected, you can reap the benefits of the **Slip On
Collision** feature on the **FLIP Solver** when viscosity is enabled. If viscosity
is disabled in the **FLIP Solver**, the resulting simulation will still exhibit
viscosity. If you want to bypass the full stokes solver without modifying the
**FLIP Solver** network further, simply select the **Pressure Only** **Scheme** on
the **Stokes** node to disable viscosity. Note that currently, only the **Stokes**
**Scheme** is fully optimized with OpenCL, other Schemes will use a more
expensive CPU based solver.


## TODO

- The pure Viscosity solve is currently broken. Need to fix that (low priority)
- Add surface tension
- Add more examples
