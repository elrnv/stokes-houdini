# Stokes Variational MicroSolver Plugin DOP for Houdini

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

## TODO

- The pure Viscosity solve is currently broken. Need to fix that (low priority)
- Add surface tension
- Add more examples
