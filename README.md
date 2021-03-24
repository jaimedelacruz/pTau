# pTau
Computation in parallel of simple H background opacities and an optical depth-scale from temperature, z-scale and [Pgas or rho].

## Compilation of the C++ module
These routines require a C++-14 compiler with support for OpenMP: g++, clan++ or icpc should work

To compile it simply use:
```
python3 setup.py build_ext --inplace
```
And then copy pTau.cpython*.so and pyTau.py to your PYTHONPATH folder or
to the folder where you want to execute these routines if you have not defined a python library folder.

Look into the example folder for a commented example.