# To build and install the stokes DOP, run
#			$ make install
# To simply build the library, run 
#			$ make

#CXX = clang-omp

include EigenPathMakefile # sets the EIGEN_INCLUDE_PATH variable

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	DSONAME = libStokes.so
	INCDIRS = -fopenmp
else ifeq ($(UNAME_S),Darwin)
	DSONAME = libStokes.dylib
	LIBS = -framework OpenCL
endif

INCDIRS += -I$(EIGEN_INCLUDE_PATH)

SOURCES = main.C SIM/SIM_Stokes.C

#OPTIMIZER = -g
OPTIMIZER = -O3 -DNDEBUG

###############################################################################
# For GNU make, use this line:
#      include $(HFS)/toolkit/makefiles/Makefile.gnu
#
include $(HFS)/toolkit/makefiles/Makefile.gnu

