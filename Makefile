# To build and install the stokes DOP, run
#			$ make install
# To simply build the library, run 
#			$ make

#CXX = clang-omp

HOSTNAME := $(shell hostname | sed 's/.cs//')
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	DSONAME = libStokes.so
	INCDIRS = -fopenmp -I./solver/
	ifeq ($(HOSTNAME),poisson)
		INCDIRS += -I/home/elariono/local/include/eigen3 
	else
		INCDIRS += -I/home/elarionov/proj/eigen
	endif
else ifeq ($(UNAME_S),Darwin)
	DSONAME = libStokes.dylib
	INCDIRS = -I/Users/egor/proj/Eigen -I./solver/
	LIBS = -framework OpenCL
endif

SOP_SOURCES = SOP/SOP_Stokes.C \
							SOP/resample_vdb.C \
							SOP/advect_vdb.C
SOLVER_SOURCES = solver/stokes3d.cpp
SOURCES = main.C SIM/SIM_Stokes.C
#$(SOP_SOURCES) 
#OPTIMIZER = -g
OPTIMIZER = -O3 -DNDEBUG

###############################################################################
# For GNU make, use this line:
#      include $(HFS)/toolkit/makefiles/Makefile.gnu
#
include $(HFS)/toolkit/makefiles/Makefile.gnu

