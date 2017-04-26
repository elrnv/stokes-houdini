# To build and install the stokes DOP, run
#			$ make install
# To simply build the library, run 
#			$ make

#CXX = clang-omp

ifndef EIGEN_INCLUDE_DIR
	EIGEN_INCLUDE_DIR = /usr/local/include/eigen3
endif

ifndef DSO_DIR
	DSO_DIR = .
endif

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	DSONAME = $(DSO_DIR)/libStokes.so
	LIBS = -fopenmp
else ifeq ($(UNAME_S),Darwin)
	DSONAME = $(DSO_DIR)/libStokes.dylib
	LIBS = -framework OpenCL
endif

INCDIRS = -I$(EIGEN_INCLUDE_DIR)
SOURCES = main.C SIM/SIM_Stokes.C

#OPTIMIZER = -g
OPTIMIZER = -O3 -DNDEBUG

###############################################################################
# For GNU make, use this line:
#      include $(HFS)/toolkit/makefiles/Makefile.gnu
#
include $(HFS)/toolkit/makefiles/Makefile.gnu
