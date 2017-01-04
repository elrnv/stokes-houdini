# get number of jobs:
MAKE_PID := $(shell echo $$PPID)
JOB_FLAG := $(filter -j%, $(subst -j ,-j,$(shell ps T | grep "^\s*$(MAKE_PID).*$(MAKE)")))
JOBS     := $(subst -j,,$(JOB_FLAG))
UNAME_S  := $(shell uname -s)
ifeq ($(JOBS),)
	ifeq ($(UNAME_S),Linux)
		JOBS = $(shell nproc --all) # get the number of cpus for default number of jobs
	else ifeq ($(UNAME_S),Darwin)
		JOBS = $(shell sysctl -n hw.ncpu) # get the number of cpus for default number of jobs
	endif
endif

HXX=hython -j $(JOBS) run.py 

usage:
	@echo "Available jobs: " $(JOBS)
	@echo "Please chose one of the following targets:"
	@echo "   collision_houdini"
	@echo "   collision_stokes"
	@echo "   collision"
	@echo "   coil_houdini"
	@echo "   coil_pvp"
	@echo "   coil_vp"
	@echo "   coil_stokes"
	@echo "   conveyor_belt"
	@echo "   conveyor_belt_stokes"
	@echo "   viscous_coil_houdini"
	@echo "   viscous_coil_stokes"
	@echo "   viscous_coils"
	@echo "   dam_break_houdini"
	@echo "   dam_break_stokes"
	@echo "   dam_break"
	@echo "   buckling"
	@echo "   surface_tension_stokes"
	@echo "   surface_tension_decoupled"
	@echo "   surface_tension"


collision_houdini:
	$(HXX) collision.hip houdini_fluid/houdini_fluid
collision_stokes:
	$(HXX) collision.hip stokes_fluid/stokes_fluid

collision: collision_stokes collision_houdini

coil_houdini:
	$(HXX) coil_compare.hip houdini/houdini_fluid_half
coil_pvp:
	$(HXX) coil_compare.hip decoupled_pvp/decoupled_pvp_fluid
coil_vp:
	$(HXX) coil_compare.hip decoupled_vp/decoupled_vp_fluid
coil_stokes:
	$(HXX) coil_compare.hip stokes/viscous_fluid

conveyor_belt:
	$(HXX) conveyor_belt.hip dopnet1/output

conveyor_belt_stokes:
	$(HXX) conveyor_belt.hip stokes/stokes

viscous_coil_houdini:
	$(HXX) viscous_coil_houdini.hip fluid/viscous_fluid

viscous_coil_stokes:
	$(HXX) viscous_coil_stokes.hip fluid/viscous_fluid

viscous_coils: viscous_coil_houdini viscous_coil_stokes

dam_break_houdini:
	$(HXX) dam_break.hip houdini/houdini_fluid

dam_break_stokes:
	$(HXX) dam_break.hip stokes/stokes_fluid

dam_break: dam_break_stokes dam_break_houdini

buckling:
	$(HXX) buckling.hip dopnet1/output

surface_tension_stokes:
	$(HXX) surface_tension.hip stokes_fluid/stokes

surface_tension_decoupled:
	$(HXX) surface_tension.hip decoupled_fluid/decoupled

surface_tension_stokes_5:
	$(HXX) surface_tension_5.hip stokes_fluid/stokes

surface_tension_decoupled_5:
	$(HXX) surface_tension_5.hip decoupled_fluid/decoupled

surface_tension_stokes_10-20:
	$(HXX) surface_tension_10-20.hip stokes_fluid/stokes

surface_tension_decoupled_10-20:
	$(HXX) surface_tension_10-20.hip decoupled_fluid/decoupled

surface_tension_stokes_50-20:
	$(HXX) surface_tension_50-20.hip stokes_fluid/stokes

surface_tension_decoupled_50-20:
	$(HXX) surface_tension_50-20.hip decoupled_fluid/decoupled

surface_tension: surface_tension_stokes surface_tension_decoupled
surface_tension_5: surface_tension_stokes_5 surface_tension_decoupled_5
surface_tension_10-20: surface_tension_stokes_10-20 surface_tension_decoupled_10-20
surface_tension_50-20: surface_tension_stokes_50-20 surface_tension_decoupled_50-20
surface_tension_all: surface_tension surface_tension_5 surface_tension_10-20 surface_tension_50-20
