# get number of jobs:
MAKE_PID := $(shell echo $$PPID)
JOB_FLAG := $(filter -j%, $(subst -j ,-j,$(shell ps T | grep "^\s*$(MAKE_PID).*$(MAKE)")))
JOBS     := $(subst -j,,$(JOB_FLAG))
ifeq ($(JOBS),)
	JOBS = 8 # default number of jobs, I assume we have at least 4 cores
endif

HXX=hython -j $(JOBS) run.py 

usage:
	@echo "Please chose one of the following targets:"
	@echo "   collision_houdini"
	@echo "   collision_stokes"
	@echo "   collision"
	@echo "   coil_houdini"
	@echo "   coil_pvp"
	@echo "   coil_vp"
	@echo "   coil_stokes"
	@echo "   conveyor_belt"
	@echo "   viscous_coil"
	@echo "   viscous_coil_stokes"
	@echo "   viscous_coils"
	@echo "   dam_break_houdini"
	@echo "   dam_break_stokes"
	@echo "   dam_break"


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
