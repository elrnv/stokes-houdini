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
	@echo "To reproduce the simulations in our paper, please choose one of the following targets:"
	@echo "   rope_coiling_25                   (rope coling example with moderate viscosity (stokes vs. decoupled))"
	@echo "   rope_coiling_100                  (rope coling example with high viscosity (stokes vs. decoupled))"
	@echo "   rope_coiling_molasses             (rope coling with surface tension)"
	@echo "   rope_coiling_conveyor_belt        (rope coling on a moving conveyor belt)"
	@echo "   piling_armadillos                 (viscous armadillos falling into a pile)"
	@echo "   colliding_characters              (an armadillo and a stanford bunny colliding in zero gravity)"
	@echo "   ball_through_goop                 (a solid ball emerging from a box of viscous goop)"
	@echo "   collapsing_liquid_squab           (the Houdini Squab as a viscous liquid with surface tension in zero gravity)"
	@echo "   variable_viscosity_block          (a block of viscous liquid falling into a rectangular box)"
	@echo "   caramel_on_wafer                  (caramel buckling on top of a wafer)"
	@echo ""
	@echo "To render a corresponding animation with Mantra, please choose one of the following targets"
	@echo "   render_rope_coiling_25"
	@echo "   render_rope_coiling_100"
	@echo "   render_rope_coiling_molasses"
	@echo "   render_rope_coiling_conveyor_belt"
	@echo "   render_piling_armadillos"
	@echo "   render_colliding_characters"
	@echo "   render_ball_through_goop"
	@echo "   render_collapsing_liquid_squab"
	@echo "   render_variable_viscosity_block"
	@echo "   render_caramel_on_wafer"
	@echo ""
	@echo "Targets for individual simulations can be found below:"
	@echo "   collision"
	@echo "   coil_houdini"
	@echo "   coil_pvp"
	@echo "   coil_vp"
	@echo "   coil_stokes"
	@echo "   viscous_coil_houdini"
	@echo "   viscous_coil_stokes"
	@echo "   viscous_coils"
	@echo "   dam_break_houdini"
	@echo "   dam_break_stokes"
	@echo "   dam_break"
	@echo "   surface_tension_stokes"
	@echo "   surface_tension_decoupled"
	@echo "   surface_tension"
	@echo "   rotating_ball"
	@echo "   ball"

# Simulation targets
rope_coiling_25: viscous_coil_stokes_25 viscous_coil_houdini_25
rope_coiling_100: viscous_coil_stokes_100 viscous_coil_houdini_100
rope_coiling_molasses:
rope_coiling_conveyor_belt:
	$(HXX) conveyor_belt.hip /obj/stokes/stokes
piling_armadillos: armadillos_stokes armadillos_houdini
colliding_characters:
ball_through_goop:
collapsing_liquid_squab:
variable_viscosity_block:
	$(HXX) variable_viscosity.hip /obj/houdini_fluid/houdini_fluid
caramel_on_wafer:
	$(HXX) buckling.hip /obj/dopnet1/output


collision_houdini:
	$(HXX) collision.hip /obj/houdini_fluid/houdini_fluid
collision_stokes:
	$(HXX) collision.hip /obj/stokes_fluid/stokes_fluid

collision: collision_stokes collision_houdini

armadillos_stokes:
	$(HXX) armadillos.hip /obj/stokes/surface_cache
armadillos_houdini:
	$(HXX) armadillos.hip /obj/houdini/surface_cache

coil_houdini:
	$(HXX) coil_compare.hip /obj/houdini/houdini_fluid_half
coil_pvp:
	$(HXX) coil_compare.hip /obj/decoupled_pvp/decoupled_pvp_fluid
coil_vp:
	$(HXX) coil_compare.hip /obj/decoupled_vp/decoupled_vp_fluid
coil_stokes:
	$(HXX) coil_compare.hip /obj/stokes/viscous_fluid

viscous_coil_houdini:
	$(HXX) viscous_coil_houdini.hip /obj/fluid/viscous_fluid

viscous_coil_stokes:
	$(HXX) viscous_coil_stokes.hip /obj/fluid/viscous_fluid

viscous_coils: viscous_coil_houdini viscous_coil_stokes

dam_break_houdini:
	$(HXX) dam_break.hip /obj/houdini/houdini_fluid

dam_break_stokes:
	$(HXX) dam_break.hip /obj/stokes/stokes_fluid

dam_break: dam_break_stokes dam_break_houdini

surface_tension_stokes:
	$(HXX) surface_tension.hip /obj/stokes_fluid/stokes

surface_tension_decoupled:
	$(HXX) surface_tension.hip /obj/decoupled_fluid/decoupled

surface_tension_stokes_5:
	$(HXX) surface_tension_5.hip /obj/stokes_fluid/stokes

surface_tension_decoupled_5:
	$(HXX) surface_tension_5.hip /obj/decoupled_fluid/decoupled

surface_tension_stokes_10-5:
	$(HXX) surface_tension_10-5.hip /obj/stokes_fluid/stokes

surface_tension_decoupled_10-5:
	$(HXX) surface_tension_10-5.hip /obj/decoupled_fluid/decoupled

surface_tension_stokes_10-20:
	$(HXX) surface_tension_10-20.hip /obj/stokes_fluid/stokes

surface_tension_decoupled_10-20:
	$(HXX) surface_tension_10-20.hip /obj/decoupled_fluid/decoupled

surface_tension_stokes_50-20:
	$(HXX) surface_tension_50-20.hip /obj/stokes_fluid/stokes

surface_tension_decoupled_50-20:
	$(HXX) surface_tension_50-20.hip /obj/decoupled_fluid/decoupled

surface_tension: surface_tension_stokes surface_tension_decoupled
surface_tension_5: surface_tension_stokes_5 surface_tension_decoupled_5
surface_tension_10-5: surface_tension_stokes_10-5 surface_tension_decoupled_10-5
surface_tension_10-20: surface_tension_stokes_10-20 surface_tension_decoupled_10-20
surface_tension_50-20: surface_tension_stokes_50-20 surface_tension_decoupled_50-20
surface_tension_all: surface_tension_5 surface_tension_10-5 surface_tension_10-20 surface_tension_50-20

rotating_ball_stokes:
	$(HXX) rotating_ball_stokes.hip /obj/fluid/together | tee -a momentum_stokes.txt
rotating_ball_vp:
	$(HXX) rotating_ball_vp.hip /obj/fluid/together | tee -a momentum_vp.txt
rotating_ball_pvp:
	$(HXX) rotating_ball_pvp.hip /obj/fluid/together | tee -a momentum_pvp.txt

rotating_ball: rotating_ball_stokes rotating_ball_vp rotating_ball_pvp

ball:
	$(HXX) ball_thru.hip /obj/fluid/fluid

# Render targets
render_rope_coiling_25: viscous_coil_stokes_25 viscous_coil_houdini_25
render_rope_coiling_100: viscous_coil_stokes_100 viscous_coil_houdini_100
render_rope_coiling_molasses:
render_rope_coiling_conveyor_belt:
	$(HXX) conveyor_belt.hip /out/mantra_ipr
render_piling_armadillos: render_armadillos_stokes render_armadillos_houdini
render_colliding_characters:
render_ball_through_goop:
render_collapsing_liquid_squab:
render_variable_viscosity_block:
	$(HXX) variable_viscosity.hip /out/mantra_ipr
render_caramel_on_wafer:
	$(HXX) buckling.hip /out/mantra_ipr

render_armadillos_stokes:
	$(HXX) armadillos.hip /out/stokes
render_armadillos_houdini:
	$(HXX) armadillos.hip /out/houdini

render_rotating_ball: 
	$(HXX) rotating_ball.hip /out/mantra_ipr

render_ball:
	$(HXX) ball_thru.hip /out/houdini
