HXX=hython -j 24 run.py 

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

viscous_coil:
	$(HXX) viscous_coil.hip fluid/viscous_fluid

viscous_coil_stokes:
	$(HXX) viscous_coil_stokes.hip fluid/viscous_fluid

viscous_coils: viscous_coil viscous_coil_stokes

dam_break_houdini:
	$(HXX) dam_break.hip houdini/houdini_fluid

dam_break_stokes:
	$(HXX) dam_break.hip stokes/stokes_fluid

dam_break: dam_break_stokes dam_break_houdini
