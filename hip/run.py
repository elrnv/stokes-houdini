"""
A simple script to kick off a certain ROP remotely, optionally
with a given framerange.
Usage: hython path/to/script/houBatch.py /path/to/hipfile /hou/path/to/rop
TODO: Add options for multiple ROPs, hperf.
"""
import hou, sys
from os.path import basename, splitext

usage = "usage: run.py <hip file> <node name relative to /obj/>"

if len(sys.argv) != 3:
  print usage
  sys.exit(1)

# Load the hip file
hipfile = sys.argv[1]
hou.hipFile.load(hipfile, ignore_load_warnings=True)

nodepath = sys.argv[2]
node = hou.node("/obj/" + nodepath)
if node is None:
  print "Couldn't locate the node to cook"
  print usage
  sys.exit(2)

nodepath = nodepath.replace("/","-")
projname = splitext(basename(hipfile))[0] # project name
profilename = projname + "_" + nodepath

# setup performance monitor
opts = hou.PerfMonRecordOptions(
    cook_stats = False,
    solve_stats = True,
    draw_stats = False,
    gpu_draw_stats = False,
    viewport_stats = False,
    script_stats = False,
    render_stats = False,
    thread_stats = True,
    frame_stats = False,
    memory_stats = True,
    errors = True) # new options object
profile = hou.perfMon.startProfile(profilename, opts)

# do the thing
node.parm("executebackground").pressButton()

profile.stop()

hperfpath = "stats/" + profilename + ".hperf"

print("Saving hperf to " + hperfpath)
profile.save(hperfpath)
