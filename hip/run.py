"""
A simple script to kick off a certain ROP remotely
Usage: hython path/to/script/houBatch.py /path/to/hipfile /hou/path/to/rop
"""
import hou, sys
import argparse
from os.path import basename, splitext, dirname, exists
from os import makedirs

parser = argparse.ArgumentParser()
parser.add_argument("hip_file", help="Hip file to load, which contains the desired node to be run.")
parser.add_argument("node_path", help="Absolute node path. This node must have a button named <executebackground>")
parser.add_argument("--nohperf", help="Do not collect performance stats.", action="store_true")
args = parser.parse_args()

# Load the hip file
hipfile = args.hip_file
hou.hipFile.load(hipfile, ignore_load_warnings=True)

nodepath = args.node_path
node = hou.node(nodepath)
if node is None:
  print "Couldn't locate the node to cook"
  print usage
  sys.exit(2)

statspath = "stats"
if not exists(statspath):
  makedirs(statspath)


# do the thing function
def run(node):
  node.parm("executebackground").pressButton()
  

if args.nohperf:
  run(node) # do the thing

else:
  nodepath = nodepath.replace("/","-")
  projectname = splitext(basename(hipfile))[0]
  profilename = projectname + "_" + nodepath

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

  run(node) # do the thing

  profile.stop()

  hperfpath = statspath + "/" + profilename + ".hperf"

  print("Saving hperf to " + hperfpath)
  profile.save(hperfpath)
