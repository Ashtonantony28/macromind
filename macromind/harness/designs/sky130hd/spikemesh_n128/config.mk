# MacroMind harness adapter: SpikeMesh neurocore macro variant, config n128 (10 sky130 OpenRAM macros).
# Lets the harness address it as sky130hd/spikemesh_n128 without editing designs/spikemesh.
# macromind-digest: designs/spikemesh
export SM_CONFIG = n128
include $(dir $(lastword $(MAKEFILE_LIST)))../../../../../designs/spikemesh/config.mk
