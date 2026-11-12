# MacroMind: SpikeMesh neurocore macro variant on sky130hd + OpenRAM 1R1W macros.
# Usage: make -C $FLOW_HOME DESIGN_CONFIG=$PWD/designs/spikemesh/config.mk SM_CONFIG=n128 [target]
# SM_CONFIG: n128 (10 macros), n128_l2 (on-chip learning, more macros). See configs.mk.
SM_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SM_CONFIG ?= n128
include $(SM_DIR)/configs.mk

export DESIGN_NICKNAME = spikemesh_$(SM_CONFIG)
export DESIGN_NAME     = spikemesh_core_top
export PLATFORM        = sky130hd

export SYNTH_HDL_FRONTEND = slang
export VERILOG_FILES   = $(addprefix $(SM_DIR)/,$(shell cat $(SM_DIR)/files.f))
export VERILOG_DEFINES = $(SM_DEFINES)
export SDC_FILE        = $(SM_DIR)/constraint.sdc
export SM_CLOCK_PERIOD ?= $(SM_CLOCK)

SM_RAM := sky130_sram_1rw1r_64x256_8
SM_RAM_DIR := $(FLOW_HOME)/platforms/sky130ram/$(SM_RAM)
export ADDITIONAL_LEFS = $(SM_RAM_DIR)/$(SM_RAM).lef
export ADDITIONAL_LIBS = $(SM_RAM_DIR)/$(SM_RAM)_TT_1p8V_25C.lib
export ADDITIONAL_GDS  = $(SM_RAM_DIR)/$(SM_RAM).gds

export DIE_AREA  = $(SM_DIE)
export CORE_AREA = $(SM_CORE)
export MACRO_PLACE_HALO ?= 40 40
export PLACE_DENSITY ?= 0.45
export TNS_END_PERCENT ?= 100
export SKIP_LAST_GASP ?= 1
