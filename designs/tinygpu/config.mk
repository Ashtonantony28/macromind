# MacroMind SG9: tiny-gpu-extended macro variant on NanGate45 + fakeram45.
# Usage (from anywhere):
#   make -C $FLOW_HOME DESIGN_CONFIG=$PWD/designs/tinygpu/config.mk TG_CONFIG=c8 [target]
# TG_CONFIG selects a configuration (see configs.mk): ff2 (upstream default,
# flops only), c4 / c8 / c16 (SRAM-macro variant with 4 / 8 / 16 cores).
TG_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TG_CONFIG ?= c8
include $(TG_DIR)/configs.mk

export DESIGN_NICKNAME = tinygpu_$(TG_CONFIG)
export DESIGN_NAME     = tinygpu_top
export PLATFORM        = nangate45

export SYNTH_HDL_FRONTEND = slang
export VERILOG_FILES   = $(addprefix $(TG_DIR)/,$(shell cat $(TG_DIR)/files.f))
export VERILOG_DEFINES = $(TG_DEFINES)
export SDC_FILE        = $(TG_DIR)/constraint.sdc
export TG_CLOCK_PERIOD ?= $(TG_CLOCK)

TG_RAMS ?= fakeram45_64x32 fakeram45_64x21 fakeram45_256x34
ifneq ($(TG_MACROS),0)
export ADDITIONAL_LEFS = $(foreach r,$(TG_RAMS),$(PLATFORM_DIR)/lef/$(r).lef)
export ADDITIONAL_LIBS = $(foreach r,$(TG_RAMS),$(PLATFORM_DIR)/lib/$(r).lib)
endif

export CORE_UTILIZATION ?= $(TG_UTIL)
export CORE_ASPECT_RATIO ?= 1
export CORE_MARGIN ?= 4
export PLACE_DENSITY ?= $(TG_DENSITY)
export MACRO_PLACE_HALO ?= 8 8
export TNS_END_PERCENT ?= 100
export SKIP_LAST_GASP ?= 1
