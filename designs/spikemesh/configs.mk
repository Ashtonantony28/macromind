# Per-configuration knobs for designs/spikemesh/config.mk.
# Macros (sky130_sram_1rw1r_64x256_8, 1040.76 x 403.035 um each): weight memory
# N*A/8 rows of 64 bits -> N*A/2048 macros, fan-out tables FD/256 macros each (x2).
ifeq ($(SM_CONFIG),n128)
  SM_DEFINES := -D SM_N=128 -D SM_A=128 -D SM_FD=256 -D SM_LEARN=0
  SM_DIE ?= 0 0 3000 3000
  SM_CORE ?= 10 10 2990 2990
else ifeq ($(SM_CONFIG),n128_l2)
  SM_DEFINES := -D SM_N=128 -D SM_A=128 -D SM_FD=256 -D SM_LEARN=2
  SM_DIE ?= 0 0 4000 4000
  SM_CORE ?= 10 10 3990 3990
else
  $(error unknown SM_CONFIG=$(SM_CONFIG))
endif
SM_CLOCK ?= 25.0
