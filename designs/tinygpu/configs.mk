# Per-configuration knobs for designs/tinygpu/config.mk.
# All macro configs: 32-bit words, 4096-word data memory, 4 data channels,
# 4 threads/core, I-cache 64 lines/core, shared memory 64 words/core,
# L1-D of TG_DCACHE_BANKS banks x 256 lines (1024 lines; 2048 for c16).
# Clock: the ALU's single-cycle 32-bit divider makes the critical path ~15 ns on
# NanGate45 (first c4 floorplan at 5 ns: WNS -10.25 ns), hence 15 ns (~67 MHz).
# Macros: NUM_CORES x (fakeram45_64x32 SMEM + fakeram45_64x21 I-cache) + BANKS x fakeram45_256x34.
ifeq ($(TG_CONFIG),ff2)
  # Upstream default configuration (8-bit, 2 cores, no caches): flop-only baseline.
  TG_MACROS := 0
  TG_DEFINES := -D TG_MACROS=0 -D TG_NUM_CORES=2 -D TG_WORD_BITS=8 -D TG_DATA_ADDR_BITS=8 -D TG_WARPS=1 \
                -D TG_ICACHE_LINES=0 -D TG_DCACHE_LINES=0
  TG_CLOCK ?= 5.0
  TG_UTIL ?= 40
  TG_DENSITY ?= 0.60
else ifeq ($(TG_CONFIG),c4)
  TG_MACROS := 1
  TG_DEFINES := -D TG_MACROS=1 -D TG_NUM_CORES=4 -D TG_DCACHE_BANKS=4
  TG_CLOCK ?= 15.0
  TG_UTIL ?= 35
  TG_DENSITY ?= 0.60
else ifeq ($(TG_CONFIG),c8)
  TG_MACROS := 1
  TG_DEFINES := -D TG_MACROS=1 -D TG_NUM_CORES=8 -D TG_DCACHE_BANKS=4
  TG_CLOCK ?= 15.0
  TG_UTIL ?= 35
  TG_DENSITY ?= 0.60
else ifeq ($(TG_CONFIG),c16)
  TG_MACROS := 1
  TG_DEFINES := -D TG_MACROS=1 -D TG_NUM_CORES=16 -D TG_DCACHE_BANKS=8 -D TG_DCACHE_LINES=2048
  TG_CLOCK ?= 15.0
  TG_UTIL ?= 35
  TG_DENSITY ?= 0.60
else
  $(error unknown TG_CONFIG=$(TG_CONFIG): use ff2, c4, c8 or c16)
endif
