# MacroMind harness adapter: tiny-gpu-extended macro variant, config c4 (12 macros).
# Lets the harness address it as nangate45/tinygpu_c4 without editing designs/tinygpu.
# macromind-digest: designs/tinygpu
export TG_CONFIG = c4
include $(dir $(lastword $(MAKEFILE_LIST)))../../../../../designs/tinygpu/config.mk
