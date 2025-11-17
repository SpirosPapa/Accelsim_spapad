# Classic (pre-daemon) accel-sim build
# Builds: ../bin/$(ACCELSIM_CONFIG)/accel-sim-classic.out

ACCELSIM_ROOT      := ..
ACCELSIM_CONFIG    ?= release

# These are normally exported by ../setup_environment.sh
GPGPUSIM_ROOT      ?= $(ACCELSIM_ROOT)/gpgpu-sim
GPGPUSIM_CONFIG    ?= $(ACCELSIM_CONFIG)
CUDA_INSTALL_PATH  ?= /usr/local/cuda

BIN_DIR   := $(ACCELSIM_ROOT)/bin/$(ACCELSIM_CONFIG)
BUILD_DIR := $(ACCELSIM_ROOT)/build/$(ACCELSIM_CONFIG)/classic

CXX       ?= g++

DEBUG ?= 0
ifeq ($(DEBUG),1)
  CXXFLAGS := -Wall -O0 -g3 -fPIC -std=c++17
else
  CXXFLAGS := -Wall -O3 -g3 -fPIC -std=c++17
endif

# Match the includes used by normal accel-sim :contentReference[oaicite:0]{index=0}
CXXFLAGS += -I$(ACCELSIM_ROOT)/trace-driven \
            -I$(ACCELSIM_ROOT)/trace-parser \
            -I$(GPGPUSIM_ROOT)/libcuda \
            -I$(GPGPUSIM_ROOT)/src \
            -I$(CUDA_INSTALL_PATH)/include \
            -I$(ACCELSIM_ROOT)/build/$(ACCELSIM_CONFIG)

LIBS := -L$(GPGPUSIM_ROOT)/lib/$(GPGPUSIM_CONFIG)/ \
        -lcudart -lm -lz -lGL -pthread

# ------------------------------------------------------------
# Sources:
#  - Old main + accel-sim live in this directory
#  - We also pull in all trace-driven and trace-parser .cc files
#    from the parent tree so classic uses the same frontend code
# ------------------------------------------------------------

LOCAL_SRCS         := main_classic.cc accel-sim-classic.cc
TRACE_DRIVEN_SRCS  := $(wildcard $(ACCELSIM_ROOT)/trace-driven/*.cc)
TRACE_PARSER_SRCS  := $(wildcard $(ACCELSIM_ROOT)/trace-parser/*.cc)

LOCAL_OBJS         := $(patsubst %.cc,$(BUILD_DIR)/classic_%.o,$(LOCAL_SRCS))
TRACE_DRIVEN_OBJS  := $(patsubst $(ACCELSIM_ROOT)/trace-driven/%.cc,$(BUILD_DIR)/classic_td_%.o,$(TRACE_DRIVEN_SRCS))
TRACE_PARSER_OBJS  := $(patsubst $(ACCELSIM_ROOT)/trace-parser/%.cc,$(BUILD_DIR)/classic_tp_%.o,$(TRACE_PARSER_SRCS))

OBJS := $(LOCAL_OBJS) $(TRACE_DRIVEN_OBJS) $(TRACE_PARSER_OBJS)

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

all: $(BIN_DIR)/accel-sim-classic.out

$(BIN_DIR)/accel-sim-classic.out: $(OBJS) | $(BIN_DIR)
	$(CXX) -o $@ $^ $(LIBS)

$(BUILD_DIR)/classic_%.o: %.cc | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/classic_td_%.o: $(ACCELSIM_ROOT)/trace-driven/%.cc | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/classic_tp_%.o: $(ACCELSIM_ROOT)/trace-parser/%.cc | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)/accel-sim-classic.out

.PHONY: all clean
