# Copyright (c) Adithya Kumar, The Pennsylvania State University. All rights reserved.
# Licensed under the MIT License.

##############################################################################################
# GLOBALS
# Compiler Flags
.DEFAULT_GOAL := all
CXX = g++

# Some OS Specific and Host Specific Settings
HOST_SYSTEM = $(shell uname | cut -f 1 -d_)
SYSTEM ?= $(HOST_SYSTEM)
HOST_NAME = $(shell hostname | cut -f 1 -d.)

APP := dist_rpc_bench 

# Build using pkg-config variables if possible
#ifneq ($(shell pkg-config --exists libdpdk && echo 0),0)
#$(error "no installation of DPDK found")
#endif
PKGCONF ?= pkg-config
DPDK_PC_FILE := $(shell $(PKGCONF) --path libdpdk 2>/dev/null)

# Debug build flags
ifeq ($(dbg),1)
NVCCFLAGS += -g -G
CXXFLAGS   += -g3 -ggdb
BUILD_TYPE := debug
else
BUILD_TYPE := release
endif

SOURCES := $(wildcard *.cc)
OBJECTS := $(patsubst %.cc,%.o,$(SOURCES))
DEPENDS := $(patsubst %.cc,%.d,$(SOURCES))

CXXFLAGS += -std=c++17 -O2 -I./utils/
#CXXFLAGS += $(shell $(PKGCONF) --cflags libdpdk) 
#LDFLAGS := $(shell $(PKGCONF) --libs libdpdk) -lstdc++
LDFLAGS += -lstdc++ -lpthread
#LDFLAGS += -Wl,--no-as-needed -Wl,--as-needed -lstdc++\
#					 -ldl -lrt -pthread -lpthread -lstdc++fs

# ADD MORE WARNINGS!
WARNING := -Wall -Wextra

.PHONY: all clean

all: build/$(APP)

# Ensure build folder is created
build:
	@mkdir -p build/

build/%.o: %.cc Makefile | build
	$(CXX) $(WARNING) $(CXXFLAGS) -MMD -MP -c $< -o $@

##################################################################################################

# Linking the executable from the object files
#$(APP): $(OBJECTS) $(DPDK_PC_FILE) | build
build/$(APP): $(OBJECTS) | build
	$(CXX) $(WARNING) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

-include $(DEPENDS)

##################################################################################################

debug: CXXFLAGS += -DDEBUG -g -std=c++17 -O0
debug: CCFLAGS += -DDEBUG -g
debug: $(TARGETS)

##################################################################################################

clean:
	$(RM) $(OBJECTS) $(DEPENDS) $(APP)
	test -d build && rmdir -p build || true
