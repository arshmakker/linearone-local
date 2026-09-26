# l1score: portable x86-64-v3 / apple-m1 build; ONNX Runtime C API from vendor/onnxruntime (see README) or ORT=path.
CC      ?= cc
ORT     ?= $(abspath ../../vendor/onnxruntime)
OS      := $(shell uname -s)
ARCH    := $(shell uname -m)
ifeq ($(OS),Darwin)
  GC_FLAG := -Wl,-dead_strip
else
  GC_FLAG := -Wl,--gc-sections
endif
ifneq ($(filter x86_64 amd64,$(ARCH)),)
  MARCH := -march=x86-64-v3 -mtune=generic
else ifeq ($(OS),Darwin)
  MARCH := -mcpu=apple-m1
else
  MARCH := -march=armv8.2-a
endif
CFLAGS  ?= -O3 $(MARCH) -flto -ffast-math -fno-plt -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS += -isystem $(ORT)/include -DNDEBUG
LDFLAGS += -flto -L$(ORT)/lib -Wl,-rpath,$(ORT)/lib $(GC_FLAG)
LDLIBS  += -lonnxruntime -lm
l1score: l1score.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)
clean:
	rm -f l1score
