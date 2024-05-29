#--------------------------------------------------------------------------#
# The target name should be the name of this app, which actually should be
# the same as the name of this directory.
#--------------------------------------------------------------------------#

TARGET=$(shell basename "`pwd`")

#--------------------------------------------------------------------------#
# The environment variable 'IPAMIRSOLVER' is used to locate the MaxSAT solver
# library from ../../maxsat/$(IPAMIRSOLVER). For testing purposes we set it to
# the non-incremental MaxSAT solver 'solver2022' which is compiled using
# 'minisat220' as the SAT solver.
#--------------------------------------------------------------------------#

IPAMIRSOLVER ?= solver2022
IPASIRSOLVER ?= minisat220

#--------------------------------------------------------------------------#
# Compiler flags and library dependencies.
#--------------------------------------------------------------------------#

CC = g++
CFLAGS ?= -Wall -DNDEBUG -O3 -std=c++17

DEPS = ../../maxsat/$(IPAMIRSOLVER)/libipamir$(IPAMIRSOLVER).a
LIBS = -L../../maxsat/$(IPAMIRSOLVER)/ -lipamir$(IPAMIRSOLVER)
LIBS += $(shell cat ../../maxsat/$(IPAMIRSOLVER)/LIBS 2>/dev/null)
LIBS += $(shell cat ../../maxsat/$(IPASIRSOLVER)/LIBS 2>/dev/null)
LINK=$(CC)

# Rust build system doesn't automatically link to C++ standard library
LIBS += -lstdc++


#--------------------------------------------------------------------------#
# Targets 'all' and 'clean'.
#--------------------------------------------------------------------------#

all: $(TARGET)

clean:
	rm -f $(TARGET) *.o
	rm -rf pblib

ipamir_trainscheduling: src/*rs build.rs Cargo.toml $(DEPS)
	touch build.rs && LIBS="$(LIBS)" cargo build --release && cp target/release/ipamir_trainscheduling .

#--------------------------------------------------------------------------#
# Local app specific rules.
#--------------------------------------------------------------------------#

../../maxsat/$(IPAMIRSOLVER)/libipamir$(IPAMIRSOLVER).a:
	make -C ../../maxsat/$(IPAMIRSOLVER)

