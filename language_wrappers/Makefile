# This Makefile template requires the following variables to be set
# in the environment or on the command-line:
#   JULIA: path to julia[.exe] executable
#   BIN:   binary build directory

ifndef JULIA
  $(error "Please pass JULIA=[path of target julia binary], or set as environment variable!")
endif
ifndef BIN
  $(error "Please pass BIN=[path of build directory], or set as environment variable!")
endif

#=============================================================================
# location of test source
SRCDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PKGDIR := $(dir $(SRCDIR))
JULIAHOME := $(abspath $(JULIA)/../..)
BUILDSCRIPT := $(BIN)/../share/julia/juliac-buildscript.jl
# include $(JULIAHOME)/Make.inc

# FIXME
SHLIB_EXT := so

# get the executable suffix, if any
EXE := $(suffix $(abspath $(JULIA)))

# get compiler and linker flags. (see: `contrib/julia-config.jl`)
JULIA_CONFIG := $(JULIA) -e 'include(joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia", "julia-config.jl"))' --
CPPFLAGS_ADD :=
CFLAGS_ADD = $(shell $(JULIA_CONFIG) --cflags)
LDFLAGS_ADD = -lm $(shell $(JULIA_CONFIG) --ldflags --ldlibs) -ljulia-internal

#=============================================================================

release: NMFMerge-jl.$(SHLIB_EXT)

NMFMerge-jl.$(SHLIB_EXT): $(PKGDIR)/src/NMFMerge.jl $(BUILDSCRIPT)
	$(JULIA) -t 1 -J $(BIN)/../lib/julia/sys.$(SHLIB_EXT) --project --startup-file=no --history-file=no --output-incremental=no --strip-ir --strip-metadata --experimental --trim $(BUILDSCRIPT) $(SRCDIR)/lib.jl --output-lib NMFMerge-jl.so

check: hello$(EXE) basic_jll$(EXE)   # FIXME
	$(JULIA) --depwarn=error $(SRCDIR)/../runtests.jl $(SRCDIR)/trimming

clean:  # FIXME
	-rm -f hello$(EXE) basic_jll$(EXE) init.o hello.o basic_jll.o

.PHONY: release clean check

# Makefile debugging trick:
# call print-VARIABLE to see the runtime value of any variable
print-%:
	@echo '$*=$($*)'
