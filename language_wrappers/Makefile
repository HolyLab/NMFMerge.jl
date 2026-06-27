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
PKGDIR := $(abspath $(dir $(SRCDIR)))
JULIAHOME := $(abspath $(JULIA)/../..)
# BUILDSCRIPT := $(BIN)/../share/julia/juliac-buildscript.jl
# include $(JULIAHOME)/Make.inc

# FIXME
SHLIB_EXT := so

# get the executable suffix, if any
EXE := $(suffix $(abspath $(JULIA)))

# get compiler and linker flags. (see: `contrib/julia-config.jl`)
JULIA_CONFIG := $(JULIA) -e 'include(joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia", "julia-config.jl"))' --
JULIA_LIBDIR := $(shell $(JULIA) -e 'println(joinpath(Sys.BINDIR, "..", "lib"))' --)
CPPFLAGS_ADD :=
CFLAGS_ADD = $(shell $(JULIA_CONFIG) --cflags)
LDFLAGS_ADD = -lm $(shell $(JULIA_CONFIG) --ldflags --ldlibs) -ljulia-internal

# get the JuliaC build script
JULIAC_BUILDSCRIPT := $(shell $(JULIA) -e 'print(joinpath(Sys.BINDIR, Base.DATAROOTDIR, "julia", "juliac", "juliac-buildscript.jl"))')

#=============================================================================

release: libNMFMerge-o.a

$(BIN)/libNMFMerge-o.a: $(PKGDIR)/language_wrappers/lib.jl $(JULIAC_BUILDSCRIPT)
	$(JULIA) -t 1 -J $(JULIA_LIBDIR)/julia/sys.$(SHLIB_EXT) --startup-file=no --history-file=no --project=$(PKGDIR)/language_wrappers/ --output-o $@ --output-incremental=no --strip-ir --strip-metadata --experimental --trim $(JULIAC_BUILDSCRIPT) $< --output-lib true $(BIN)/bindinginfo_libNMFMerge.log

# check: $(BIN)/libNMFMerge$(EXE)
# 	$(JULIA) --depwarn=error $(SRCDIR)/trimming.jl $<

clean:
	-rm -f $(BIN)/libNMFMerge-o.a $(BIN)/bindinginfo_libNMFMerge.log

.PHONY: release clean

# Makefile debugging trick:
# call print-VARIABLE to see the runtime value of any variable
print-%:
	@echo '$*=$($*)'
