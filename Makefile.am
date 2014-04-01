AUTOMAKE_OPTIONS = foreign subdir-objects

AM_CPPFLAGS = -DDSFMT_MEXP=19937 -I. -I@abs_top_srcdir@/include\
	-I${top_srcdir}/ranlxd -I${top_srcdir}/sun-sources/include\
	-I/include/Kernels/generic

ACLOCAL_AMFLAGS = -I m4
noinst_HEADERS = Background.h MassC.h MyTime.h Point.hpp newQCDpt.h \
	PtTypes.hpp newMyQCD.h Types.h PtAlgo.hpp GammaStuff.h \
	LocalField.hpp Geometry.hpp Kernels.hpp uparam.hpp \
	IO.hpp util.hpp ranlxd/ranlxd.hpp\
	include/Kernels/generic/RungeKutta.hpp Inverter.hpp\
	include/Kernels/generic/Plaquette.hpp Communicator.hpp\
	include/Kernels/generic/IO.hpp fft.hpp

dist_data_DATA = input run.sh

bin_PROGRAMS = LocalQuenchFlow LocalQuench LocalUnquench Fermi Invert

#ParQuenched_SOURCES = ParQuenched.cc \
#  newQCDpt.cc Point.cc Background.cc IO.cc ranlxd/ranlxd.cc

#Quench_SOURCES = Quench.cc \
#  newQCDpt.cc Point.cc Background.cc IO.cc ranlxd/ranlxd.cc

LocalQuench_SOURCES = LocalQuench.cc \
  newQCDpt.cc Point.cc Background.cc IO.cc ranlxd/ranlxd.cc

LocalQuenchFlow_SOURCES = LocalQuenchFlow.cc \
  newQCDpt.cc Point.cc Background.cc IO.cc ranlxd/ranlxd.cc


#LocalQuenchFlowTorus_SOURCES = LocalQuenchFlowTorus.cc MyMath.cc MyRand.cc \
#  newQCDpt.cc Point.cc Background.cc IO.cc

Fermi_SOURCES = Fermi.cc newQCDpt.cc Point.cc Background.cc IO.cc	\
	ranlxd/ranlxd.cc
Fermi_LDADD = -lfftw3 -lfftw3_threads

Invert_SOURCES = Invert.cc ranlxd/ranlxd.cc \
  newQCDpt.cc Point.cc Background.cc IO.cc

LocalUnquench_SOURCES = LocalUnquench.cc \
  newQCDpt.cc Point.cc Background.cc IO.cc ranlxd/ranlxd.cc
LocalUnquench_LDADD = -lfftw3 -lfftw3_threads

if GTEST
  SUBDIRS = testing
else
  DIST_SUBDIRS = $(SUBDIRS)
endif

.PHONY: docs

docs:
	mkdir -p ./docs/doxygen
	doxygen Doxyfile

