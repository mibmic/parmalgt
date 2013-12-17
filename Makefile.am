AUTOMAKE_OPTIONS = foreign subdir-objects

AM_CPPFLAGS = -DHAVE_TR1 -DDSFMT_MEXP=19937 -I. -I@abs_top_srcdir@/include\
	 -I${top_srcdir}/ranlxd -I${top_srcdir}/sun-sources/include

ACLOCAL_AMFLAGS = -I m4
noinst_HEADERS = Background.h nspt.h MassC.h input.h MyTime.h \
  QCDenvNODEpt.h QCDenvNODE.h Point.hpp newQCDpt.h lattice.h \
  PtTypes.hpp MyQCD.h newMyQCD.h Types.h \
  choices.h PtAlgo.hpp GammaStuff.h LocalField.hpp Geometry.hpp \
  Kernels.hpp uparam.hpp IO.hpp util.hpp ranlxd/ranlxd.hpp\
  include/Kernels/generic/RungeKutta.hpp Communicator.hpp\
  Inverter.hpp include/Kernels/generic/Plaquette.hpp

dist_data_DATA = input run.sh

bin_PROGRAMS = LocalQuenchFlow LocalQuench #Invert Quench

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

#Fermi_SOURCES = Fermi.cc MyMath.cc MyRand.cc \
#  newQCDpt.cc Point.cc Background.cc IO.cc

#Invert_SOURCES = Invert.cc ranlxd/ranlxd.cc \
#  newQCDpt.cc Point.cc Background.cc IO.cc

if GTEST
  SUBDIRS = testing
else
  DIST_SUBDIRS = $(SUBDIRS)
endif

.PHONY: docs

docs:
	mkdir -p ./docs/doxygen
	doxygen Doxyfile

