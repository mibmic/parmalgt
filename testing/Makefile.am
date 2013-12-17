# global includes

AUTOMAKE_OPTIONS = foreign subdir-objects

AM_CPPFLAGS = -I${GTEST_DIR}/include -I${GTEST_DIR} -I..\
	-I${top_srcdir} -I${top_srcdir}/include -DHAVE_TR1 \
	-DDSFMT_MEXP=19937 -I${top_srcdir}/ranlxd \
	-I${top_srcdir}/sun-sources/include

# gtest stuff

lib_LTLIBRARIES = lib/libgtest.la lib/libgtest_main.la
lib_libgtest_la_SOURCES = ${GTEST_DIR}/src/gtest-all.cc
lib_libgtest_main_la_SOURCES = ${GTEST_DIR}/src/gtest_main.cc
lib_libgtest_main_la_LIBADD = lib/libgtest.la

# test programs we want to build

# NOTE: The point test does not work with the new lattice class at the
# moment

bin_PROGRAMS = test_Geometry test_Bgfield test_Helper test_newQCDpt \
	test_PtTypes test_LocalField test_Clover 

test_LocalField_SOURCES = test_LocalField.cc
test_LocalField_LDADD = lib/libgtest_main.la -lpthread

# DH: does not compile!
#test_CommunicatorSU3_SOURCES = test_CommunicatorSU3.cc
#test_CommunicatorSU3_LDADD = lib/libgtest_main.la

# DH: Does not compile!
#test_Communicator_SOURCES = test_Communicator.cc
#test_Communicator_LDADD = lib/libgtest_main.la

test_Clover_SOURCES = test_Clover.cc ../MyRand.cc\
	../Background.cc ../ranlxd/ranlxd.cc
test_Clover_LDADD = lib/libgtest_main.la -lpthread

test_Geometry_SOURCES = test_Geometry.cc
test_Geometry_LDADD = lib/libgtest_main.la -lpthread

test_Bgfield_SOURCES = test_Bgfield.cc ../MyRand.cc \
		       ../Background.cc
test_Bgfield_LDADD = lib/libgtest_main.la -lpthread

test_Helper_SOURCES = test_Helper.cc ../MyRand.cc
test_Helper_LDADD = lib/libgtest_main.la -lpthread

test_newQCDpt_SOURCES = test_newQCDpt.cc ../MyRand.cc \
	../newQCDpt.cc ../Background.cc ../ranlxd/ranlxd.cc
test_newQCDpt_LDADD = lib/libgtest.la -lpthread

test_PtTypes_SOURCES = test_PtTypes.cc ../MyRand.cc \
	../ranlxd/ranlxd.cc
test_PtTypes_LDADD = lib/libgtest_main.la -lpthread

#test_Fermi_SOURCES = test_Fermi.cc MyMath.cc MyRand.cc \
# newQCDpt.cc Point.cc Background.cc IO.cc

TESTS = ${bin_PROGRAMS}
