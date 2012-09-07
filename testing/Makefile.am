GTEST_SRC = @gtest_srcdir@

# global includes

AM_CPPFLAGS = -I${GTEST_SRC}/include -I${GTEST_SRC} -I.. -DSF\
	-I@abs_top_srcdir@

# gtest stuff

lib_LTLIBRARIES = lib/libgtest.la lib/libgtest_main.la
lib_libgtest_la_SOURCES = ${GTEST_SRC}/src/gtest-all.cc
lib_libgtest_main_la_SOURCES = ${GTEST_SRC}/src/gtest_main.cc
lib_libgtest_main_la_LIBADD = lib/libgtest.la

# test programs we want to build

# NOTE: The point test does not work with the new lattice class at the
# moment

bin_PROGRAMS = test_Geometry test_Bgfield test_Helper test_SU3 test_newQCDpt \
	test_PtTypes test_Field #test_Point

test_Geometry_SOURCES = test_Geometry.cc
test_Geometry_LDADD = lib/libgtest_main.la


test_Bgfield_SOURCES = test_Bgfield.cc ../MyRand.cc \
		       ../MyMath.cc ../Background.cc
test_Bgfield_LDADD = lib/libgtest_main.la

test_Helper_SOURCES = test_Helper.cc ../MyRand.cc
test_Helper_LDADD = lib/libgtest_main.la

test_SU3_SOURCES = test_SU3.cc ../MyMath.cc ../MyRand.cc
test_SU3_LDADD = lib/libgtest_main.la

test_newQCDpt_SOURCES = test_newQCDpt.cc ../MyMath.cc ../MyRand.cc \
	../newQCDpt.cc ../Background.cc
test_newQCDpt_LDADD = lib/libgtest.la

test_PtTypes_SOURCES = test_PtTypes.cc ../MyMath.cc ../MyRand.cc
test_PtTypes_LDADD = lib/libgtest_main.la

test_Field_SOURCES = test_Field.cc ../MyMath.cc ../MyRand.cc \
	../newQCDpt.cc ../Background.cc
test_Field_LDADD = lib/libgtest_main.la

#test_Point_SOURCES = test_Point.cc
#test_Point_LDADD = lib/libgtest_main.la

TESTS = ${bin_PROGRAMS}