#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.59])
AC_INIT([ParmaLGT], [0.11], [])
AM_INIT_AUTOMAKE([1.09])
AC_CONFIG_SRCDIR([Background.h])
AC_CONFIG_HEADERS([config.h])

# Prevent autoconf from trying to regenerate configure
AM_MAINTAINER_MODE

# Checks for programs.
AC_PROG_AWK
AC_PROG_CXX
AC_PROG_LIBTOOL
AC_PROG_MAKE_SET
AC_LANG_CPLUSPLUS

# Checks for c++0x and c++11
AX_CXX_COMPILE_STDCXX_0X
AM_CONDITIONAL(HAVE_STDCXX_0X, test "x$ax_cv_cxx_compile_cxx0x_cxx" == "xyes")

AX_CXX_COMPILE_STDCXX_11
AM_CONDITIONAL(HAVE_STDCXX_11, test "x$ax_cv_cxx_compile_cxx11_cxx" == "xyes")

# Checks for header files.
AC_CHECK_HEADERS([fenv.h inttypes.h limits.h stdlib.h string.h sys/time.h unistd.h])

AC_ARG_WITH([gtest-srcdir], [AS_HELP_STRING([--with-gtest-srcdir=DIR],
 [Path to Googletest.])], [], [])

AC_SUBST([GTEST_DIR],["$with_gtest_srcdir"])
AM_CONDITIONAL([GTEST], [test x$with_gtest_srcdir != x])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
# AC_TYPE_UINT32_T
# AC_TYPE_UINT64_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([bzero clock_gettime gettimeofday memset pow sqrt])
AC_CONFIG_MACRO_DIR([m4])

AC_CONFIG_FILES([Makefile Doxyfile testing/Makefile])

AC_OUTPUT





