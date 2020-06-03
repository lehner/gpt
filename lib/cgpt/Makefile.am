
# libtools preferrs this
ACLOCAL_AMFLAGS = -I m4

# make the cgpt a libtool library.
# Conveniently, this will take care of the rest (prefix, suffix,
# linking, fPIC, ....)

pyexec_LTLIBRARIES = cgpt.la

include ccfiles.inc
include hfiles.inc

# Automake docu says to list convenience headers together with the
# sources.  noinst_HEADERS would also be possible, but manual
# (https://www.gnu.org/software/automake/manual/automake.html)
# recommends to use sources in our case.
cgpt_la_SOURCES = $(CGPT_CCFILES) $(CGPT_HFILES)

# The flags in ..._LDFLAGS are __not__ linker flags, but flags for
# GNU libtool. It takes care of all the platform specific stuff when
# creating shared library objects:
# -avoid-version   avoid the linux typicall *.so.major.minor stuff.
# -module          avoid putting lib* as prefix.
# -shared          make it a shared object
# -export-dynamic  Very important in our case and easily overlooked:
#                  cgpt itself will link to another library. But
#                  when loaded by python, don't use @rpath or other
#                  tricks, but python will dlopen the necessary symbols.
#                  To avoit errors, use this flag!
cgpt_la_LDFLAGS = -avoid-version -module -export-dynamic -shared

# Compiler Flags.
# ..._CFLAGS not needed, since this is a pure C++ library.
# ..._CXXFLAGS  is set for compiling and linking,
# ..._CPPFLAGS  is set only for compiling (not for linking)
# ..._LIBADD    are flags for linking
# cgpt_la_CFLAGS = $(PYTHON_CFLAGS) $(GRID_CFLAGS)
cgpt_la_CXXFLAGS = $(PYTHON_INCLUDES) $(GRID_CXXFLAGS)
cgpt_la_CPPFLAGS = $(PYTHON_INCLUDES) $(GRID_CPPFLAGS) \
		   -I$(GRID_PREFIX)/include \
		   -I$(NUMPY_INCLUDE)
cgpt_la_LIBADD = -L$(GRID_PREFIX)/lib -lGrid \
		 $(GRID_LDFLAGS) $(GRID_LIBS)

# link dependencies go into <programname>_LDADD
# libtool figures out all the -l  and -L compiler flags for you
# if this has an *.la  file

# Additional flags shoudl go into AM_CPPFLAGS.
# There is a big difference between setting CFLAGS or AM_CFLAGS
# If the user has CFLAGS defined e.g., as a shell environment variable,
# the value defined here in Makefile.am will be overwritten!
# The values of AM_CFLAGS will be prepended to CFLAGS.
# Therefore, default values should usually go into AM_CFLAGS

# to make a tarball, try "make dist"
# and to check for errors/forgotten files, use "make distcheck".

all: gen_versions

.PHONY: gen_versions

gen_versions:
	@if ( git -C $(top_srcdir) status --porcelain  > /dev/null 2>&1  ); \
	then \
	if [ `git -C "$(top_srcdir)" status --porcelain | grep -v '??'\
	    | wc -l` -gt 0 ];\
	then \
                a="uncommited changes";\
        else \
                a="clean"; \
        fi; \
	git -C "$(top_srcdir)" \
	    log -n 1 \
	    --format=format:"#define GPT_GIT_HASH \"%H:%d $$a\"%n" \
	    HEAD \
	    > lib/gversions.h; \
	else \
	cp "$(top_srcdir)/lib/gversions.h" \
	   "$(top_builddir)/lib"; \
	fi;


lib/gversions.h: gen_versions

CLEANFILES = lib/gversions.h
EXTRA_DIST = LICENSE \
	     README.setup.md
dist-hook: gen_versions
	chmod u+w \
	    $(distdir)/lib/ \
	    $(distdir)/lib/gversions.h; \
	echo '#define GPT_FROM_DIST "beware! from autotools dist."' \
	    >> $(top_builddir)/lib/gversions.h; \
	cp  $(top_builddir)/lib/gversions.h \
	    $(distdir)/lib/gversions.h;

# Note: currently tests, docs, etc. are not listed here.
# Possibly use dist_doc_DATA for documentation.
# Tests: can use TAP harness.
