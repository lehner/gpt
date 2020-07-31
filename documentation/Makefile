# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
BUILDHTMLDIR  = "$(BUILDDIR)/html"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help Makefile


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html: Makefile
	@echo "Copy Figures"
	@cp -p logo/logo.png source/_static/logo.png
	@echo "Build HTML Documentation"
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "Adjust HTML Paths"
	find ${BUILDHTMLDIR} -name \*.html -exec sed -i -- 's/_static/static/g'   {} \;
	find ${BUILDHTMLDIR} -name \*.html -exec sed -i -- 's/_sources/sources/g' {} \;
	find ${BUILDHTMLDIR} -name \*.html -exec sed -i -- 's/_images/images/g'   {} \;
	find ${BUILDHTMLDIR} -name \*.html -exec sed -i -- 's/_modules/modules/g' {} \;	
	cd ${BUILDHTMLDIR}; if [ -e _static/ ] ; then mv _static/ ./static/; fi
	cd ${BUILDHTMLDIR}; if [ -e _sources/ ] ; then mv _sources/ ./sources/; fi
	cd ${BUILDHTMLDIR}; if [ -e _images/ ] ; then mv _images/ ./images/; fi
	cd ${BUILDHTMLDIR}; if [ -e _modules/ ] ; then mv _modules/ ./modules/; fi
	@echo "Done"
