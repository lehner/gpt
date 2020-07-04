
# Using github-pages with sphinx-build documentation
# is a pain in the ass.
# This makefile solves the problems quickly.
# just add these lines to the "html" rule that sphinx-quickstart
# generated and save this file with the name github-make in 
# the folder that contains your sphinx-Makefile:
#
#	@echo "Copying files to /docs..."
#	if [ -e ../docs ] ;then rm -rf ../docs;fi
#	cp -r $(BUILDDIR)/html ../docs
#	@echo "done"
#	@echo "calling postbuild..."
#	cp github-make ../docs/Makefile
#	cd ../docs && make github
#	@echo "done"
#
# then the command "make html" will build a 
# site suiting github-pages in ../docs


# ~Makefile~
github:
	find . -name \*.html -exec sed -i -- 's/_static/static/g'   {} \;
	find . -name \*.html -exec sed -i -- 's/_sources/sources/g' {} \;
	find . -name \*.html -exec sed -i -- 's/_images/images/g'   {} \;
	find . -name \*.html -exec sed -i -- 's/_modules/modules/g' {} \;	
	if [ -e  _static/ ] ; then mv _static/ ./static/; fi
	if [ -e  _sources/ ] ; then mv _sources/ ./sources/; fi
	if [ -e  _images/ ] ; then mv _images/ ./images/; fi
	if [ -e _modules/ ] ; then mv _modules/ ./modules/; fi
