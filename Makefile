
cover_packages=waymax

out=out
tr=$(out)/test-results

junit=--junitxml=$(tr)/junit.xml
parallel=-n auto --dist=loadfile
extra=--capture=no -v

clean-test:
	poetry run coverage erase
	rm -rf $(tr) $(tr)

test: clean-test
	mkdir -p  $(tr)
	poetry run pytest $(extra) $(junit) waymax

test-parallel: clean-test
	mkdir -p  $(tr)
	poetry run pytest $(extra) $(junit) $(parallel) waymax
