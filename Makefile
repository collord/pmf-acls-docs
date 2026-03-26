# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run sphinx-build
SOURCEDIR     = docs
BUILDDIR      = _build

.PHONY: help html clean livehtml

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	@rm -rf $(BUILDDIR)

livehtml:
	uv run sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
