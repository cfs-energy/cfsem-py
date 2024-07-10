#!/bin/bash

# Run the python examples that generate figures for the documentation.
(cd examples && for f in *.py; do python "$f"; done)
cp examples/*.png docs/python/example_outputs

# Build the top-level documentation (including the python docs)
mkdocs build
