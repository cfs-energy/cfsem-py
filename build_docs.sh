#!/bin/bash

# Run the python examples that generate figures for the documentation.
export CFSEM_TESTING="True";
(cd examples && for f in *.py; do uv run python "$f"; done)
cp examples/*.png docs/python/example_outputs

# Build the top-level documentation (including the python docs)
if [[ -z "$READTHEDOCS_OUTPUT" ]]; then
    # This is not a readthedocs build
    mkdocs build;
fi
