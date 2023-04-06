#!/bin/bash

# Run the test
cp tests/test_* ./src && cd src
python -m unittest discover . "test_*.py"
rm test_*.py