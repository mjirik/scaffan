#!/bin/bash

echo "==============================="
echo "==============================="
echo "======build.sh================="
echo "==============================="
echo "==============================="
echo "build.sh"
echo "scaffan build.sh" > ~/scaffan.txt
pwd
ls

echo "prefix"
echo %PREFIX%
ls %PREFIX%
echo "recipe dir"
echo %RECIPE_DIR%
ls %RECIPE_DIR%
# rm -rf examples
$PYTHON setup.py install
