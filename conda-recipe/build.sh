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

echo "prefix $PREFIX"
ls $PREFIX
echo "recipe dir $RECIPE_DIR"
ls $RECIPE_DIR
echo "prefix $PREFIX" >> ~/scaffan.txt
ls $PREFIX >> ~/scaffan.txt
echo "recipe dir $RECIPE_DIR" >> ~/scaffan.txt
ls $RECIPE_DIR >> ~/scaffan.txt
# rm -rf examples
$PYTHON setup.py install
