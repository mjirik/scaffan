#!/bin/bash

echo "==============================="
echo "==============================="
echo "======build.sh================="
echo "==============================="
echo "==============================="
echo "build.sh"
pwd
ls
echo "scaffan build.sh" > ~/scaffan.txt
date >> ~/scaffan.txt
echo "===== recipe dir $RECIPE_DIR" >> ~/scaffan.txt
ls $RECIPE_DIR >> ~/scaffan.txt
# rm -rf examples
echo "===== prefix $PREFIX ======" >> ~/scaffan.txt
ls $PREFIX >> ~/scaffan.txt
echo "=========== prefix dir lib =====" >> ~/scaffan.txt
ls $PREFIX/lib >> ~/scaffan.txt
echo "prefix $PREFIX"
ls $PREFIX
echo "recipe dir $RECIPE_DIR ====="
ls $RECIPE_DIR
$PYTHON setup.py install

echo "install finished ======= " >> ~/scaffan.txt
# ls $PREFIX/lib >> ~/scaffan.txt
