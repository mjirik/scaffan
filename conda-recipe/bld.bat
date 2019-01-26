echo "prefix"
echo %PREFIX%
dir %PREFIX%
echo "recipe dir"
echo %RECIPE_DIR%
dir %RECIPE_DIR%
"%PYTHON%" setup.py install
if errorlevel 1 exit 1

mkdir -p "$PREFIX%"/graphics
copy "%RECIPE_DIR%"/graphics/scaffan_icon512.png "%PREFIX%"/graphics/
