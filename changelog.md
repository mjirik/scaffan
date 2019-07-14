# Changelog

## Unreleased

## 0.14

* [Changed] Compacntess definition
* [Fixed] Problem with percentile evaluation
* [Fixed] Debug print on annotation read fail

## 0.13

* [Added] GLCM texture feature percentile added

## 0.12

* [Added] Ability to supress exception if no color found (usefull for batch processing)
* [Added] Calculate perimeter of lobulus
* [Added] Set report level from GUI
* [Changed] Report levels for various output images


## 0.11

* [Added] If more control annotations for central vein are given, use them all.
* [Added] Testing for manual segmentation
* [Fixed] Manual segmentation for lobulus border only

## 0.10

* [Added] Use manual lobulus segmentation with black color
* [Fixed] Fixed problem in openslide with location idivisible by level scale

## 0.9

* [Added] Persistent columns added to all rows for description of an experiment
* [Added] Control parameters from command line
* [Added] Jaccard index calculation added to lobulus segmentation evaluation
* [Added] All application parameters are in spreadsheet now

## 0.8

* [Added] Previous location of Common spreadsheet file is used if available
* [Added] Annotation title and details are stored to .xls file

## 0.7
* [Added] Save all data also into common .xlsx file
* [Added] Computer info saved into spreadsheet
* [Added] Image raw input stored to .npz files


## 0.6

* [Added] Data file info in GUI
* [Added] GLCM texture features analysis independent on input data pixelsize
* [Added] Texture features parameters controled from GUI
* [Changed] Inner view representation allow use pixelsize independent on input data levels
* [Added] Open output directory when processing is finished
* [Added] Open output directory can be controlled from GUI
* [Changed] Lobulus segmentation pixelsize can be defined
* [Changed] Lobulus boundary and Central vein segmentation parameters updated
* [Added] Images from Lobulus boundary and Central vein segmentation
* [Added] Processing parameters saved to yaml and json file
* [Added] Collapsed parameter sub-tree with complex parameters
* [Added] Algorithm evaluation added
* [Added] Lobulus border segmentation and Central vein segmentation evaluation

## 0.4

* [Added] Create desktop icon from command line

## 0.3

* [Added] Support for Openslide on 32-bit Windows

## 0.2

* [Added] Suppression of unimportant warnings
* [Added] Black code format 
* [Added] Central vein area computation
* [Added] Dead ends number computation
* [Added] Logo added
* [Added] Icon added into window title

## 0.1

* [Added] Version number in window title
* [Added] Skeleton threshold is calculated from inner part of lobulus
* [Changed] Write images in high resolution
* [Added] Figures in mm
* [Added] Resolution and size of output images is stored in report
* [Added] Suggest output dir name by date and time
* [Added] Error message in GUI
* [Changed] Threshold level for central vein segmentation 
* [Changed] Size of view according to size of user input increased

## 0.0
* [Added] AnotatedImage class
* [Added] Read pixelsize from image
* [Added] Export annotation to json
* [Added] Convert annotation to json automatically if necessery
* [Added] Get mask from annotation
* [Added] Views can be used for looking on one annotated image
* [Added] Show tile centers in fit() and in add_training_data()
* [Added] Build by conda
* [Added] Automatic download missing dll's no Windows
* [Added] Save to excel
