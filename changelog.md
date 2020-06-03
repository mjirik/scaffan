# Changelog

# 0.24 

* [Added] More debug options
* [Changed] Increased efectivity during training step by removing prediction
* [Changed] Reset parameters after training

# 0.23

* [Added] Manual lobuli selection from GUI added
* [Added] Annotations from ImageJ are readed. The file with annotations is expected to be in the same dir and same name
 with file extension `.roi.zip`. Color of segmentation can be part of the polyon name ("name #00FF00")
 
# 0.22

* [Fixed] Automatic selection of multiple points in the same lobule
* [Changed] Margin calculation is available not only for views given by annotation.
* [Added] Zeiss `.czi` image format support added
* [Added] Save training labels image from input annotation
* [Added] Get all annotation raster with the same color in view
* [Added] Tiff file format added into reader

# 0.21

* [Added] SNI prediction based on texture analysis added
* [Added] Important packages version info added to report
* [Added] Repository commit identifier added to report
* [Changed] Automatic lobulus segmentation is default now

# 0.20

* [Changed] View implementation slightly changed. All `get_view(s)` functions are now more consistent.
* [Added] Whole scan segmentation based on U-Net

# 0.18

* [Added] Intensity normalization of input data

# 0.17

* [Added] Whole scan Classifier description to output spreadsheet
* [Added] Training weight added in whole scan analysis
* [Added] All parameters stored also if only whole scan analysis is performed

# 0.16 

* [Fixed] Get inner and outer annotation. Fixed call without color.
* [Added] Get just outer annotation and find the holes.
* [Added] Scan segmentation to calculate intra-lobular and extra-lobular surface area
* [Added] Fixed reading file with no annotation
* [Added] GLCM texture features used for whole scan segmentation
* [Changed] GUI update

# 0.15

* [Added] Get view by center in mm

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
