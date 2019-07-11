# Scaffan

Main goal of application Scaffan is quantitative description of scaffold based on
image slice. # TODO

The algorithm can be separated into two steps. First step is lobulus segmentation. 
In this step the region of segmentation. Based on user interactivity the rough 
localization of examined lobulus is known. This information is used as initial 
step for iterative segmentation algortihm. Output of this process is delineation
lobulus boundary and central vein. 

Second step is lobulus description. Input of this process is the lobulus area. 
Texture in this area is described with several methods. 
These methods were corelated with manual annotation.


## Lobulus segmentation

Initial step of lobulus segmentation is user interactivity over the image slice.
User is expected to draw closed curve around the central vein of examined lobulus.
This curve is used in several moments of processing. At first, the input image is 
cropped around input curve with margin size defined by Annotation Margin parameter 
(default vaule is 180%).
The image area should be big enough to cover whole area of lobulus. 
The resolution of image is changed to 1.82um. It is enough to keep all important 
structures visible and also keep the computation time low. 

The preprocessing step of lobulus segmentation is to use Hessian based Frangi 
filter [Frangi, Kroon] to make the 



References:
[Frangi]
A. Frangi, W. Niessen, K. Vincken, and M. Viergever. “Multiscale vessel enhancement filtering,” In LNCS, vol. 1496, pages 130-137, Germany, 1998. Springer-Verlag

[Kroon]
Kroon, D.J.: Hessian based Frangi vesselness filter.