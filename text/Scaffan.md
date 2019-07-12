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
filter [Frangi](#frangi), [Kroon](#kroon) to supress the high-frequency signal 
in lobulus area and to make the lobulus border more dominant. Then a method 
Morphological Active Contours without Edges described in (Casselles)[#casselles], 
(Neila)[#neila] and (Avarez)[#avarez] is used to extract outer boundary of liver
lobulus. 

Due to lower contrast the segmentation of Central Vein is more complex task. 
We use the texture analysis to detect roughly the area of Central Vein. 
Gradient magnitude filtration using Gaussian derivatives combined with the 
Frangi filter is used to have more details of the shape of Central Vein.






# References:

## Frangi
A. Frangi, W. Niessen, K. Vincken, and M. Viergever. “Multiscale vessel enhancement filtering,” In LNCS, vol. 1496, pages 130-137, Germany, 1998. Springer-Verlag

## Kroon
Kroon, D.J.: Hessian based Frangi vesselness filter.


## Neila
A Morphological Approach to Curvature-based Evolution of Curves and Surfaces, Pablo Márquez-Neila, Luis Baumela and Luis Álvarez. In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 2014, DOI 10.1109/TPAMI.2013.106

## Alvarez
Morphological Snakes. Luis Álvarez, Luis Baumela, Pablo Márquez-Neila. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2010 (CVPR10).

## Casselles
Geodesic Active Contours, Vicent Caselles, Ron Kimmel and Guillermo Sapiro. In International Journal of Computer Vision (IJCV), 1997, DOI:10.1023/A:1007979827043

## Chan
Active Contours without Edges, Tony Chan and Luminita Vese. In IEEE Transactions on Image Processing, 2001, DOI:10.1109/83.902291