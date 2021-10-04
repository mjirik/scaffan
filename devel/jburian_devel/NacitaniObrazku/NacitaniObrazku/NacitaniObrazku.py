# Importy
from pathlib import Path
import sys

# path_to_script =  Path("~/projects/scaffan/").expanduser()
# path_to_script =  Path("~/miniconda3/envs/scaffan/").expanduser()
# sys.path.insert(0,str(path_to_script))
import scaffan
import io3d  # just to get data
import scaffan.image as scim
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom
from czifile import CziFile
import numpy as np

# Get the data
io3d.datasets.download("SCP003", dry_run=True)
fn = io3d.datasets.join_path(
    "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
)

# Read annotated image
def get_metadata(anim, picture_path_annotation):
    with CziFile(picture_path_annotation) as czi:
        metadata = czi.metadata()
    return metadata


# Vytvoreni souboru XML s metadaty - pomocna metoda
def create_XML_file():
    metadataToFile = open("metadata.xml", "w", encoding="utf-8")
    metadataToFile.write(metadata)
    metadataToFile.close


# create_XML_file()


def load_zeiss_elements(anim, metadata, pixelSizeMM):
    root = minidom.parseString(metadata)
    elements = root.getElementsByTagName("Elements")

    listOfBeziers = []
    listOfCircles = []
    listOfRectangles = []
    listOfEllipses = []

    for j in range(len(elements)):
        for child in elements[j].childNodes:
            if child.nodeName == "Bezier":
                listOfPoints_temp = []
                points = child.getElementsByTagName("Points")[0].firstChild.nodeValue
                temp_points = points.split()
                for i in range(len(temp_points)):
                    point_X = float(temp_points[i].split(",")[0]) * pixelSizeMM[0][0]
                    point_Y = float(temp_points[i].split(",")[1]) * pixelSizeMM[0][1]

                    pointsXY_float = (point_X, point_Y)
                    listOfPoints_temp.append(pointsXY_float)
                    if i == 0:
                        lastPointsXY = pointsXY_float  # aby byla krivka spojena
                listOfPoints_temp.append(lastPointsXY)
                listOfBeziers.append(listOfPoints_temp)

            elif child.nodeName == "Circle":
                center_X = (
                    float(child.getElementsByTagName("CenterX")[0].firstChild.data)
                    * pixelSizeMM[0][0]
                )  # v mm od leveho okraje
                center_Y = (
                    float(child.getElementsByTagName("CenterY")[0].firstChild.data)
                    * pixelSizeMM[0][1]
                )  # v mm od horniho okraje obrazku
                radius = (
                    float(child.getElementsByTagName("Radius")[0].firstChild.data)
                    * pixelSizeMM[0][0]
                )
                listOfCircles.append((center_X, center_Y, radius))

            elif child.nodeName == "Rectangle":
                X_left_top = (
                    float(child.getElementsByTagName("Left")[0].firstChild.data)
                    * pixelSizeMM[0][0]
                )
                Y_left_top = (
                    float(child.getElementsByTagName("Top")[0].firstChild.data)
                    * pixelSizeMM[0][1]
                )
                width_rec = (
                    float(child.getElementsByTagName("Width")[0].firstChild.data)
                    * pixelSizeMM[0][0]
                )
                height_rec = (
                    float(child.getElementsByTagName("Height")[0].firstChild.data)
                    * pixelSizeMM[0][1]
                )
                listOfRectangles.append((X_left_top, Y_left_top, width_rec, height_rec))

            elif child.nodeName == "Ellipse":
                center_X = (
                    float(child.getElementsByTagName("CenterX")[0].firstChild.data)
                    * pixelSizeMM[0][0]
                )
                center_Y = (
                    float(child.getElementsByTagName("CenterY")[0].firstChild.data)
                    * pixelSizeMM[0][1]
                )
                radiusX = (
                    float(child.getElementsByTagName("RadiusX")[0].firstChild.data)
                    * pixelSizeMM[0][0]
                )
                radiusY = (
                    float(child.getElementsByTagName("RadiusY")[0].firstChild.data)
                    * pixelSizeMM[0][1]
                )
                listOfEllipses.append((center_X, center_Y, radiusX, radiusY))

        return listOfBeziers, listOfCircles, listOfRectangles, listOfEllipses


def insert_zeiss_annotation_bezier(anim, listOfBeziers, pixelSizeMM, *args, **kwargs):
    if len(listOfBeziers) != 0:
        anim.annotations = []
        for bezier in listOfBeziers:
            x_mm = []
            y_mm = []
            for tuple_XY in bezier:
                x_mm.append(tuple_XY[0])
                y_mm.append(tuple_XY[1])

            x_px = np.asarray(x_mm) / pixelSizeMM[0][0]
            y_px = np.asarray(y_mm) / pixelSizeMM[0][1]

            anim.annotations.append(
                {
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "color": "#ff0000",
                    "x_px": x_px,
                    "y_px": y_px,
                }
            )

            # views = anim.get_views([0], pixelsize_mm = [0.01, 0.01])
        views = anim.get_views(*args, **kwargs)  # vybiram, jakou chci zobrazit anotaci
        view = views[0]
        img = view.get_region_image(as_gray=False)
        plt.imshow(img)
        view.plot_annotations()
        plt.show()


def insert_zeiss_annotation_rectangle(anim, listOfRectangles, pixelSizeMM):
    if len(listOfRectangles) != 0:
        anim.annotations = []
        for rectangle in listOfRectangles:
            x_mm = []
            y_mm = []

            width = rectangle[2]
            height = rectangle[3]

            x_top_left = rectangle[0]
            x_top_right = x_top_left + width
            x_down_right = x_top_left
            x_down_left = x_top_right

            x_param_rec = (
                x_top_left,
                x_top_right,
                x_down_left,
                x_down_right,
                x_top_left,
            )

            y_top_left = rectangle[1]
            y_top_right = y_top_left
            y_down_right = y_top_left + height
            y_down_left = y_down_right

            y_param_rec = (
                y_top_left,
                y_top_right,
                y_down_left,
                y_down_right,
                y_top_left,
            )

            for paramX_mm_rec in x_param_rec:
                x_mm.append(paramX_mm_rec)

            for paramY_mm_rec in y_param_rec:
                y_mm.append(paramY_mm_rec)

            x_px = np.asarray(x_mm) / pixelSizeMM[0][0]
            y_px = np.asarray(y_mm) / pixelSizeMM[0][1]

            anim.annotations.append(
                {
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "color": "#ff0000",
                    "x_px": x_px,
                    "y_px": y_px,
                }
            )
            # views = anim.get_views([0], pixelsize_mm = [0.01, 0.01])
        views = anim.get_views([0])
        view = views[0]
        img = view.get_region_image(as_gray=False)
        plt.imshow(img)
        view.plot_annotations()
        plt.show()


# Nacteni elementu ze souboru XML -> elementy ulozene do prislusnych listu
# picture_path = Path(r"G:\.shortcut-targets-by-id\18TvIZlK5UywpgyOLb6xGOA07h2ZZesN7\Scaffold_implants\I13_S1_1\I13_S1_1.czi")
picture_path_annotation = Path(r"H:\zeiss_test_shapes.czi")
# fn = io3d.datasets.join_path(picture_path, get_root=True)
# fn = picture_path
fn = picture_path_annotation
print(f"filename: {fn} {fn.exists()}")

anim = scim.AnnotatedImage(str(fn))

metadata = get_metadata(anim, picture_path_annotation)
pixelSizeMM = anim.get_pixel_size()
listOfBeziers, listOfCircles, listOfRectangles, listOfEllipses = load_zeiss_elements(
    anim, metadata, pixelSizeMM
)

# Zobrazeni anotaci - bezier a obdelnik
# annotationNumber = 2 # -> zavisle na delce listOfBeziers
insert_zeiss_annotation_bezier(anim, listOfBeziers, pixelSizeMM, [1], margin=2.0)
insert_zeiss_annotation_rectangle(anim, listOfRectangles, pixelSizeMM)


# Get grayscale image by center
pixelsize_mm = [0.005, 0.005]

view = anim.get_view(center=(25000, 10000), size_mm=[1, 1], pixelsize_mm=pixelsize_mm)
img = view.get_region_image(as_gray=False)
plt.imshow(img)
plt.show()
# view = anim.get_views(ann_ids, pixelsize_mm=pixelsize_mm)[0]


# Get grayscale image by annotation color
ann_ids = anim.select_annotations_by_color("#000000")
print(ann_ids)
view = anim.get_views(ann_ids, level=7)[0]
img = view.get_region_image(as_gray=True)
plt.imshow(img, cmap="gray")
plt.show()

# Change one view into other view
detail_view = view.to_pixelsize(pixelsize_mm=[0.005, 0.005])
img = detail_view.get_region_image(as_gray=True)
plt.imshow(img, cmap="gray")
plt.show()

# Annotations
# ann_ids = anim.select_annotations_by_color("#000000")
ann_ids = anim.select_annotations_by_title("raindrop")
print(ann_ids)
view = anim.get_views(ann_ids, level=7)[0]

# Show annotation
img = view.get_region_image(as_gray=True)

plt.imshow(img, cmap="gray")
view.plot_annotations(ann_ids[0])
plt.show()
