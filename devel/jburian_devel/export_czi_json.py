import datetime
from pathlib import Path
import sys
import skimage.io
import numpy as np
import math
import matplotlib.pyplot as plt

path_to_script = Path("~/projects/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import scaffan.image
from datetime import date


def get_image_properties(dataset_directory):
    image_name = 0
    list_image_dictionaries = []
    image_name_id = 0

    while True:
        filename_string = (
            str(dataset_directory) + "\\" + str(image_name).zfill(4) + ".jpg"
        )
        filename_path = Path(filename_string)
        if not filename_path.exists():
            break
        image = skimage.io.imread(filename_string)
        height = image.shape[0]
        width = image.shape[1]

        image_dictionary = {
            "id": image_name_id,
            "width": width,
            "height": height,
            "file_name": str(image_name).zfill(4) + ".jpg",
        }
        list_image_dictionaries.append(image_dictionary)

        image_name += 1
        image_name_id += 1
    return list_image_dictionaries


def get_category_properties(dataset_directory, filename):
    list_category_dictionaries = []
    with open(str(dataset_directory) + "\\" + filename) as f:
        lines = f.readlines()
    number_lines = len(lines)

    for i in range(number_lines):
        supercategory = lines[i].rstrip()
        category_dictionary = {
            "supercategory": supercategory,
            "id": i + 1,
            "name": supercategory,
        }
        list_category_dictionaries.append(category_dictionary)

    return list_category_dictionaries


def count_polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_annotations_properties(czi_files_directory, annotation_name):
    index = 0
    annotation_id = 1
    category_id = 1  # only one category - cells
    image_id = 0

    list_annotation_dictionaries = []

    while True:
        filename_string = (
            str(czi_files_directory)
            + "\\"
            + annotation_name
            + str(index).zfill(4)
            + ".czi"
        )
        filename_path = Path(filename_string)
        if not filename_path.exists():
            break

        anim = scaffan.image.AnnotatedImage(path=str(filename_path))
        view = anim.get_full_view(
            pixelsize_mm=[0.0003, 0.0003]
        )  # wanted pixelsize in mm in view
        annotations = view.annotations

        """ 
        img = view.get_raster_image()
        view.plot_annotations()
        plt.imshow(img)
        plt.show()
        """

        for j in range(len(annotations)):
            xy_px_list = []

            x_px_list = annotations[j]["x_px"].tolist()
            y_px_list = annotations[j]["y_px"].tolist()

            x_px_min = float(math.floor(np.min(x_px_list)))
            y_px_min = float(math.floor(np.min(y_px_list)))
            width = float(math.floor(np.max(x_px_list)) - x_px_min)
            height = float(math.floor(np.max(y_px_list)) - y_px_min)

            bbox = [x_px_min, y_px_min, width, height]

            # polygon_area = count_polygon_area(np.array(x_px_list) * 0.0003, np.array(y_px_list) * 0.0003) # in mm
            polygon_area = count_polygon_area(
                np.array(x_px_list), np.array(y_px_list)
            )  # in pixels
            x_px_list = np.asarray(x_px_list)
            for i in range(len(x_px_list)):
                xy_px_list.append(x_px_list[i])
                xy_px_list.append(y_px_list[i])

            segmentation = xy_px_list

            # segmentation[0::2]
            # segmentation[1::2]
            # np.max(x_px_list)

            annotation_dictionary = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [segmentation],  # RLE or [polygon]
                "area": polygon_area,  # Shoelace formula
                "bbox": bbox,  # [x, y, width, height]
                "iscrowd": 0,
            }
            annotation_id += 1

            list_annotation_dictionaries.append(annotation_dictionary)

        image_id += 1
        index += 1

    return list_annotation_dictionaries


# Vytvoreni souboru .json
import json

# .json structure
"""
data = {    
    "info" : info, 
    "images": [image],
    "categories": [],
    "annotations": [annotation]
    "licenses": [license]
}
"""

data = {}
info_dictionary = {
    "year": str(date.today().year),
    "version": 1.0,
    "description": "COCO dataset for scaffan",
    "contributor": "Jan Burian",
    "date_created": date.today().strftime("%d/%m/%Y"),
}
data.update({"info": info_dictionary})

dataset_directory = Path(r"H:\COCO_dataset\images")

list_image_dictionaries = get_image_properties(dataset_directory)
data.update({"images": list_image_dictionaries})

list_category_dictionaries = get_category_properties(
    dataset_directory, "categories.txt"
)  # directory and .txt file
data.update({"categories": list_category_dictionaries})

czi_files_directory = Path(r"H:\zeiss_export_json")  # path to .czi files directory
annotation_name = "annotation"

list_annotation_dictionaries = get_annotations_properties(
    czi_files_directory, annotation_name
)
data.update({"annotations": list_annotation_dictionaries})

# Creating COCO format
path_json = "H:\\COCO_dataset"  # path to directory, where the images will be exported
# Creating .json file
with open(path_json + "\\" + "trainval.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
