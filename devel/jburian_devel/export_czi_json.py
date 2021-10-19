import datetime
from pathlib import Path
import sys
import skimage.io

path_to_script = Path("~/projects/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import matplotlib.pyplot as plt
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


def get_annotations_properties(
    czi_files_directory, annotation_name
):  # TODO: add bbbox and area if needed
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

        for j in range(len(annotations)):
            xy_px_list = []
            x_px_list = annotations[j]["x_px"].tolist()
            y_px_list = annotations[j]["y_px"].tolist()

            for i in range(len(x_px_list)):
                xy_px_list.append(x_px_list[i])
                xy_px_list.append(y_px_list[i])

            segmentation = xy_px_list

            annotation_dictionary = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [segmentation],  # RLE or [polygon]
                "area": 1234,  # prozatimni hodnota; Shoelace formula
                "bbox": [],  # [x, y, width, height] # je treba?
                "iscrowd": 0,  # prozatimni hodnota; 0 nebo 1?
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

dataset_directory = Path(r"H:\dataset")

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

# Creating .json file
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
