import matplotlib
import matplotlib.pyplot as plt
import scaffan.image as scim
import pandas as pd
import cv2
import numpy as np
from scaffan import lobulus
import tkinter
import shutil
import exsu
from pathlib import Path
import matplotlib.patches as patches
import re

FILE_NAME = "PIG-002_J-18-0091_HE.ndpi"
FILE_PATH = "D:\\FAV\\Scaffold\\Scaffan-analysis\\"

EXCEL_NAME = "new_parameter_values_HOM.xlsx"
EXCEL_PATH = "D:\\FAV\\Scaffold\\homogeneity\\"

DISPLAY_SIZE = 80

TRAIN_DATA_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\train_data.npy"
LABELS_SAVE_FILE = "D:\\FAV\\Scaffold\\data\\labels.npy"

CUT_SIZE = 0.2  # Size of training data [mm]
STEPS_PER_CUT = 4


def get_annotation_ids_for_file(source_file_name, excel_df):
    anotation_ids = excel_df.loc[
        (excel_df["File Name"] == source_file_name), ["Annotation ID"]
    ].values[:, 0]

    return anotation_ids


def get_homogenity(source_fname, excel_df, ann_id):
    homo = excel_df.loc[
        (excel_df["File Name"] == source_fname) & (excel_df["Annotation ID"] == ann_id),
        ["HOM"],
    ]
    return homo.values[0][0]


def my_plot(img, ann_id, hom, details):
    img = cv2.resize(
        img, dsize=(DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_CUBIC
    )
    plt.figure()
    plt.title("ID: " + str(ann_id) + ", HOM: " + str(hom) + ", " + details)
    plt.imshow(img, cmap="gray")


def get_all_filenames(excel_df):
    return set(excel_df["File Name"].tolist())


def shrink_image(img):
    return cv2.resize(
        img, dsize=(DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_CUBIC
    )


def cut_the_image(lobul):
    """
    Returns sequence of square images (in form of 3D ndarray) cut out of image from input lobulus.
    """

    if lobul.view.region_pixelunit is not "mm":
        raise Exception("Algorithm is implemented only for region_pixelunit = mm")

    if lobul.view.region_pixelsize[0] != lobul.view.region_pixelsize[1]:
        raise Exception("pixelsize is not the same in both x and y axis")

    pixel_size = lobul.view.region_pixelsize[0]
    cut_pixel_size = int((1 / pixel_size) * CUT_SIZE)
    step_size = int(cut_pixel_size / STEPS_PER_CUT)

    cuts_list = []

    x = 0

    while x < lobul.image.shape[1]:
        skip_next_line = False

        y = 0

        while y < lobul.image.shape[0]:
            if does_the_square_fit(lobul.lobulus_mask, x, y, step_size):
                cuts_list.append([y, x])
                y = y + cut_pixel_size
                skip_next_line = True
            else:
                y = y + step_size

        if skip_next_line:
            x = x + cut_pixel_size
        else:
            x = x + step_size

    return cuts_list


def does_the_square_fit(mask, x_start, y_start, step_size) -> bool:
    if mask.shape[0] <= x_start + STEPS_PER_CUT * step_size:
        return False

    if mask.shape[1] <= y_start + STEPS_PER_CUT * step_size:
        return False

    for i in range(STEPS_PER_CUT + 1):
        x = x_start + i * step_size
        y = y_start
        if not mask[x, y]:
            return False

    for i in range(STEPS_PER_CUT + 1):
        x = x_start + STEPS_PER_CUT * step_size
        y = y_start + i * step_size
        if not mask[x, y]:
            return False

    for i in range(STEPS_PER_CUT + 1):
        x = x_start + i * step_size
        y = y_start + STEPS_PER_CUT * step_size
        if not mask[x, y]:
            return False

    for i in range(STEPS_PER_CUT + 1):
        x = x_start
        y = y_start + i * step_size
        if not mask[x, y]:
            return False

    return True


def load_lobulus(anim, annotation_id):
    output_dir = Path("test_output/test_lobulus_mask_output_dir").absolute()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    report = exsu.Report(outputdir=output_dir, show=False)
    lob_proc = lobulus.Lobulus(report=report)
    lob_proc.set_annotated_image_and_id(anim, annotation_id)
    lob_proc.run()
    return lob_proc


# """visual"""
# if __name__ == '__main__':
#     excel_df = pd.read_excel(EXCEL_PATH + EXCEL_NAME)
#     anim = scim.AnnotatedImage(FILE_PATH + FILE_NAME)
#     annotations = get_annotation_ids_for_file(FILE_NAME, excel_df)
#     lobulus = load_lobulus(anim, annotations[5])
#
#     cuts = cut_the_image(lobulus)
#
#     matplotlib.use('TkAgg')
#     fig, ax = plt.subplots(1)
#     ax.imshow(lobulus.lobulus_mask.astype(int))
#
#     for point in cuts:
#         # plt.plot(point[1], point[0], 'o', color='black')
#         rect = patches.Rectangle((point[0], point[1]), int((1/lobulus.view.region_pixelsize[0]) * CUT_SIZE), int((1/lobulus.view.region_pixelsize[0]) * CUT_SIZE), linewidth=1, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#
#     plt.show()
#
#     print(cuts)

if __name__ == "__main__":
    excel_df = pd.read_excel(EXCEL_PATH + EXCEL_NAME)
    anim = scim.AnnotatedImage(FILE_PATH + FILE_NAME)
    annotations = get_annotation_ids_for_file(FILE_NAME, excel_df)

    train_data = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 0))  # cropped images
    labels = np.zeros((2, 0))  # [SCI, HOM]

    for ann in annotations:
        lobul = load_lobulus(anim, ann)
        cuts = cut_the_image(lobul)
        crop_size = int((1 / lobul.view.region_pixelsize[0]) * CUT_SIZE)

        hom = get_homogenity(FILE_NAME, excel_df, ann)
        details = anim.annotations[ann]["details"]
        sni = float(re.match(r".*SNI=(\d*\.?\d*)", details).group(1)) / 2

        for cut_point in cuts:
            image_crop = lobul.image[
                cut_point[0] : cut_point[0] + crop_size,
                cut_point[1] : cut_point[1] + crop_size,
            ]

            if image_crop.shape[0] < DISPLAY_SIZE:
                print("WARNING: Cropped image is smaller than display size.")

            image_crop = shrink_image(image_crop)

            train_data = np.concatenate(
                (train_data, image_crop.reshape(DISPLAY_SIZE, DISPLAY_SIZE, 1)), axis=2
            )
            labels = np.concatenate(
                (labels, np.asarray([hom, sni]).reshape(2, 1)), axis=1
            )

    np.save(TRAIN_DATA_SAVE_FILE, train_data)
    np.save(LABELS_SAVE_FILE, labels)
