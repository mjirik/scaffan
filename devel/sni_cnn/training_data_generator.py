import os.path
import re
import shutil
from pathlib import Path

import cv2
import exsu
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import scaffan.scaffan.image as scim
from scaffan.scaffan import lobulus
import json
import pickle as plk
from image_with_mask import ImageWithMask

FILE_NAME = "PIG-002_J-18-0091_HE.ndpi"
FILE_PATH = 'D:\\FAV\\Scaffold\\Scaffan-analysis\\'

EXCEL_NAME = "parameter_table_SNI_HOM.xlsx"
EXCEL_PATH = 'D:\\FAV\\Scaffold\\homogeneity\\'



DISPLAY_SIZE = 80

TRAIN_DATA_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\train_data.npy'
TRAIN_LABELS_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\train_labels.npy'
TEST_DATA_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\test_data.npy'
TEST_LABELS_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\test_labels.npy'
LOBULUS_DATA_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\lobulus_data.npy'
LOBULUS_LABELS_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\lobulus_labels.npy'

MASK_MAP_PATH = 'D:\\FAV\\Scaffold\\data\\mask_map.json'
MASKS_PATH = 'D:\\FAV\\Scaffold\\data\\masks.plk'
IMAGES_PATH = 'D:\\FAV\\Scaffold\\data\\images.plk'

CUT_SIZE = 0.2  # Size of training data [mm]
STEPS_PER_CUT = 4


def get_annotation_ids_for_file(source_file_name, excel_df):
    anotation_ids = excel_df.loc[(excel_df['File Name'] == source_file_name), ['Annotation ID']].values[:, 0]

    return anotation_ids


def get_homogenity(source_fname, excel_df, ann_id):
    homo = excel_df.loc[(excel_df['File Name'] == source_fname) & (excel_df['Annotation ID'] == ann_id), ['HOM']]
    return homo.values[0][0]

def get_sni(source_fname, excel_df, ann_id):
    sni = excel_df.loc[(excel_df['File Name'] == source_fname) & (excel_df['Annotation ID'] == ann_id), ['SNI']]
    return sni.values[0][0]


def my_plot(img, ann_id, hom, details):
    img = cv2.resize(img, dsize=(DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_CUBIC)
    plt.figure()
    plt.title('ID: ' + str(ann_id) + ', HOM: ' + str(hom) + ', ' + details)
    plt.imshow(img, cmap='gray')


def get_all_filenames(excel_df):
    return enumerate(set(excel_df['File Name'].tolist()))


def shrink_image(img):
    return cv2.resize(img, dsize=(DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_CUBIC)


def cut_the_image(mask_img, overlap=True):
    cut_pixel_size = int((1 / mask_img.pixel_size) * CUT_SIZE)
    step_size = int(cut_pixel_size / STEPS_PER_CUT)

    cuts_list = []

    x = 0

    while x < mask_img.image.shape[1]:
        skip_next_line = False

        y = 0

        while y < mask_img.image.shape[0]:
            if does_the_square_fit(mask_img.mask, x, y, step_size):
                cuts_list.append([y, x])

                if overlap:
                    y = y + step_size
                else:
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


def remove_images_with_0_hom(data, labels):
    data_count = labels.shape[-1]

    i = 0

    while i < data_count:
        hom = labels[0, i]
        if hom < 0.01:
            data = np.delete(data, i, -1)
            labels = np.delete(labels, i, -1)
            data_count = data_count - 1
        else:
            i = i + 1


    return data, labels

def shufle_data(data, labels):
    data_count = labels.shape[-1]
    rand_permut = np.random.permutation(data_count)

    return data[:, :, rand_permut], labels[:, rand_permut]


def split_train_test(data, labels, test_portion=0.1):
    assert 1 > test_portion > 0
    data_count = labels.shape[-1]
    split_index = round((1-test_portion)*data_count)

    train_data = data[:, :, 0:split_index]
    test_data = data[:, :, split_index:data_count]
    train_labels = labels[:, 0:split_index]
    test_labels = labels[:, split_index:data_count]

    return train_data, test_data, train_labels, test_labels


def extend_data_rotation(data, labels):
    data_rotated = np.rot90(data, k=1, axes=(0, 1))  # 90 degrees
    data = np.concatenate((data, data_rotated), axis=2)

    data_rotated = np.rot90(data_rotated, k=1, axes=(0, 1))  # 180 degrees
    data = np.concatenate((data, data_rotated), axis=2)
    data_rotated = np.rot90(data_rotated, k=1, axes=(0, 1))  # 270 degrees
    data = np.concatenate((data, data_rotated), axis=2)

    labels = np.concatenate((labels, labels, labels, labels), axis=1)

    return data, labels


def extend_data_blur(data, labels):
    data_blurred = gaussian_filter(data, sigma=1)
    data = np.concatenate((data, data_blurred), axis=2)

    labels = np.concatenate((labels, labels), axis=1)

    return data, labels

def create_lobulus_dataset(mask_imgs, mask_map, excel_df, test_ratio):
    train_data = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 0))  # cropped images
    train_labels = np.zeros((2, 0))  # [SCI, HOM]
    test_data = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 0))  # cropped images
    test_labels = np.zeros((2, 0))  # [SCI, HOM]

    for i in range(len(mask_imgs)):
        mask_img = mask_imgs[i]
        hom = get_homogenity(mask_img.file_name, excel_df, mask_img.ann_id)
        sni = get_sni(mask_img.file_name, excel_df, mask_img.ann_id) / 2

        if not sni or not (sni >= 0 and sni <=1 ):
            print('Skipping data with missing SNI')
            continue

        if hom < 0.01:
            print('Skipping data with homogeneity < 0.01')
            continue

        cuts = cut_the_image(mask_img)
        crop_size = int((1 / mask_img.pixel_size) * CUT_SIZE)

        for cut_point in cuts:

            image_crop = mask_img.image[cut_point[1]:cut_point[1] + crop_size, cut_point[0]:cut_point[0] + crop_size]

            if image_crop.shape[0] < DISPLAY_SIZE:
                print('Skipping cropped image which is smaller than display size.')
                # Throw away images with too low resolution
                continue

            image_crop = shrink_image(image_crop)

            if i % test_ratio == 0:
                test_data = np.append(test_data, image_crop.reshape(DISPLAY_SIZE, DISPLAY_SIZE, 1), axis=2)
                test_labels = np.append(test_labels, np.asarray([hom, sni]).reshape(2, 1), axis=1)
            else:
                train_data = np.append(train_data, image_crop.reshape(DISPLAY_SIZE, DISPLAY_SIZE, 1), axis=2)
                train_labels = np.append(train_labels, np.asarray([hom, sni]).reshape(2, 1), axis=1)


    return train_data, train_labels, test_data, test_labels

def visual_1(lobulus):
    cuts = cut_the_image(lobulus)

    fig, ax = plt.subplots(1)
    ax.imshow(lobulus.lobulus_mask.astype(float)*lobulus.image)

    for point in cuts:
        rect = patches.Rectangle((point[0], point[1]), int((1 / lobulus.view.region_pixelsize[0]) * CUT_SIZE),
                                 int((1 / lobulus.view.region_pixelsize[0]) * CUT_SIZE), linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    plt.show()

    print(cuts)

def save_lobulus_masks(excel_path, ndpi_directory):
    mask_imgs = []
    mask_index = 0
    mask_map = dict()

    excel_df = pd.read_excel(excel_path)

    file_names = get_all_filenames(excel_df)

    for i, file_name in file_names:

        if not os.path.exists(ndpi_directory + file_name):
            continue

        mask_map[file_name] = dict()

        anim = scim.AnnotatedImage(ndpi_directory + file_name)
        annotations = get_annotation_ids_for_file(file_name, excel_df)

        for ann in annotations:
            mask_map[file_name][str(ann)] = mask_index

            lobulus = load_lobulus(anim, ann)

            mask_imgs.append(ImageWithMask(lobulus.image, lobulus.lobulus_mask, file_name, ann, lobulus.view.region_pixelsize[0]))

            mask_index = mask_index + 1

    with open(MASK_MAP_PATH, 'w') as fp:
        json.dump(mask_map, fp)

    plk.dump(mask_imgs, open(MASKS_PATH, 'wb'))



def create_training_data(excel_path, ndpi_directory):
    """
    Returns: training and testing data sets.
    Params: excel_path
            - Path to excel file containing homogeneity-annotation mapping
            ndpi_directory
            - Path to directory containing .ndpi files with annotated images

    Note: Each phase is independent and could be executed separately.
            - This can be useful since the "Phase 1" is very time consuming if the lobulus-masks are not created yet.
    """
    # TODO: HOM and SNI should be separated: HOM as sample_weights and SNI as labels - storing in one tensor named
    #  "labels" is not well-arranged.

    # Phase 1: extract data from excel and .ndpi files using lobulus mask
    if os.path.exists(MASKS_PATH) and os.path.exists(MASK_MAP_PATH):
        mask_imgs = plk.load(open(MASKS_PATH, 'rb'))
        with open(MASK_MAP_PATH, 'r') as f:
            mask_map = json.load(f)
    else:
        print('No data found. Create lobulus masks started.')
        save_lobulus_masks(EXCEL_PATH + EXCEL_NAME, ndpi_directory)
        return

    excel_df = pd.read_excel(excel_path)
    train_data, train_labels, test_data, test_labels = create_lobulus_dataset(mask_imgs, mask_map, excel_df, 10)
    # np.save(LOBULUS_DATA_SAVE_FILE, data)
    # np.save(LOBULUS_LABELS_SAVE_FILE, labels)

    # Phase 2: Shuffle data
    train_data, train_labels = shufle_data(train_data, train_labels)
    test_data, test_labels = shufle_data(test_data, test_labels)

    # Phase 3: Extend the data set by images rotated 90, 180 and 270 degrees
    train_data, train_labels = extend_data_rotation(train_data, train_labels)
    test_data, test_labels = extend_data_rotation(test_data, test_labels)

    # # Phase 4: Extend the data set by blurred images
    # train_data, train_labels = extend_data_blur(train_data, train_labels)

    # Phase 5: Shuffle train data again
    train_data, train_labels = shufle_data(train_data, train_labels)
    test_data, test_labels = shufle_data(test_data, test_labels)

    return train_data, test_data, train_labels, test_labels



if __name__ == '__main__':
    # matplotlib problem fix
    matplotlib.use('TkAgg')
    train_data, test_data, train_labels, test_labels = create_training_data(EXCEL_PATH + EXCEL_NAME, FILE_PATH)

    np.save(TRAIN_DATA_SAVE_FILE, train_data)
    np.save(TRAIN_LABELS_SAVE_FILE, train_labels)
    np.save(TEST_DATA_SAVE_FILE, test_data)
    np.save(TEST_LABELS_SAVE_FILE, test_labels)
