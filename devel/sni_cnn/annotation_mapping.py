import matplotlib
import matplotlib.pyplot as plt
import scaffan.image as scim
import pandas as pd
import cv2
import numpy as np
from scaffan import lobulus
import tkinter

FILE_NAME = "PIG-002_J-18-0091_HE.ndpi"
FILE_PATH = 'D:\\FAV\\Scaffold\\Scaffan-analysis\\'

EXCEL_NAME = "new_parameter_values_HOM.xlsx"
EXCEL_PATH = 'D:\\FAV\\Scaffold\\homogeneity\\'

DISPLAY_SIZE = 300

IMAGES_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\images.npy'
HOMOS_SAVE_FILE = 'D:\\FAV\\Scaffold\\data\\homos.npy'

LABEL_MAPPING = {
    0.00: np.asarray([1, 0, 0, 0, 0]).reshape(5, 1),
    0.25: np.asarray([0, 1, 0, 0, 0]).reshape(5, 1),
    0.50: np.asarray([0, 0, 1, 0, 0]).reshape(5, 1),
    0.75: np.asarray([0, 0, 0, 1, 0]).reshape(5, 1),
    1.00: np.asarray([0, 0, 0, 0, 1]).reshape(5, 1)
}


def get_annotation_ids_for_file(source_file_name, excel_df):
    anotation_ids = excel_df.loc[(excel_df['File Name'] == source_file_name), ['Annotation ID']].values[:, 0]

    return anotation_ids


def get_homogenity(source_fname, excel_df, ann_id):
    homo = excel_df.loc[(excel_df['File Name'] == source_fname) & (excel_df['Annotation ID'] == ann_id), ['HOM']]
    return homo.values[0][0]


def my_plot(img, ann_id, hom, details):
    img = cv2.resize(img, dsize=(DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_CUBIC)
    plt.figure()
    plt.title('ID: ' + str(ann_id) + ', HOM: ' + str(hom) + ', ' + details)
    plt.imshow(img, cmap='gray')


def tensorify_views(excel_path):
    excel_df = pd.read_excel(excel_path)
    file_names = get_all_filenames(excel_df)

    tensor = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 0))
    homos = np.zeros((5, 0))

    for fname in file_names:
        anim = scim.AnnotatedImage(FILE_PATH + FILE_NAME)
        ann_ids = get_annotation_ids_for_file(FILE_NAME, excel_df)
        views = anim.get_views(ann_ids)
        for index, view in enumerate(views):
            ann_id = ann_ids[index]
            hom = get_homogenity(FILE_NAME, excel_df, ann_id)
            img = shrink_image(view.get_region_image(as_gray=True)).reshape((DISPLAY_SIZE, DISPLAY_SIZE, 1))
            tensor = np.concatenate((tensor, img), axis=2)
            homos = np.concatenate((homos, LABEL_MAPPING[hom]), axis=1)

    return tensor, homos


def get_all_filenames(excel_df):
    return set(excel_df['File Name'].tolist())


def shrink_image(img):
    return cv2.resize(img, dsize=(DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_CUBIC)

def test_get_lobulus_mask():
    # this is hack to fix the problem with non existing report - not useful anymore
    #
    import shutil
    import exsu
    from pathlib import Path
    output_dir = Path("test_output/test_lobulus_mask_output_dir").absolute()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    report = exsu.Report(outputdir=output_dir, show=False)


    anim = scim.AnnotatedImage(FILE_PATH + FILE_NAME)
    anns = anim.select_annotations_by_color("#0000FF")

    # report = None
    lob_proc = lobulus.Lobulus(report=report)
    lob_proc.set_annotated_image_and_id(anim, anns[0])
    lob_proc.run()
    # there are several useful masks
    #
    # lob_proc.annotation_mask
    # lob_proc.lobulus_mask
    # lob_proc.central_vein_mask
    # lob_proc.border_mask

    matplotlib.use('TkAgg')
    plt.figure()
    plt.imshow(lob_proc.lobulus_mask.astype(int))
    plt.show()

    # this is for testing
    assert np.sum(lob_proc.annotation_mask) > 100, "segmentation should have more than 100 px"
    assert np.sum(lob_proc.lobulus_mask) > 100, "segmentation should have more than 100 px"
    assert np.sum(lob_proc.central_vein_mask) > 0, "segmentation should have more than 0 px"
    assert np.sum(lob_proc.annotation_mask) < np.sum(lob_proc.lobulus_mask), "annotation should be smaller than lobulus"

# """SAVE TENSOR DATA"""
# if __name__ == '__main__':
#     images_tensor, homos_vector = tensorify_views(EXCEL_PATH + EXCEL_NAME)
#     np.save(IMAGES_SAVE_FILE, images_tensor)
#     np.save(HOMOS_SAVE_FILE, homos_vector)
#
#     np.savez_compressed(IMAGES_SAVE_FILE, images_tensor)

"""DATA MAPPING VISUALISATION"""
if __name__ == '__main__':
    # Load data from source files
    anim = scim.AnnotatedImage(FILE_PATH + FILE_NAME)
    excel_df = pd.read_excel(EXCEL_PATH + EXCEL_NAME)

    # Select data with information about homogeneity
    ann_ids = get_annotation_ids_for_file(FILE_NAME, excel_df)
    views = anim.get_views(ann_ids)

    matplotlib.use('TkAgg')

    # View information about the data
    for index, view in enumerate(views):
        ann_id = ann_ids[index]
        hom = get_homogenity(FILE_NAME, excel_df, ann_id)
        img = view.get_region_image(as_gray=True)
        my_plot(img, ann_id, hom, anim.annotations[ann_id]['details'])

    plt.show()


# if __name__ == '__main__':
#     test_get_lobulus_mask()

