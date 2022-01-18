from pathlib import Path
import sys
import skimage.io
import os

path_to_script = Path("~/projects/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import scaffan.image


def export_czi_annotations_to_jpg(path_annotations, annotation_name, path_images):
    """
    Creates image dataset from .czi annotations

    Parameters:
    -----------
    path_annotations : Path
                       path to .czi files
    annotation_name : str
                      wanted name of the annotations (e.g. ann, annotation, a,...)
    path_images : str
                  path to directory where the images will be saved
    """
    index = 0
    while True:
        fn_str = (
            str(path_annotations)
            + "\\"
            + annotation_name
            + str(index).zfill(4)
            + ".czi"
        )
        fn_path = Path(fn_str)
        if not fn_path.exists():
            break
        print(f"filename: {fn_path} {fn_path.exists()}")

        anim = scaffan.image.AnnotatedImage(path=fn_str)

        view = anim.get_full_view(
            pixelsize_mm=[0.0003, 0.0003]
        )  # wanted pixelsize in mm in view
        img = view.get_raster_image()
        os.makedirs(os.path.dirname(path_images), exist_ok=True)
        skimage.io.imsave(path_images + str(index).zfill(4) + ".jpg", img)
        index += 1


# Loading .czi annotations
path_annotations = Path(
    r"H:\BP\data\czi_files_validate"
)  # path to main directory, that is where .czi files are
path_images = (
    "H:\\BP\\COCO_dataset_validate\\images\\"  # path to directory, where the images will be exported
)
path_images = "H:\\BP\\COCO_dataset_train\\images\\"  # path to directory, where the images will be exported
annotation_name = "annotation"

export_czi_annotations_to_jpg(path_annotations, annotation_name, path_images)
