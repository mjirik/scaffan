from pathlib import Path
import sys
import skimage.io
path_to_script = Path("~/projects/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import scaffan.image

def export_czi_annotations_to_jpg(path_annotations, annotation_name, path_images):
    index = 0
    while True:
        fn_str = str(path_annotations) + "\\" + annotation_name + str(index) + ".czi"
        fn_path = Path(fn_str)
        if not fn_path.exists():
            break
        print(f"filename: {fn_path} {fn_path.exists()}")

        anim = scaffan.image.AnnotatedImage(path=fn_str)

        view = anim.get_full_view(pixelsize_mm=[0.0003, 0.0003]) # wanted pixelsize in mm in view
        img = view.get_raster_image()
        Path(path_images).mkdir(parents=True, exist_ok=True)
        skimage.io.imsave(path_images + str(index) + ".jpg", img)
        index += 1


# Loading .czi annotations
path_annotations = Path(r"H:\zeiss_export_json") # path to main directory, that is where .czi files are
path_images = "H:\\dataset\\" # path to directory, where the images will be exported
annotation_name = "annotation"

export_czi_annotations_to_jpg(path_annotations, annotation_name, path_images)