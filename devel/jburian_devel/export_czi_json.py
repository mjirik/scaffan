from pathlib import Path
import sys

path_to_script = Path("~/projects/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import matplotlib.pyplot as plt
import scaffan.image
import io3d


# Loading .czi annotations TODO: Add loading of multiple annotations
path_annotation = Path(r"H:\zeiss_export_json\annotation1.czi")
# picture_path_annotation_string = str(Path(r"H:\zeiss_export_json\annotation1.czi"))
fn = path_annotation
print(f"filename: {fn} {fn.exists()}")

anim = scaffan.image.AnnotatedImage(path=str(path_annotation))
print(anim.annotations)
