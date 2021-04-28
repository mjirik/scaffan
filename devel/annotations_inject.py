import scaffan.image
from matplotlib import pyplot as plt
import numpy as np
import imma.image
from loguru import logger


anim = scaffan.image.AnnotatedImage(r"H:\biomedical\orig\Scaffold_implants\I11_S2_1\I11_S2_1.czi")
pxsz, unit = anim.get_pixel_size()

x_mm = [2, 3, 4]
y_mm = [2, 3, 2]

x_px = np.asarray(x_mm) / pxsz[0]
y_px = np.asarray(y_mm) / pxsz[1]

anim.annotations = [{"x_mm": x_mm, "y_mm": y_mm, "color": "#ff0000", "x_px": x_px, "y_px": y_px}]

logger.debug(pxsz)
# views = anim.get_views([0], pixelsize_mm = [0.002, 0.002])
views = anim.get_views([0])
view = views[0]
img = view.get_region_image(as_gray = False)
plt.imshow(img)
view.plot_annotations(0)
plt.show()
