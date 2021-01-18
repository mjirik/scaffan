"""Script for manual dataset creation"""
import tkinter as tk
from tkinter import filedialog

import numpy as np
import sed3

import scaffan.image as scim


def zoom_img(img, stride=10):
    """Use sed3 tool to manually zoom to wanted image region.

    parameter-stride: for image view (Displaying image in full resolution is both resource and time consuming.)
    """
    # show sed3 tool for region selection
    ed = sed3.sed3(img[::stride, ::stride, :])
    ed.show()

    # save user selected region
    nzx, nzy, nzz = np.nonzero(ed.seeds)

    # crop original image
    img = img[
          np.min(nzx) * stride:np.max(nzx) * stride,
          np.min(nzy) * stride:np.max(nzy) * stride,
          :
          ]
    return img


def annotate(img):
    """Use sed3 tool to annotate data from given image."""
    ed = sed3.sed3(img)
    ed.show()

    seeds = ed.seeds
    seeds = seeds[:, :, 0]
    seeds = seeds.astype('int8')

    return seeds


def load_image(f_path=None):
    """Load and return .czi image from given path."""

    if f_path is None:
        root = tk.Tk()
        root.withdraw()

        f_path = filedialog.askopenfilename()

    anim = scim.AnnotatedImage(str(f_path))
    img = anim.get_full_view().get_region_image()

    img = zoom_img(img)

    return img


def run():
    """Run the annotation process."""
    img = load_image()
    seeds = annotate(img)
    np.save('seeds.npy', seeds)
    np.save('image.npy', img)


if __name__ == '__main__':
    run()
