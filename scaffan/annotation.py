# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""
import logging
logger = logging.getLogger(__name__)
# problem is loading lxml together with openslide
# from lxml import etree
import json
import os.path as op
import glob
import matplotlib.pyplot as plt


def get_one_annotation(viewstate):
    titles_list = viewstate.xpath(".//title/text()")
    if len(titles_list) == 0:
        an_title = ""
    elif len(titles_list) == 1:
        an_title = titles_list[0]
    else:
        raise ValueError("More than one title in viewstate")

    annotations = viewstate.xpath(".//annotation")
    if len(annotations) > 1:
        raise ValueError("More than one annotation found")
    annot = annotations[0]
    an_color = annot.get("color")
    #     display(len(annotation))
    an_x = list(map(int, annot.xpath(".//pointlist/point/x/text()")))
    an_y = list(map(int, annot.xpath(".//pointlist/point/y/text()")))
    return dict(title=an_title, color=an_color, x=an_x, y=an_y)


def ndpa_file_to_json(pth):

    # problem is loading lxml together with openslide
    from lxml import etree
    tree = etree.parse(pth)
    viewstates = tree.xpath("//ndpviewstate")
    all_anotations = list(map(get_one_annotation, viewstates))
    fn = pth + ".json"
    with open(fn, 'w') as outfile:
        json.dump(all_anotations, outfile)


def ndpa_to_json(path):
    """
    :param path: path to file or dir contaning .ndpa files
    """
    if op.isfile(path):
        ndpa_file_to_json(path)
    else:
        extended_path = op.join(path, "*.ndpa")
        #         print(extended_path)
        files = glob.glob(extended_path)
        for fl in files:
            ndpa_file_to_json(fl)


def read_annotations(pth):
    fn = pth + ".ndpa.json"
    with open(fn) as f:
        data = json.load(f)
    return data

def plot_annotations(annotations):
    for annotation in annotations:
        plt.hold(True)
        plt.plot(annotation["x"], annotation["y"], c=annotation["color"])
