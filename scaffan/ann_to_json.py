# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""
import logging
logger = logging.getLogger(__name__)
import lxml
from lxml import etree
import json
import os.path as op
import glob
import sys

from scaffan import annotation
print("ann to json")

if __name__ == "__main__":

    print(sys.argv)
    pth = op.expanduser(sys.argv[1])
    print(pth)
    annotation.ndpa_to_json(pth)
