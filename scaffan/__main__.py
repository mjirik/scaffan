# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""
import logging

logger = logging.getLogger(__name__)
# print("start")
# from . import image

# print("start 5")
# print("start 6")

from scaffan import algorithm

# print("Running __main__.py")
mainapp = algorithm.Scaffan()
mainapp.start_gui()
