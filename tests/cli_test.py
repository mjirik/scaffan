# /usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
import click.testing
import shutil
import pytest
import scaffan.main_cli
import io3d
from pathlib import Path
import unittest.mock
from unittest.mock import patch
path_to_script = Path(__file__).parent


def test_cli():
    pth = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )

    logger.debug(f"pth={pth}, exists={Path(pth).exists()}")
    expected_pth = path_to_script / "test_output/data.xlsx"
    logger.debug(f"expected_pth={expected_pth}, exists: {expected_pth.exists()}")
    if expected_pth.exists():
        shutil.rmtree(expected_pth.parent)

    # print("start")
    runner = click.testing.CliRunner()
    # runner.invoke(anwa.main_click.nogui, ["-i", str(pth)])
    import scaffan.image
    original_foo = scaffan.image.AnnotatedImage.get_annotations_by_color
    with patch.object(scaffan.image.AnnotatedImage, 'select_annotations_by_color', autospec=True) as mock_foo:
        def side_effect(*args, **kwargs):
            logger.debug("mocked function select_annotations_by_color()")
            original_list = original_foo(*args, **kwargs)
            logger.debug(f"original ann_ids={original_list}")
            # print(f"original ann_ids={original_list}")
            new_list = [original_list[-1]]
            logger.debug(f"new ann_ids={new_list}")
            return new_list

        mock_foo.side_effect = side_effect
        # print("in with statement")
        runner.invoke(
            scaffan.main_cli.run,
            ["nogui", "-i", pth, "-o", expected_pth.parent, "-c", "#FFFF00"],
        )

    assert expected_pth.exists()
