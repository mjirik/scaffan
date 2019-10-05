# /usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
import click.testing
import shutil
import pytest
import scaffan.main_cli
import io3d
from pathlib import Path


def test_cli():
    pth = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )

    logger.debug(f"pth={pth}")
    expected_pth = Path(".test_output/data.xlsx")
    logger.debug(f"expected_pth={expected_pth}")
    if expected_pth.exists():
        shutil.rmtree(expected_pth.parent)

    runner = click.testing.CliRunner()
    # runner.invoke(anwa.main_click.nogui, ["-i", str(pth)])
    runner.invoke(scaffan.main_cli.run,
                  ["nogui", "-i", pth, "-o", expected_pth, "-c", "#FFFF00"])

    assert expected_pth.exists()
