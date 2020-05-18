# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""

from loguru import logger
import sys
import click
from pathlib import Path
import ast

# print("start")
# from . import image

# print("start 5")
# print("start 6")

from scaffan import algorithm
from . import app_tools

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


# print("Running __main__.py")
# @batch_detect.command(context_settings=CONTEXT_SETTINGS)
# @click.argument("image_stack_dir", type=click.Path(exists=True))
# @click.argument("working_dir", type=click.Path())
# @click.option("--create-icon", is_flag=True,
#               help="Create desktop icon"
#               )
@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.option(
    "--log-level",
    "-ll",
    # type=,
    help="Set logging level",
    default="INFO",
)
@click.pass_context
def run(ctx, log_level, *args, **kwargs):
    if log_level is not None:
        try:
            log_level = int(log_level)
        except ValueError as e:
            log_level = log_level.upper()
        logger.remove()
        i = logger.add(sys.stderr, level=log_level, colorize=True)
    if ctx.invoked_subcommand is None:
        # click.echo("I am about to invoke GUI")
        ctx.invoke(gui, *args, **kwargs)
    else:
        pass
        # click.echo("I am about to invoke %s" % ctx.invoked_subcommand)
        # next command is useless. It is invoked automatically
        # ctx.invoke(ctx.invoked_subcommand, *args, **kwargs)


@run.command(context_settings=CONTEXT_SETTINGS, help="Set persistent values")
@click.option(
    "--common-spreadsheet-file",
    help="Set path for common spreadsheet file.",
    type=click.Path(),
)
def set(common_spreadsheet_file=None):
    mainapp = algorithm.Scaffan()
    if common_spreadsheet_file is not None:
        mainapp.set_common_spreadsheet_file(path=common_spreadsheet_file)
        logger.info(f"Common spreadsheet file path is : {common_spreadsheet_file}")
        print(f"Common spreadsheet file path is : {common_spreadsheet_file}")


# def print_params(params):
#     algorithm.Scaffan().parameters.
#     params.


@run.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--params",
    "-p",
    multiple=True,
    default="",
    nargs=2,
    help='Set parameter. First argument is path to parameter separated by ";". Second is the value.'
    "python -m scaffan gui -p Processing,Show True",
)
# @click.option(
#     "--log-level",
#     "-ll",
#     # type=,
#     help="Set logging level",
#     default=None,
# )
@click.option("--print-params", "-pp", is_flag=True, help="Print parameters")
def gui(params, print_params):
    mainapp = algorithm.Scaffan()
    if print_params:
        import pprint

        pprint.pprint(mainapp.parameters_to_dict())
        exit()
    for param in params:
        mainapp.set_parameter(param[0], value=ast.literal_eval(param[1]))
        # mainapp.parameters.param(*param[0].split(";")).setValue(ast.literal_eval(param[1]))
    mainapp.start_gui()


@run.command(
    context_settings=CONTEXT_SETTINGS, help="Create an icon on Windows platform"
)
def install():
    import platform

    print(platform.system)
    if platform.system() == "Windows":
        from .app_tools import create_icon
        import pathlib

        logo_fn2 = pathlib.Path(__file__).parent / pathlib.Path("scaffan_icon512.ico")
        create_icon(
            "Scaffan", logo_fn2, conda_env_name="scaffan", package_name="scaffan"
        )

        # logo_fn = op.join(op.dirname(__file__), "scaffan_icon512.ico")
        # import win32com.client
        # shell = win32com.client.Dispatch("WScript.Shell")
        #
        # pth = Path.home()
        # pth = pth / "Desktop" / Path("Scaffan.lnk")
        # shortcut = shell.CreateShortcut(str(pth))
        # # cmd
        # # ln =  "call activate scaffan; {} -m scaffan".format(sys.executable)
        # shortcut.TargetPath = sys.executable
        # shortcut.Arguments = "-m scaffan"
        # # shortcut.TargetPath = cmd
        # # shortcut.Arguments = '/C "call activate scaffan & python -m scaffan" '
        # shortcut.IconLocation = "{},0".format(logo_fn)
        # shortcut.Save()
    pass


@run.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--input-path",
    "-i",
    type=click.Path(exists=True),
    help="Path to input directory with video files.",
    default=None,
)
@click.option(
    "--color", "-c", type=str, help="Annotation collor in hexa (#0000FF)", default=None,
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(),
    help="Path to output directory with video files.",
    default=None,
)
# @click.option(
#     "--log-level",
#     "-ll",
#     # type=,
#     help="Set logging level",
#     default=None,
# )
@click.option(
    "--params",
    "-p",
    multiple=True,
    default="",
    nargs=2,
    help='Set parameter. First argument is path to parameter separated by ";". Second is the value.'
    "python -m scaffan gui -p Processing,Show True",
)
def nogui(input_path, color, output_path, params):
    # if log_level is not None:
    #     i = logger.add(level=log_level)
    logger.debug(
        f"input path={input_path} color={color}, output_path={output_path}, params={params}"
    )
    mainapp = algorithm.Scaffan()
    logger.debug(f"Scaffan created")
    app_tools.set_parameters_by_path(mainapp.parameters, params)
    # for param in params:
    #     logger.debug(f"param={param}")
    #     mainapp.parameters.param(*param[0].split(";")).setValue(ast.literal_eval(param[1]))

    logger.debug("before input file")
    if input_path is not None:
        mainapp.set_input_file(input_path)
    if output_path is not None:
        mainapp.set_output_dir(output_path)
    if color is not None:
        logger.debug(f"color={color}")
        mainapp.set_annotation_color_selection(color)

    mainapp.run_lobuluses()


# def install():
def create_icon(
        app_name: str, icon_filename=None, conda_env_name=None, package_name=None
):
    """

    :param app_name: Used for desktop icon name
    :param icon_filename: absolute path to icon usually:
        pathlib.Path(__file__).parent / pathlib.Path("app_icon512.ico")
    :param conda_env_name: conda environment. The app_name is used if conda_env_name is set to None.
    :param package_name: in `conda -m 'package_name'` . The app_name is used if conda_env_name is set to None.
    :return:
    """
    import platform

    # print(platform.system)
    if conda_env_name is None:
        conda_env_name = app_name

    # if icon_filename is None:
    #     icon_filename = app_name

    if Path(icon_filename).suffix == "":
        icon_filename += ".ico"

    if package_name is None:
        package_name = app_name

    if platform.system() == "Windows":

        # logo_fn2 = pathlib.Path(__file__).parent / pathlib.Path("scaffan_icon512.ico")

        # logo_fn = op.join(op.dirname(__file__), icon_filename)
        logo_fn = icon_filename
        import win32com.client

        shell = win32com.client.Dispatch("WScript.Shell")

        pth = Path.home()
        pth = pth / "Desktop" / Path(f"{app_name}.lnk")
        shortcut = shell.CreateShortcut(str(pth))
        # cmd
        # ln =  "call activate scaffan; {} -m scaffan".format(sys.executable)
        # C:\Windows\System32\cmd.exe /C "call activate anwaapp & pause &  python -m anwa & pause"
        # shortcut.TargetPath = sys.executable
        # shortcut.Arguments = f"-m {app_name}"
        shortcut.TargetPath = "cmd"
        # C:\Windows\System32\cmd.exe /C "call activate anwaapp & pause &  python -m anwa & pause"
        shortcut.Arguments = (
            f'/C "call activate {conda_env_name} & python -m {package_name} & pause" '
        )
        shortcut.IconLocation = "{},0".format(logo_fn)
        shortcut.Save()
