import sys
import time
import argparse
import time
import os
from pathlib import Path
from datetime import datetime

from tomocupy import logging
from tomocupy import config
from tomocupy import GPURec
from tomocupy import FindCenter
from tomocupy import GPURecSteps
from tomocupy.global_vars import args

from tomocupy.dataio import reader
from tomocupy.dataio import writer

log = logging.getLogger(__name__)


def init(args):
    if not os.path.exists(str(args.config)):
        config.write(args.config)
    else:
        log.error("{0} already exists".format(args.config))


def run_status(args):
    config.log_values(args)


def run_rec(args, cl_reader, cl_writer):
    file_name = Path(args.file_name)
    if not file_name.is_file():
        log.error("File Name does not exist: %s" % args.file_name)
        exit()

    t = time.time()
    # set the default parameters
    args.retrieve_phase_method = 'none'
    args.rotate_proj_angle = 0
    args.lamino_angle = 0
    # rotation axis search
    if args.rotation_axis_auto == 'auto':
        clrotthandle = FindCenter(cl_reader)
        args.rotation_axis = clrotthandle.find_center()
        log.warning(f'set rotaion  axis {args.rotation_axis}')

    # create reconstruction object and run reconstruction
    clpthandle = GPURec(cl_reader, cl_writer)

    if args.reconstruction_type == 'full':

        clpthandle.recon_all()
    if args.reconstruction_type == 'try':
        clpthandle.recon_try()
    rec_time = (time.time()-t)

    log.warning(f'Reconstruction time {rec_time:.1e}s')


def run_recsteps(args, cl_reader, cl_writer):
    file_name = Path(args.file_name)
    if not file_name.is_file():
        log.error("File Name does not exist: %s" % args.file_name)
        exit()
    t = time.time()

    if args.rotation_axis_auto == 'auto':
        clrotthandle = FindCenter(cl_reader, cl_writer)
        args.rotation_axis = clrotthandle.find_center()
        log.warning(f'set rotaion  axis {args.rotation_axis}')

    clpthandle = GPURecSteps(cl_reader, cl_writer)
    # does all preprocessing for both full and try reconstructions
    clpthandle.recon_steps_all()

    log.warning(f'Reconstruction time {(time.time()-t):.01f}s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', **config.SECTIONS['general']['config'])
    tomo_params = config.RECON_PARAMS
    tomo_steps_params = config.RECON_STEPS_PARAMS
    #

    cmd_parsers = [
        ('init',        init,            (),
         "Create configuration file"),
        ('recon',       run_rec,         tomo_params,
         "Run tomographic processing by splitting data into chunks in z "),
        ('recon_steps',   run_recsteps,     tomo_steps_params,
         "Run tomographic reconstruction by splitting by chunks in z and angles (step-wise)"),
        ('status',      run_status,      tomo_steps_params,
         "Show the tomographic reconstruction status"),
    ]

    subparsers = parser.add_subparsers(title="Commands", metavar='')

    for cmd, func, sections, text in cmd_parsers:
        cmd_params = config.Params(sections=sections)
        cmd_parser = subparsers.add_parser(
            cmd, help=text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        cmd_parser = cmd_params.add_arguments(cmd_parser)
        cmd_parser.set_defaults(_func=func)

    global args
    args.__dict__.update(config.parse_known_args(
        parser, subparser=True).__dict__)

    # create logger
    try:
        logs_home = args.logs_home
    except AttributeError:
        parser.print_help(sys.stderr)
        sys.exit(1)
    # test cupy
    import cupy as cp
    c = cp.ones(1)
    # make sure logs directory exists
    if not os.path.exists(logs_home):
        os.makedirs(logs_home)

    lfname = os.path.join(logs_home, 'tomocupyfp16on_' +
                          datetime.strftime(datetime.now(), "%Y-%m-%d_%H_%M_%S") + '.log')
    log_level = 'DEBUG' if args.verbose else "INFO"
    logging.setup_custom_logger(lfname, level=log_level)
    log.debug("Started tomocupyfp16on")
    log.info("Saving log at %s" % lfname)

    try:
        if args._func == init:
            args._func(args)
        else:
            cl_reader = reader.Reader()
            cl_writer = writer.Writer()
            args._func(args, cl_reader, cl_writer)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
