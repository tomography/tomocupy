import sys
import time
import argparse
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py
import subprocess
import time

from tomocupy import logging
from tomocupy import config
from tomocupy import GPURec
from tomocupy import FindCenter
from tomocupy import GPURecSteps
from tomocupy import GPUProc

log = logging.getLogger(__name__)


def init(args):
    if not os.path.exists(str(args.config)):
        config.write(args.config)
    else:
        log.error("{0} already exists".format(args.config))


def run_status(args):
    config.log_values(args)


def run_rec(args):
    t = time.time()
    file_name = Path(args.file_name)
    if file_name.is_file():
        args.retrieve_phase_method = 'none'  # don no allow phase retrieval here
        args.rotate_proj_angle = 0  # do not allow to rotate projections
        args.lamino_angle = 0
        if args.rotation_axis_auto == 'auto':
            clrotthandle = FindCenter(args)
            args.rotation_axis = clrotthandle.find_center()*2**args.binning
            log.warning(f'set rotaion  axis {args.rotation_axis}')
        # t1=time.time()
        clpthandle = GPURec(args)
        # init_time=time.time()-t1
        if(args.reconstruction_type == 'full'):
            clpthandle.recon_all()
        if(args.reconstruction_type == 'try'):
            clpthandle.recon_try()
        # update config file
        # sections = config.RECON_PARAMS
        # config.write(args.config, args=args, sections=sections)
    else:
        log.error("File Name does not exist: %s" % args.file_name)

    rec_time = (time.time()-t)  # -init_time
    log.warning(f'Reconstruction time {rec_time:.1e}s')
    # log.warning(f'Init time {init_time:.1e}s')
    # with h5py.File(file_name) as fid:
    # np.save('time',rec_time/fid['exchange/data'].shape[1]*fid['exchange/data'].shape[2]+init_time)

def run_proc(args):
    t = time.time()
    file_name = Path(args.file_name)
    if file_name.is_file():
        args.retrieve_phase_method = 'none'  # don no allow phase retrieval here
        args.rotate_proj_angle = 0  # do not allow to rotate projections
        args.lamino_angle = 0
        
        clpthandle = GPUProc(args)
        clpthandle.proc_sino_parallel()
    else:
        log.error("File Name does not exist: %s" % args.file_name)

    rec_time = (time.time()-t)  # -init_time
    log.warning(f'Reconstruction time {rec_time:.1e}s')


def run_recstep(args):
    # config.show_config(args)
    file_name = Path(args.file_name)
    if file_name.is_file():
        t = time.time()
        if args.lamino_angle != 0 and args.reconstruction_algorithm != 'linesummation':
            log.warning(
                'Switching to reconstruction algorithm linesummation for laminography')
            args.reconstruction_algorithm = 'linesummation'
        if args.rotation_axis_auto == 'auto':
            clrotthandle = FindCenter(args)
            args.rotation_axis = clrotthandle.find_center()
            log.warning(f'set rotaion  axis {args.rotation_axis}')
        clpthandle = GPURecSteps(args)
        clpthandle.recon_steps_all()
        log.warning(f'Reconstruction time {(time.time()-t):.01f}s')

        # # update config file
        # sections = config.RECON_STEPS_PARAMS
        # config.write(args.config, args=args, sections=sections)

    else:
        log.error("File Name does not exist: %s" % args.file_name)


def run_recmulti(args):
    line = ' '.join(sys.argv[2:])
    if(args.end_row == -1):
        with h5py.File(args.file_name, 'r') as fid:
            args.end_row = fid['/exchange/data/'].shape[1]

    cmd1 = f"ssh -t tomo@tomo1 \"bash -c 'source ~/.bashrc; conda activate tomocupy; tomocupy recon {line} --start-row {args.start_row} --end-row {args.end_row//2}\'\""
    cmd2 = f"ssh -t tomo@tomo2 \"bash -c 'source ~/.bashrc; conda activate tomocupy; tomocupy recon {line} --start-row {args.end_row//2} --end-row {args.end_row}\'\""
    print(f'Tomo1: {cmd1}')
    print(f'Tomo2: {cmd2}')
    p1 = subprocess.Popen(cmd1, shell=True)
    time.sleep(1)
    p2 = subprocess.Popen(cmd2, shell=True)
    p1.wait()
    p2.wait()


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
        ('proc',     run_proc,           tomo_params,
         "Run tomographic reconstruction by steps and saving result to h5 file"),        
        ('recon_steps',   run_recstep,     tomo_steps_params,
         "Run tomographic reconstruction by splitting by chunks in z and angles (step-wise)"),
        ('reconmulti',  run_recmulti,    tomo_params,
         "Run reconstruction on several nodes"),
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

    args = config.parse_known_args(parser, subparser=True)
    # create logger
    logs_home = args.logs_home
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
        args._func(args)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
