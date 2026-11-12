#!/usr/bin/env python
# -*- coding: utf-8 -*-

# *************************************************************************** #
#                  Copyright © 2022, UChicago Argonne, LLC                    #
#                           All Rights Reserved                               #
#                         Software Name: Tomocupy                             #
#                     By: Argonne National Laboratory                         #
#                                                                             #
#                           OPEN SOURCE LICENSE                               #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
#    this list of conditions and the following disclaimer.                    #
# 2. Redistributions in binary form must reproduce the above copyright        #
#    notice, this list of conditions and the following disclaimer in the      #
#    documentation and/or other materials provided with the distribution.     #
# 3. Neither the name of the copyright holder nor the names of its            #
#    contributors may be used to endorse or promote products derived          #
#    from this software without specific prior written permission.            #
#                                                                             #
#                                                                             #
# *************************************************************************** #
#                               DISCLAIMER                                    #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS         #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT           #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS           #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT    #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,      #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED    #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR      #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF      #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING        #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS          #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                #
# *************************************************************************** #

import sys
import time
import argparse
import os
from pathlib import Path
from datetime import datetime

from tomocupy import logging
from tomocupy import config
from tomocupy import GPURec
from tomocupy import FindCenter
from tomocupy import GPURecSteps
from tomocupy.global_vars import args, params

from tomocupy.dataio import reader
from tomocupy.dataio import writer

__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

log = logging.getLogger(__name__)

def _check_use_ai(args):
    if args.rotation_axis_auto != 'auto' or args.rotation_axis_method != 'ai':
        return False
    try:
        import torch  
        return True
    except ImportError:
        log.warning('torch is not installed — skipping AI center search, falling back to vo method')
        args.rotation_axis_method = 'vo'
        return False

def init(args):
    if not os.path.exists(str(args.config)):
        config.write(args.config)
    else:
        log.error("{0} already exists".format(args.config))


def run_status(args):
    config.log_values(args)


def _find_center(cl_reader):
    clrotthandle = FindCenter(cl_reader)
    args.rotation_axis = clrotthandle.find_center()
    params.center = args.rotation_axis
    params.centeri = args.rotation_axis
    log.warning(f'set rotation axis {args.rotation_axis}')

    # Re-anchor try-mode save_centers labels now that centeri is known.
    if args.reconstruction_type[:3] == 'try' and hasattr(params, 'shift_array'):
        params.save_centers = ((params.centeri - params.shift_array)
                               * 2**args.binning + params.st_n)


def _find_center_ai(save_test_results_ok=False,cache_preprocessed=False,_func_type='run_rec'):
    clrotthandle = FindCenter(None)
    results_all = clrotthandle.find_center_ai(save_test_results_ok=save_test_results_ok, cache_preprocessed=cache_preprocessed, _func_type=_func_type)
    args.rotation_axis, score = results_all['Stage 2']['rotation axis'], results_all['Stage 2']['score']
    params.center = args.rotation_axis
    log.warning(f'set rotation axis {args.rotation_axis}')
    log.warning(f'score is {score}')
    # Re-anchor try-mode save_centers labels now that centeri is known.
    #TODO: verify if consistent
    if args.reconstruction_type[:3] == 'try' and hasattr(params, 'shift_array'):
        params.save_centers = ((params.centeri - params.shift_array)
                                * 2**args.binning + params.st_n)
    return results_all

def run_rec(args, cl_reader, cl_writer, save_test_results_ok:bool = False):
    if not Path(args.file_name).is_file():
        log.error("File Name does not exist: %s" % args.file_name)
        exit()

    args.retrieve_phase_method = 'none'
    args.rotate_proj_angle = 0
    args.lamino_angle = 0

    use_ai = _check_use_ai(args)
    if args.rotation_axis_auto == 'auto' and not use_ai:
        _find_center(cl_reader)

    cache_to_infer = args.reconstruction_type == 'try' and use_ai

    if args.reconstruction_type == 'full':
        if use_ai:
            from copy import deepcopy
            args_dict = deepcopy(args.__dict__)
            params_dict = deepcopy(params.__dict__)
            args.reconstruction_type = 'try'
            results_all = _find_center_ai(save_test_results_ok=save_test_results_ok,_func_type='run_rec')
            rotation_axis = args.rotation_axis
            args.__dict__.clear()
            args.__dict__.update(args_dict)
            params.__dict__.clear()
            params.__dict__.update(params_dict)
            args.rotation_axis = rotation_axis
            cl_reader = reader.Reader()
            cl_writer = writer.Writer()

        clpthandle = GPURec(cl_reader, cl_writer, cache_to_infer=cache_to_infer)
        clpthandle.recon_all()
        if use_ai:
            return results_all
    elif args.reconstruction_type == 'try':
        if use_ai:
            results_all = _find_center_ai(save_test_results_ok=save_test_results_ok,_func_type='run_rec')
            return results_all
        else:
            clpthandle = GPURec(cl_reader, cl_writer, cache_to_infer=cache_to_infer)
            clpthandle.recon_try()

def run_recsteps(args, cl_reader, cl_writer, save_test_results_ok:bool = False, cache_preprocessed:bool=False):
    if not Path(args.file_name).is_file():
        log.error("File Name does not exist: %s" % args.file_name)
        exit()

    use_ai = _check_use_ai(args)
    if args.rotation_axis_auto == 'auto' and not use_ai:
        _find_center(cl_reader)

    cache_to_infer = args.reconstruction_type == 'try' and use_ai

    if cache_to_infer:
        results_all = _find_center_ai(save_test_results_ok=save_test_results_ok,cache_preprocessed=cache_preprocessed,_func_type='run_recsteps')
        return results_all
    else:
        if args.reconstruction_type == 'full' and use_ai:
            from copy import deepcopy
            args_dict = deepcopy(args.__dict__)
            params_dict = deepcopy(params.__dict__)
            args.reconstruction_type = 'try'
            results_all = _find_center_ai(save_test_results_ok=save_test_results_ok,cache_preprocessed=cache_preprocessed,_func_type='run_recsteps')
            rotation_axis = args.rotation_axis
            args.__dict__.clear()
            args.__dict__.update(args_dict)
            params.__dict__.clear()
            params.__dict__.update(params_dict)
            args.rotation_axis = rotation_axis
            cl_reader = reader.Reader()
            cl_writer = writer.Writer()

        clpthandle = GPURecSteps(cl_reader, cl_writer, cache_to_infer=cache_to_infer)
        clpthandle.recon_steps_all()
        if args.reconstruction_type == 'full' and use_ai:
            return results_all    

def save_results(results_all):
    import json
    if args.reconstruction_type == 'try':
        with open(params.fnameout[:-6]+"/test_results.json", "w", encoding="utf-8") as f:
            json.dump(results_all, f, indent=4, ensure_ascii=False)
    elif args.reconstruction_type == 'full':
        if Path(params.fnameout).is_file():
            with open(str(Path(params.fnameout).parent)+f"/test_results_{Path(params.fnameout).stem}.json", "w", encoding="utf-8") as f:
                json.dump(results_all, f, indent=4, ensure_ascii=False)
        elif (Path(params.fnameout).is_dir() and args.save_format == 'zarr'):
            with open(str(Path(params.fnameout).parent)+f"/test_results_{Path(params.fnameout).stem}.json", "w", encoding="utf-8") as f:
                    json.dump(results_all, f, indent=4, ensure_ascii=False) 
        elif args.save_format == 'tiff':
            with open(params.fnameout[:-6]+"/test_results.json", "w", encoding="utf-8") as f:
                    json.dump(results_all, f, indent=4, ensure_ascii=False)

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
            save_test_results_ok = args.save_test_results
            args.symmetric_center_search = False
            if args.rotation_axis_method != 'ai':
                cl_reader = reader.Reader()
                cl_writer = writer.Writer()
            else:
                cl_reader = None
                cl_writer = None
            if (args._func == run_rec) or (args._func == run_recsteps):
                if args._func == run_rec:
                    results_all = args._func(args, cl_reader, cl_writer, save_test_results_ok = save_test_results_ok)
                elif args._func == run_recsteps:
                    results_all = args._func(args, cl_reader, cl_writer, save_test_results_ok = save_test_results_ok, cache_preprocessed=args.bin_infer_cache_preprocessed)
                if save_test_results_ok:
                    save_results(results_all)
            else:
                args._func(args, cl_reader, cl_writer)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
