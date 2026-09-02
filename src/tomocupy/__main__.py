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


def _find_center_ai(cl_reader, img_cache, center_of_rotation_cache):
    clrotthandle = FindCenter(cl_reader)
    args.rotation_axis = clrotthandle.find_center_ai(args, img_cache, center_of_rotation_cache, params.fnameout[:-6])
    params.center = args.rotation_axis
    log.warning(f'set rotation axis {args.rotation_axis}')

    # Re-anchor try-mode save_centers labels now that centeri is known.
    if args.reconstruction_type[:3] == 'try' and hasattr(params, 'shift_array'):
        params.save_centers = ((params.centeri - params.shift_array)
                                * 2**args.binning + params.st_n)

def _find_center_range_ai(cl_reader, img_cache, center_of_rotation_cache):
    clrotthandle = FindCenter(cl_reader)
    center_lb, center_ub = clrotthandle.find_center_range_ai(args, img_cache, center_of_rotation_cache, params.fnameout[:-6])
    log.warning(f'center range refined to ({center_lb},{center_ub})')
    return center_lb, center_ub

def _check_use_ai():
    if args.rotation_axis_auto != 'auto' or args.rotation_axis_method != 'ai':
        return False
    try:
        import torch  
        return True
    except ImportError:
        log.warning('torch is not installed — skipping AI center search, falling back to vo method')
        args.rotation_axis_method = 'vo'
        return False

def run_rec_presteps(args, cl_reader, cl_writer, save_test_results_ok = False):
    if not Path(args.file_name).is_file():
        log.error("File Name does not exist: %s" % args.file_name)
        exit()

    t = time.time()
    args.retrieve_phase_method = 'none'
    args.rotate_proj_angle = 0
    args.lamino_angle = 0

    use_ai = _check_use_ai()
    if not use_ai:
        log.error("Error: ai is not properly configured.")
        exit()
    if args.reconstruction_type != 'try':
        log.error(f"Error: reconstruction type is not 'try'. Detected {args.reconstruction_type} instead.")
        exit()
    cache_to_infer = args.reconstruction_type == 'try' and use_ai
    clpthandle = GPURec(cl_reader, cl_writer, cache_to_infer=cache_to_infer)

    img_cache, center_of_rotation_cache, _ = clpthandle.recon_try()
    t1 = time.time()
    if center_of_rotation_cache[-1] < center_of_rotation_cache[0]:
        img_cache = img_cache[::-1]
        center_of_rotation_cache = center_of_rotation_cache[::-1]
    center_lb, center_ub = _find_center_range_ai(cl_reader, img_cache, center_of_rotation_cache)
    t2 = time.time()
    log.warning(f'Reconstruction time {t2-t:.1e}s')
    results = {'center_lb':float(center_lb),'center_ub':float(center_ub)}
    if save_test_results_ok:
        results['running time recon'] = t1-t
        results['running time infer'] = t2-t1
    
    return results

def run_rec(args, cl_reader, cl_writer, save_test_results_ok = False):
    if not Path(args.file_name).is_file():
        log.error("File Name does not exist: %s" % args.file_name)
        exit()

    t = time.time()
    args.retrieve_phase_method = 'none'
    args.rotate_proj_angle = 0
    args.lamino_angle = 0

    use_ai = _check_use_ai()
    if args.rotation_axis_auto == 'auto' and not use_ai:
        _find_center(cl_reader)

    cache_to_infer = args.reconstruction_type == 'try' and use_ai
    clpthandle = GPURec(cl_reader, cl_writer, cache_to_infer=cache_to_infer)

    if args.reconstruction_type == 'full':
        clpthandle.recon_all()
        t1 = time.time()
        t2 = t1
    elif args.reconstruction_type == 'try':
        if use_ai:
            img_cache, center_of_rotation_cache, _ = clpthandle.recon_try()
            t1 = time.time()
            _find_center_ai(cl_reader, img_cache, center_of_rotation_cache)
            t2 = time.time()
        else:
            clpthandle.recon_try()
            t1 = time.time()
            t2 = t1

    log.warning(f'Reconstruction time {t2-t:.1e}s')
    results = {}
    if save_test_results_ok:
        results['running time recon'] = t1-t
        results['running time infer'] = t2-t1
        results['rotation axis'] = float(args.rotation_axis)
    return results

def run_recsteps_presteps(args, cl_reader, cl_writer, save_test_results_ok:bool = False, recon_from_cache_preprocessed:bool=False, preprocessed_cache=None, cache_preprocessed:bool=False):
    if not Path(args.file_name).is_file():
        log.error("File Name does not exist: %s" % args.file_name)
        exit()

    t = time.time()

    use_ai = _check_use_ai()
    if not use_ai:
        log.error("Error: ai is not properly configured.")
        exit()

    cache_to_infer = use_ai
    clpthandle = GPURecSteps(cl_reader, cl_writer, cache_to_infer=cache_to_infer)

    results_ = clpthandle.recon_steps_all(recon_from_cache_preprocessed=recon_from_cache_preprocessed, preprocessed_cache=preprocessed_cache, cache_preprocessed=cache_preprocessed)
    img_cache, center_of_rotation_cache = results_['img_cache'], results_['center_of_rotation_cache']
    t1 = time.time()
    if center_of_rotation_cache[-1] < center_of_rotation_cache[0]:
        img_cache = img_cache[::-1]
        center_of_rotation_cache = center_of_rotation_cache[::-1]
    center_lb, center_ub = _find_center_range_ai(cl_reader, img_cache, center_of_rotation_cache)
    t2 = time.time()
    results = {'center_lb':float(center_lb), 'center_ub':float(center_ub)}
    if cache_preprocessed:
        results['preprocessed_cache'] = results_['data']
    if save_test_results_ok:
        results['running time recon'] = t1-t
        results['running time infer'] = t2-t1
    log.warning(f'Reconstruction time {t2-t:.1f}s')
    return results

def run_recsteps(args, cl_reader, cl_writer, save_test_results_ok:bool = False, recon_from_cache_preprocessed:bool=False, preprocessed_cache=None, cache_preprocessed:bool=False):
    if not Path(args.file_name).is_file():
        log.error("File Name does not exist: %s" % args.file_name)
        exit()

    t = time.time()

    use_ai = _check_use_ai()
    if args.rotation_axis_auto == 'auto' and not use_ai:
        _find_center(cl_reader)

    cache_to_infer = args.reconstruction_type == 'try' and use_ai
    clpthandle = GPURecSteps(cl_reader, cl_writer, cache_to_infer=cache_to_infer)

    if cache_to_infer:
        results_ = clpthandle.recon_steps_all(recon_from_cache_preprocessed=recon_from_cache_preprocessed, preprocessed_cache=preprocessed_cache, cache_preprocessed=cache_preprocessed)
        img_cache, center_of_rotation_cache = results_['img_cache'], results_['center_of_rotation_cache']
        t1 = time.time()
        _find_center_ai(cl_reader, img_cache, center_of_rotation_cache)
        t2 = time.time()
    else:
        clpthandle.recon_steps_all()
        t1 = time.time()
        t2 = t1

    results = {}
    if cache_preprocessed:
        results['preprocessed_cache'] = results_['data']
    if save_test_results_ok:
        results['running time recon'] = t1-t
        results['running time infer'] = t2-t1
        results['rotation axis'] = float(args.rotation_axis)
    log.warning(f'Reconstruction time {t2-t:.1f}s')
    return results

def try_recon_ai_full(results_all,save_test_results_ok=False,cache_preprocessed=False,export_results_ok=False):
    if len(args.bin_infer_bin_sizes) != len(args.bin_infer_bin_counts):
        log.error(f"Numbers of bin sizes and bin counts do not match: got {len(args.bin_infer_bin_sizes)} and {len(args.bin_infer_bin_counts)}, respectively.")
        exit()
    args.symmetric_center_search = True
    args.clear_folder = 'True'
    center_search_step = args.center_search_step
    for i, (bin_size, bin_count) in enumerate(zip(args.bin_infer_bin_sizes,args.bin_infer_bin_counts)):
        if bin_count%2 != 0:
            log.error(f"Number of bins to search should be even: got {bin_count} instead.")
            exit()
        log.info(f"Level {i+1}: search range is {bin_count} bins each of {bin_size} pixels")
        args.center_search_step = float(bin_size)
        args.center_search_width = float(bin_count) / 2 * float(bin_size)
        args.bin_infer_bin_size = float(bin_size)
            
        cl_reader = reader.Reader()
        cl_writer = writer.Writer()
        
        if args._func == run_rec:
            results = run_rec_presteps(args, cl_reader, cl_writer, save_test_results_ok = save_test_results_ok)
        elif args._func == run_recsteps:
            if i == 0:
                recon_from_cache_preprocessed = False
                preprocessed_cache = None
            else:
                cache_preprocessed = False
                recon_from_cache_preprocessed = True if preprocessed_cache is not None else False
                
            results = run_recsteps_presteps(args, cl_reader, cl_writer, save_test_results_ok = save_test_results_ok,\
                recon_from_cache_preprocessed=recon_from_cache_preprocessed,preprocessed_cache=preprocessed_cache,cache_preprocessed=cache_preprocessed)
            
            if i == 0 and cache_preprocessed: 
                preprocessed_cache = results['preprocessed_cache']
        center_lb = results['center_lb']
        center_ub = results['center_ub']
        args.rotation_axis = (center_lb+center_ub)/2
        if save_test_results_ok:
            results_all[f"Stage 1 level {i+1}"] = results
            results_all[f"Stage 1 level {i+1}"].pop("preprocessed_cache", None)
        # log.info(f"Level {i+1}: refined range is ({center_lb},{center_ub})")
    args.center_search_step = center_search_step
    args.center_search_width = (center_ub-center_lb)/2

    args.symmetric_center_search = False
    cl_reader = reader.Reader()
    cl_writer = writer.Writer()
    if args._func == run_rec:
        results = args._func(args, cl_reader, cl_writer, save_test_results_ok = save_test_results_ok)
    elif args._func == run_recsteps:
        cache_preprocessed = False
        recon_from_cache_preprocessed = True if preprocessed_cache is not None else False
        results = args._func(args, cl_reader, cl_writer, save_test_results_ok = save_test_results_ok,\
        recon_from_cache_preprocessed=recon_from_cache_preprocessed,preprocessed_cache=preprocessed_cache,cache_preprocessed=cache_preprocessed)
    if save_test_results_ok:
        results_all["Stage 2"] = results
    if export_results_ok:
        print(f"results are: {results_all}")
        import json
        with open(params.fnameout[:-6]+"/test_results.json", "w", encoding="utf-8") as f:
            json.dump(results_all, f, indent=4, ensure_ascii=False)

def recon(results_all,save_test_results_ok=False,export_results_ok=False):
    cl_reader = reader.Reader()
    cl_writer = writer.Writer()
    results = args._func(args, cl_reader, cl_writer, save_test_results_ok = save_test_results_ok)
    if save_test_results_ok:
        if args.reconstruction_type == 'try':
            results_all["Stage 2"] = results
        elif args.reconstruction_type == 'full':
            results_all["Stage full"] = results
    if export_results_ok:
        print(f"results are: {results_all}")
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
            
            results_all = {}
            args.symmetric_center_search = False
            if ((args._func == run_rec) or (args._func == run_recsteps)) and (args.rotation_axis_method == 'ai') and (args.ai_search_method == 'full') and (args.reconstruction_type == 'try'):
                try_recon_ai_full(results_all,save_test_results_ok=save_test_results_ok,cache_preprocessed=args.bin_infer_cache_preprocessed,export_results_ok=save_test_results_ok)
            elif ((args._func == run_rec) or (args._func == run_recsteps)) and (args.rotation_axis_method == 'ai') and (args.ai_search_method == 'full') and (args.reconstruction_type == 'full'):
                from copy import deepcopy
                args_dict = deepcopy(args.__dict__)
                params_dict = deepcopy(params.__dict__)
                args.reconstruction_type = 'try'
                try_recon_ai_full(results_all,save_test_results_ok=save_test_results_ok,cache_preprocessed=args.bin_infer_cache_preprocessed)
                rotation_axis = args.rotation_axis
                args.__dict__.clear()
                args.__dict__.update(args_dict)
                params.__dict__.clear()
                params.__dict__.update(params_dict)
                args.rotation_axis = rotation_axis
                recon(results_all,save_test_results_ok=save_test_results_ok,export_results_ok=save_test_results_ok)

            elif ((args._func == run_rec) or (args._func == run_recsteps)) and (args.rotation_axis_method == 'ai') and (args.ai_search_method == 'fine') and (args.reconstruction_type == 'full'):
                from copy import deepcopy
                args_dict = deepcopy(args.__dict__)
                params_dict = deepcopy(params.__dict__)
                args.reconstruction_type = 'try'
                recon(results_all,save_test_results_ok=save_test_results_ok)
                rotation_axis = args.rotation_axis
                args.__dict__.clear()
                args.__dict__.update(args_dict)
                params.__dict__.clear()
                params.__dict__.update(params_dict)
                args.rotation_axis = rotation_axis
                recon(results_all,save_test_results_ok=save_test_results_ok,export_results_ok=save_test_results_ok)

            elif (args._func == run_rec) or (args._func == run_recsteps):
                recon(results_all,save_test_results_ok=save_test_results_ok,export_results_ok=save_test_results_ok)
            else:
                cl_reader = reader.Reader()
                cl_writer = writer.Writer()
                args._func(args, cl_reader, cl_writer)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
