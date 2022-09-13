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

from tomocupy import config
from tomocupy import logging
from tomocupy import utils
import numpy as np
import numexpr as ne
import h5py
import os
import sys
import tifffile

__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Writer', ]

log = logging.getLogger(__name__)


class Writer():
    '''
    Class for configuring write operations.
    '''

    def __init__(self, args, cl_conf):
        self.args = args
        self.n = cl_conf.n
        self.nzi = cl_conf.nzi
        self.dtype = cl_conf.dtype
        self.nzchunk = cl_conf.nzchunk
        self.lzchunk = cl_conf.lzchunk
        self.ncz = cl_conf.ncz
        if self.args.reconstruction_type[:3] == 'try':
            self.init_output_files_try()
        else:
            self.init_output_files()

    def init_output_files_try(self):
        """Constructing output file names and initiating the actual files"""

        fnameout = os.path.dirname(
            self.args.file_name)+'_rec/try_center/'+os.path.basename(self.args.file_name)[:-3]
        os.system(f'mkdir -p {fnameout}')
        fnameout += '/recon'
        self.fnameout = fnameout
        log.info(f'Output: {fnameout}')

    def init_output_files(self):
        """Constructing output file names and initiating the actual files"""

        # init output files
        if(self.args.out_path_name is None):
            fnameout = os.path.dirname(
                self.args.file_name)+'_rec/'+os.path.basename(self.args.file_name)[:-3]+'_rec'
            os.system(f'mkdir -p {fnameout}')
        else:
            fnameout = str(self.args.out_path_name)

        if self.args.save_format == 'tiff':
            # if save results as tiff
            fnameout += '/recon'
            # saving command line for reconstruction
            fname_rec_line = os.path.dirname(fnameout)+'/rec_line.txt'
            rec_line = sys.argv
            rec_line[0] = os.path.basename(rec_line[0])
            with open(fname_rec_line, 'w') as f:
                f.write(' '.join(rec_line))

        elif self.args.save_format == 'h5':
            # if save results as h5 virtual datasets
            fnameout += '.h5'
            # Assemble virtual dataset
            layout = h5py.VirtualLayout(shape=(
                self.nzi/2**self.args.binning, self.n, self.n), dtype=self.dtype)
            os.system(f'mkdir -p {fnameout[:-3]}_parts')
            for k in range(self.nzchunk):
                filename = f"{fnameout[:-3]}_parts/p{k:04d}.h5"
                vsource = h5py.VirtualSource(
                    filename, "/exchange/data", shape=(self.lzchunk[k], self.n, self.n), dtype=self.dtype)
                st = self.args.start_row//2**self.args.binning+k*self.ncz
                layout[st:st+self.lzchunk[k]] = vsource

            # Add virtual dataset to output file
            if self.args.h5init == 'True':
                rec_virtual = h5py.File(fnameout, "w")
                dset_rec = rec_virtual.create_virtual_dataset(
                    "/exchange/data", layout)

            # saving command line to repeat the reconstruction as attribute of /exchange/data
            
            rec_line = sys.argv
            # remove full path to the file
            rec_line[0] = os.path.basename(rec_line[0])
            s = ' '.join(rec_line).encode("utf-8")
            dset_rec.attrs["command"] = np.array(
                s, dtype=h5py.string_dtype('utf-8', len(s)))
            dset_rec.attrs["axes"] = 'z:y:x'
            dset_rec.attrs["description"] = 'ReconData'
            dset_rec.attrs["units"] = 'counts'

            try:  # trying to copy meta
                import meta
                tree, meta_dict = meta.read_hdf(self.args.file_name)
                for key, value in meta_dict.items():
                    # print(key, value)
                    dset = rec_virtual.create_dataset(key, data=value[0])
                    if value[1] is not None:
                        dset.attrs['units'] = value[1]
            except:
                log.info('Skip copying meta')
                pass

            rec_virtual.close()
            config.update_hdf_process(fnameout, self.args, sections=(
                'file-reading', 'remove-stripe',  'reconstruction', 'blocked-views', 'fw'))
        self.fnameout = fnameout
        log.info(f'Output: {fnameout}')

    def write_data_chunk(self, rec, k):
        """Writing the kth data chunk to hard disk"""

        if self.args.crop > 0:
            rec = rec[:, self.args.crop:-
                      self.args.crop, self.args.crop:-self.args.crop]

        if self.args.save_format == 'tiff':
            st = k*self.ncz+self.args.start_row//2**self.args.binning
            for kk in range(self.lzchunk[k]):
                fid = st+kk
                tifffile.imwrite(f'{self.fnameout}_{fid:05}.tiff', rec[kk])
        elif self.args.save_format == 'h5':
            filename = f"{self.fnameout[:-3]}_parts/p{k:04d}.h5"
            with h5py.File(filename, "w") as fid:
                fid.create_dataset("/exchange/data", data=rec,
                                   chunks=(1, self.n, self.n))

    def write_data_try(self, rec, cid):
        """Write tiff reconstruction with a given name"""

        if self.args.crop > 0:
            rec = rec[self.args.crop:-self.args.crop,
                      self.args.crop:-self.args.crop]
        tifffile.imwrite(f'{self.fnameout}_{cid:05.2f}.tiff', rec)
