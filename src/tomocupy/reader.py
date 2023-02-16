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

from tomocupy import log_local as logging
from tomocupy import utils
import numpy as np
import h5py

__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Reader', ]

log = logging.getLogger(__name__)


class Reader():
    '''
    Class for configuring read operations. For constructing readers for other data formats, please implement all functions in this class. 
    '''

    def __init__(self, args):
        self.args = args
        
        if self.args.flat_dark_file_name == None:
            self.args.flat_dark_file_name = self.args.file_name
        else:
            log.warning(f'Using flat and dark fields from {self.args.flat_dark_file_name}')


    def read_sizes(self):
        '''
        Read data sizes        
        Output: dictionary with fields

        nproji - number of projections
        nzi - detector height
        ni - detector width
        nflat - number of flat fields
        ndark - number of dark fields
        dtype - data type (e.g., 'uint16', 'uint8')
        '''
        sizes = {}

        with h5py.File(self.args.file_name) as file_in:
            data = file_in['/exchange/data']
            nproj, nzi, ni = data.shape[:]

            sizes['dtype'] = data.dtype
            sizes['nproji'] = nproj
            sizes['nzi'] = nzi
            sizes['ni'] = ni

        with h5py.File(self.args.flat_dark_file_name) as file_in:
            dark = file_in['/exchange/data_dark']
            flat = file_in['/exchange/data_white']
            ndark = dark.shape[0]
            nflat = flat.shape[0]

            sizes['ndark'] = ndark
            sizes['nflat'] = nflat
            
        return sizes

    def read_theta(self):
        """Read projection angles (in radians)"""

        with h5py.File(self.args.file_name) as file_in:
            theta = file_in['/exchange/theta'][:].astype('float32')/180*np.pi

        return theta

    def read_data_chunk_to_queue(self, data_queue, ids_proj, st_z, end_z, st_n, end_n, id_z, in_dtype):
        '''
        Read a data chunk (proj, flat,dark) from the storage to a python queue, with downsampling
        Input:

        data_queue - a python queue for synchronous read/writes
        ids_proj - the first and last projection ids for reconstruction (e.g, (0,1500)), or np.array with ids (if some angles are blocked and should be ignored),
        st_z - start row in z
        end_z - end row in z
        st_n - start column in x
        end_n - end column in x
        id_z - chunk id (for ordering after parallel processing) 
        id_dtype - input data type (e.g. uint8), or reconstruction type (if binning>0)
        '''

        with h5py.File(self.args.file_name) as fid:
            if isinstance(ids_proj, np.ndarray):
                data = fid['/exchange/data'][ids_proj, st_z:end_z,
                                             st_n:end_n].astype(in_dtype, copy=False)
            else:
                data = fid['/exchange/data'][ids_proj[0]:ids_proj[1],
                                             st_z:end_z, st_n:end_n].astype(in_dtype, copy=False)
                                             

        with h5py.File(self.args.flat_dark_file_name) as fid:
            data_flat = fid['/exchange/data_white'][:,
                                                    st_z:end_z, st_n:end_n].astype(in_dtype, copy=False)
            data_dark = fid['/exchange/data_dark'][:,
                                                   st_z:end_z, st_n:end_n].astype(in_dtype, copy=False)

            item = {}
            item['data'] = utils.downsample(data, self.args.binning)
            item['flat'] = utils.downsample(data_flat, self.args.binning)
            item['dark'] = utils.downsample(data_dark, self.args.binning)
            item['id'] = id_z
            data_queue.put(item)

    def read_proj_chunk(self, data, st_proj, end_proj, st_z, end_z, st_n, end_n):
        """Read a chunk of projections with binning"""
        
        with h5py.File(self.args.file_name) as fid:
            d = fid['/exchange/data'][self.args.start_proj +
                                      st_proj:self.args.start_proj+end_proj, st_z:end_z, st_n:end_n]
            data[st_proj:end_proj] = utils.downsample(d, self.args.binning)

    def read_flat_dark(self, st_n, end_n):
        """Read flat and dark"""

        with h5py.File(self.args.flat_dark_file_name) as fid:
            dark = fid['/exchange/data_dark'][:,
                                              self.args.start_row:self.args.end_row, st_n:end_n]
            flat = fid['/exchange/data_white'][:,
                                               self.args.start_row:self.args.end_row, st_n:end_n]
            flat = utils.downsample(flat, self.args.binning)
            dark = utils.downsample(dark, self.args.binning)
            return flat, dark

    def read_pairs(self, pairs, st_z, end_z, st_n, end_n):
        """Read projection pairs for automatic search of the rotation center. E.g. pairs=[0,1499] for the regular 180 deg dataset [1500,2048,2448]. """

        with h5py.File(self.args.file_name) as fid:
            d = fid['/exchange/data'][pairs, st_z:end_z, st_n:end_n]
            data = utils.downsample(d, self.args.binning)
            return data
