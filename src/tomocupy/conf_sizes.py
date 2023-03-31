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

from tomocupy import logging
import numpy as np
from ast import literal_eval
        
log = logging.getLogger(__name__)


class ConfSizes():
    '''
    Class for configuring processing sizes
    '''

    def __init__(self, args, reader):
        self.args = args
        self.reader = reader

        self.init_sizes()
        if self.args.reconstruction_type[:3] == 'try':
            self.init_sizes_try()
        if self.args.lamino_angle != 0:
            self.init_sizes_lamino()

    def init_sizes(self):
        """Calculating and adjusting sizes for reconstruction by chunks"""

        # read data sizes and projection angles with a reader
        sizes = self.reader.read_sizes()
        theta = self.reader.read_theta()
        nproji = sizes['nproji']
        nzi = sizes['nzi']
        ni = sizes['ni']
        nflat = sizes['nflat']
        ndark = sizes['ndark']
        dtype = sizes['dtype']

        # adjust data type for input data
        if self.args.binning > 0:
            in_dtype = self.args.dtype
        else:
            in_dtype = dtype

        # adjust the last row and proj ids, so as first and last corrdinates in x
        if (self.args.end_row == -1):
            self.args.end_row = nzi
        if (self.args.end_proj == -1):
            self.args.end_proj = nproji
        if (self.args.end_column == -1):
            self.args.end_column = ni
        
        # find numebr of rows
        nz = self.args.end_row-self.args.start_row
        
        # define z chunk size for processing        
        ncz = self.args.nsino_per_chunk
        if ncz == 1 and self.args.reconstruction_algorithm == 'fourierrec':
            ncz = 2 # 2 rows are processed at the same time since fourierrec works with complex numbers
                
        
        # define projection chunk size for processing
        ncproj = self.args.nproj_per_chunk
        

        centeri = self.args.rotation_axis
        if centeri == -1:
            centeri = ni/2
        
        st_n = self.args.start_column
        end_n = self.args.end_column
        
        ni = end_n - st_n        
        centeri -= st_n
        
        # update sizes wrt binning
        ni //= 2**self.args.binning
        centeri /= 2**self.args.binning
        nz //= 2**self.args.binning
        
        # change sizes for 360 deg scans with rotation axis at the border
        if(self.args.file_type == 'double_fov'):
            n = 2*ni
            if(centeri < ni//2):
                # if rotation center is on the left side of the ROI
                center = ni-centeri
            else:
                center = centeri
        else:
            n = ni
            center = centeri

        # adjust sizes for 16bit processing
        if self.args.dtype == 'float16':
            center += (2**int(np.log2(ni))-ni)/2
            st_n = (ni-2**int(np.log2(ni)))//2
            end_n = st_n+2**int(np.log2(ni))
            ni = 2**int(np.log2(ni))
            n0 = n
            n = 2**int(np.log2(n))

            if n!=n0:
                log.warning(
                    f'Crop data to the power of 2 sizes to work with 16bit precision, output size in x dimension {ni}')
        
        # blocked views fix
        ids_proj = [self.args.start_proj, self.args.end_proj]
        theta = theta[ids_proj[0]:ids_proj[1]]
        if self.args.blocked_views !='none':            
            tmp = literal_eval(self.args.blocked_views)
            if not isinstance(tmp[0],list):
                tmp = [tmp]
            ids = np.arange(len(theta))
            for pairs in tmp:
                [st,end] = pairs
                ids = np.intersect1d(ids,np.where(((theta) % np.pi < st) +
                    ((theta-st) % np.pi > end-st))[0])
            theta = theta[ids]
            ids_proj = np.arange(ids_proj[0], ids_proj[1])[ids]
            log.info(f'angles {theta}')
        nproj = len(theta)        
        
        # calculate chunks
        nzchunk = int(np.ceil(nz/ncz))
        lzchunk = np.minimum(
            ncz, np.int32(nz-np.arange(nzchunk)*ncz))  # chunk sizes
        ntchunk = int(np.ceil(nproj/ncproj))
        ltchunk = np.minimum(
            ncproj, np.int32(nproj-np.arange(ntchunk)*ncproj))  # chunk sizes in proj

        tmp = literal_eval(self.args.nsino)
        if not isinstance(tmp, list):
            tmp = [tmp]  
        self.id_slices = np.int32(np.array(tmp)*(nz*2**self.args.binning-1) /
                            2**self.args.binning)*2**self.args.binning
        self.n = n
        self.nz = nz
        self.ncz = ncz
        self.nproj = nproj
        self.ncproj = ncproj
        self.center = center
        self.ni = ni
        self.nzi = nzi
        self.centeri = centeri
        self.ndark = ndark
        self.nflat = nflat
        self.ids_proj = ids_proj
        self.theta = theta
        self.nzchunk = nzchunk
        self.lzchunk = lzchunk
        self.ntchunk = ntchunk
        self.ltchunk = ltchunk
        self.dtype = self.args.dtype
        self.in_dtype = in_dtype
        self.st_n = st_n
        self.end_n = end_n

    def init_sizes_try(self):
        """Calculating sizes for try reconstruction by chunks"""

        if self.args.reconstruction_type == 'try':
            # invert shifts for calculations if centeri<ni for double_fov
            shift_array = np.arange(-self.args.center_search_width,
                                    self.args.center_search_width, self.args.center_search_step*2**self.args.binning).astype('float32')/2**self.args.binning
            save_centers = (self.centeri - shift_array)*2**self.args.binning+self.st_n
            if (self.args.file_type == 'double_fov') and (self.centeri < self.ni//2):
                shift_array = -shift_array
            
        elif self.args.reconstruction_type == 'try_lamino':
            shift_array = np.arange(-self.args.lamino_search_width,
                                    self.args.lamino_search_width, self.args.lamino_search_step).astype('float32')
            save_centers = self.args.lamino_angle + shift_array
        # calculate chunks
        nschunk = int(np.ceil(len(shift_array)/self.ncz))
        lschunk = np.minimum(self.ncz, np.int32(
            len(shift_array)-np.arange(nschunk)*self.ncz)) 
        self.shift_array = shift_array
        self.save_centers = save_centers
        self.nschunk = nschunk
        self.lschunk = lschunk        

    def init_sizes_lamino(self):
        """Calculating sizes for laminography reconstruction by chunks"""

        # calculate reconstruction height        
        if self.args.lamino_end_row == -1:
            rh = int(np.ceil((self.nz*2**self.args.binning/np.cos(self.args.lamino_angle/180*np.pi))/2**self.args.binning)) - self.args.lamino_start_row//2**self.args.binning
        else:
            rh = self.args.lamino_end_row//2**self.args.binning - self.args.lamino_start_row//2**self.args.binning
        
        # calculate chunks
        nrchunk = int(np.ceil(rh/self.ncz))
        lrchunk = np.minimum(
            self.ncz, np.int32(rh-np.arange(nrchunk)*self.ncz))
            
        self.nrchunk = nrchunk
        self.lrchunk = lrchunk
        self.rh = rh
        self.lamino_angle = self.args.lamino_angle
