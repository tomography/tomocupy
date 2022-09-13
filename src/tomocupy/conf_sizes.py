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

        sizes = self.reader.read_sizes()
        theta = self.reader.read_theta()

        nproji = sizes['nproji']
        nzi = sizes['nzi']
        ni = sizes['ni']
        nflat = sizes['nflat']
        ndark = sizes['ndark']
        dtype = sizes['dtype']

        if self.args.binning > 0:
            in_dtype = self.args.dtype
        else:
            in_dtype = dtype

        if (self.args.end_row == -1):
            self.args.end_row = nzi
        if (self.args.end_proj == -1):
            self.args.end_proj = nproji
        st_n = 0
        end_n = ni

        # define chunk size for processing
        ncz = self.args.nsino_per_chunk
        if ncz == 1 and self.args.reconstruction_algorithm == 'fourierrec':
            ncz = 2

        ncproj = self.args.nproj_per_chunk
        # take center
        centeri = self.args.rotation_axis
        if centeri == -1:
            centeri = ni/2
        # update sizes wrt binning
        ni //= 2**self.args.binning
        centeri /= 2**self.args.binning
        self.args.crop = int(self.args.crop/2**self.args.binning)

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

        if self.args.dtype == 'float16':
            center += (2**int(np.log2(ni))-ni)/2
            st_n = (ni-2**int(np.log2(ni)))//2
            end_n = st_n+2**int(np.log2(ni))
            ni = 2**int(np.log2(ni))
            n = 2**int(np.log2(n))

            log.warning(
                f'Crop data to the power of 2 sizes to work with 16bit precision, output size in x dimension {ni}')

        # blocked views fix
        ids_proj = [self.args.start_proj, self.args.end_proj]
        theta = theta[ids_proj[0]:ids_proj[1]]

        if self.args.blocked_views:
            st = self.args.blocked_views_start
            end = self.args.blocked_views_end
            ids = np.where(((theta) % np.pi < st) +
                           ((theta-st) % np.pi > end-st))[0]
            theta = theta[ids]
            ids_proj = np.arange(ids_proj[0], ids_proj[1])[ids]

        nproj = len(theta)

        if self.args.end_row == -1:
            nz = nzi-self.args.start_row
        else:
            nz = self.args.end_row-self.args.start_row

        nz //= 2**self.args.binning

        nzchunk = int(np.ceil(nz/ncz))
        lzchunk = np.minimum(
            ncz, np.int32(nz-np.arange(nzchunk)*ncz))  # chunk sizes
        ntchunk = int(np.ceil(nproj/ncproj))
        ltchunk = np.minimum(
            ncproj, np.int32(nproj-np.arange(ntchunk)*ncproj))  # chunk sizes in proj

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
            save_centers = (self.centeri - shift_array)*2**self.args.binning
            if (self.args.file_type == 'double_fov') and (self.centeri < self.ni//2):
                shift_array = -shift_array
        elif self.args.reconstruction_type == 'try_lamino':
            shift_array = np.arange(-self.args.lamino_search_width,
                                    self.args.lamino_search_width, self.args.lamino_search_step).astype('float32')
            save_centers = self.args.lamino_angle + shift_array

        nschunk = int(np.ceil(len(shift_array)/self.ncz))
        lschunk = np.minimum(self.ncz, np.int32(
            len(shift_array)-np.arange(nschunk)*self.ncz))  # chunk sizes
        self.shift_array = shift_array
        self.save_centers = save_centers
        self.nschunk = nschunk
        self.lschunk = lschunk
        self.id_slice = int(self.args.nsino*(self.nz*2**self.args.binning-1) /
                            2**self.args.binning)*2**self.args.binning

    def init_sizes_lamino(self):
        """Calculating sizes for laminography reconstruction by chunks"""

        # take reconstruction height
        rh = int(np.ceil((self.nz*2**self.args.binning/np.cos(self.args.lamino_angle/180*np.pi))/2**self.args.binning)) * \
            2**self.args.binning
        rh //= 2**self.args.binning
        nrchunk = int(np.ceil(rh/self.ncz))
        lrchunk = np.minimum(
            self.ncz, np.int32(rh-np.arange(nrchunk)*self.ncz))
        self.nrchunk = nrchunk
        self.lrchunk = lrchunk
        self.rh = rh
        self.lamino_angle = self.args.lamino_angle
