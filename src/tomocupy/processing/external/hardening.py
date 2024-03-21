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
import re
import numpy as np
import cupy as cp
import h5py
from tomocupy import logging
from tomocupy import utils
from beamhardening import beamhardening as bh

log = logging.getLogger(__name__)

class Beam_Corrector():
    def __init__(self, params):
        print(params)
        params = self.parse_meta(params)

        #Read in the flat and dark
        with h5py.File(params.file_name,'r') as fid:
            flat = fid['/exchange/data_white'][:] 
            dark = fid['/exchange/data_dark'][:] 
        median_flat = np.median(flat, axis=0) - np.median(dark, axis=0)
        self.beam_corr= bh.BeamCorrector(calculate_source = params.calculate_source,
                                        e_storage_ring = params.e_storage_ring,
                                        b_storage_ring = params.b_storage_ring,
                                        minimum_E = params.minimum_E,
                                        maximum_E = params.maximum_E,
                                        step_E = params.step_E,
                                        maximum_psi_urad = params.maximum_psi_urad,
                                        )
        self.beam_corr.add_scintillator(
                            params.scintillator_material,
                            params.scintillator_thickness,
                            params.scintillator_density,
                            )
        self.beam_corr.add_sample(
                            params.sample_material,
                            params.sample_density,
                            )
        self.beam_corr.add_filter(
                            params.filter_1_material,
                            params.filter_1_thickness,
                            params.filter_1_density,
                            )
        self.beam_corr.add_filter(
                            params.filter_2_material,
                            params.filter_2_thickness,
                            params.filter_2_density,
                            )
        self.beam_corr.add_filter(
                            params.filter_3_material,
                            params.filter_3_thickness,
                            params.filter_3_density,
                            )
        self.beam_corr.set_geometry(
                            params.source_distance,
                            params.pixel_size,
                            )
        self.beam_corr.find_angles(median_flat)

        #Put the linear interpolation values in params
        self.beam_corr.compute_interp_values()
        self.interp_angles = cp.array(self.beam_corr.angular_interp_values[0])
        self.interp_corrector = cp.array(self.beam_corr.angular_interp_values[1])
        self.interp_trans = cp.array(self.beam_corr.centerline_interp_values[0])
        self.interp_pathlength = cp.array(self.beam_corr.centerline_interp_values[1])
        self.params = params

    def parse_meta(self, params):
        params = self.read_pixel_size(params)
        params = self.read_filter_materials(params)
        params = self.read_scintillator(params)
        params = utils.read_bright_ratio(params)
        return params

    def correct_centerline(self, data):
        data[:] = cp.interp(data, self.interp_trans, self.interp_pathlength)
        return data

    def correct_angle(self, data, current_rows):
        angles = cp.array(self.beam_corr.angles[current_rows])
        correction = cp.interp(angles, self.interp_angles, self.interp_corrector)
        for i in range(correction.shape[0]):
            data[:,i,:] = data[:,i,:] * correction[i]
        return data

    def read_filter_materials(self, params):
        '''Read the beam filter configuration.
        This discriminates between files created with tomoScan and
        the previous meta data format.
        '''
        if utils.check_item_exists_hdf(params.file_name, '/measurement/instrument/attenuator_1'):
            return self.read_filter_materials_tomoscan(params)
        else:
            return self.read_filter_materials_old(params)


    def read_filter_materials_tomoscan(self, params):
        '''Read the beam filter configuration from the HDF file.
        
        If params.filter_{n}_auto for n in [1,2,3] is True,
        then try to read the filter configuration recorded during
        acquisition in the HDF5 file.
        
        Parameters
        ==========
        params
        
          The global parameter object, should have *filter_n_material*,
          *filter_n_thickness*, and *filter_n_auto* for n in [1,2,3]
        
        Returns
        =======
        params
          An equivalent object to the *params* input, optionally with
          *filter_n_material* and *filter_n_thickness*
          attributes modified to reflect the HDF5 file.
        
        '''
        log.info('  *** auto reading filter configuration')
        # Read the relevant data from disk
        filter_path = '/measurement/instrument/attenuator_{idx}'
        param_path = 'filter_{idx}_{attr}'
        for idx_filter in range(1,4,1):
            if not utils.check_item_exists_hdf(params.file_name, filter_path.format(idx = idx_filter)):
                log.warning('  *** *** Filter {idx} not found in HDF file.  Set this filter to none'
                                        .format(idx = idx_filter))
                setattr(params, param_path.format(idx=idx_filter, attr='material'), 'Al')
                setattr(params, param_path.format(idx=idx_filter, attr='thickness'), 0.0)
                continue
            filter_auto = getattr(params, param_path.format(idx=idx_filter, attr='auto'))
            if filter_auto != 'True' and filter_auto != True:
                log.warning('  *** *** do not auto read filter {n}'.format(n=idx_filter))
                continue
            log.warning('  *** *** auto reading parameters for filter {0}'.format(idx_filter))
            # See if there are description and thickness fields
            if utils.check_item_exists_hdf(params.file_name, filter_path.format(idx = idx_filter) + '/description'):
                filt_material = utils.param_from_dxchange(params.file_name,
                                            filter_path.format(idx=idx_filter) + '/description',
                                            char_array = True, scalar = False)
                filt_thickness = int(utils.param_from_dxchange(params.file_name,
                                            filter_path.format(idx=idx_filter) + '/thickness',
                                            char_array = False, scalar = True))
            else:
                #The filter info is just the raw string from the filter unit.
                log.warning('  *** *** filter {idx} info must be read from the raw string'
                                .format(idx = idx_filter))
                filter_str = utils.param_from_dxchange(params.file_name,
                                            filter_path.format(idx=idx_filter) + '/setup/filter_unit_text',
                                            char_array = True, scalar = False)
                if filter_str is None:
                    log.warning('  *** *** Could not load filter %d configuration from HDF5 file.' % idx_filter)
                    filt_material, filt_thickness = self._filter_str_to_params('Open')
                else: 
                    filt_material, filt_thickness = self._filter_str_to_params(filter_str)

            # Update the params with the loaded values
            setattr(params, param_path.format(idx=idx_filter, attr='material'), filt_material)
            setattr(params, param_path.format(idx=idx_filter, attr='thickness'), filt_thickness)
            log.info('  *** *** Filter %d: (%s %f)' % (idx_filter, filt_material, filt_thickness))
        return params


    def read_filter_materials_old(self, params):
        '''Read the beam filter configuration from the HDF file.
        
        If params.filter_1_material and/or params.filter_2_material are
        'auto', then try to read the filter configuration recorded during
        acquisition in the HDF5 file.
        
        Parameters
        ==========
        params
        
          The global parameter object, should have *filter_1_material*,
          *filter_1_thickness*, *filter_2_material*, and
          *filter_2_thickness* attributes.
        
        Returns
        =======
        params
          An equivalent object to the *params* input, optionally with
          *filter_1_material*, *filter_1_thickness*, *filter_2_material*,
          and *filter_2_thickness* attributes modified to reflect the HDF5
          file.
        
        '''
        log.info('  *** auto reading filter configuration')
        # Read the relevant data from disk
        filter_path = '/measurement/instrument/filters/Filter_{idx}_Material'
        param_path = 'filter_{idx}_{attr}'
        for idx_filter in (1, 2):
            filter_param = getattr(params, param_path.format(idx=idx_filter, attr='material'))
            if filter_param == 'auto':
                # Read recorded filter condition from the HDF5 file
                filter_str = utils.param_from_dxchange(params.file_name,
                                                        filter_path.format(idx=idx_filter),
                                                        char_array=True, scalar=False)
                if filter_str is None:
                    log.warning('  *** *** Could not load filter %d configuration from HDF5 file.' % idx_filter)
                    material, thickness = self._filter_str_to_params('Open')
                else:
                    material, thickness = self._filter_str_to_params(filter_str)
                # Update the params with the loaded values
                setattr(params, param_path.format(idx=idx_filter, attr='material'), material)
                setattr(params, param_path.format(idx=idx_filter, attr='thickness'), thickness)
                log.info('  *** *** Filter %d: (%s %f)' % (idx_filter, material, thickness))
        return params


    def _filter_str_to_params(self, filter_str):
        # Any material with zero thickness is equivalent to being open
        open_filter = ('Al', 0.)
        if filter_str == 'Open':
            # No filter is installed
            material, thickness = open_filter
        else:
            # Parse the filter string to get the parameters
            filter_re = '(?P<material>[A-Za-z_]+)_(?P<thickness>[0-9.]+)(?P<unit>[a-z]*)'
            match = re.match(filter_re, filter_str)
            if match:
                material, thickness, unit = match.groups()
            else:
                log.warning('  *** *** Cannot interpret filter "%s"' % filter_str)
                material, thickness = open_filter
                unit = 'um'
            # Convert strings into numbers
            thickness = float(thickness)
            factors = {
                'nm': 1e-3,
                'um': 1,
                'mm': 1e3,
            }
            try:
                factor = factors[unit]
            except KeyError:
                log.warning('  *** *** Cannot interpret filter unit in "%s"' % filter_str)
                factor = 1
            thickness *= factor
        return material, thickness


    def read_pixel_size(self, params):
        '''
        Read the pixel size and magnification from the HDF file.
        Use to compute the effective pixel size.
        '''
        log.info('  *** auto pixel size reading')
        if params.read_pixel_size != True:
            log.info('  *** *** OFF')
            return params

        if utils.check_item_exists_hdf(params.file_name,
                                    '/measurement/instrument/detection_system/objective/resolution'):
            params.pixel_size = utils.param_from_dxchange(params.file_name,
                                                '/measurement/instrument/detection_system/objective/resolution')
            log.info('  *** *** effective pixel size = {:6.4e} microns'.format(params.pixel_size))
            return(params)
        log.warning('  *** tomoScan resolution parameter not found.  Try old format')
        pixel_size = utils.param_from_dxchange(params.file_name,
                                                '/measurement/instrument/detector/pixel_size_x')
        mag = utils.param_from_dxchange(params.file_name,
                                        '/measurement/instrument/detection_system/objective/magnification')
        #Handle case where something wasn't read right
        if not (pixel_size and mag):
            log.warning('  *** *** problem reading pixel size from the HDF file')
            return params
        #What if pixel size isn't in microns, but in mm or m?
        for i in range(3):
            if pixel_size < 0.5:
                pixel_size *= 1e3
            else:
                break
        params.pixel_size = pixel_size / mag
        log.info('  *** *** effective pixel size = {:6.4e} microns'.format(params.pixel_size))
        return params


    def read_scintillator(self, params):
        '''Read the scintillator type and thickness from the HDF file.
        '''
        if params.read_scintillator:
            log.info('  *** auto reading scintillator params')
            possible_names = ['/measurement/instrument/detection_system/scintillator/scintillating_thickness',
                            '/measurement/instrument/detection_system/scintillator/active_thickness']
            for pn in possible_names:
                if utils.check_item_exists_hdf(params.file_name, pn):
                    val = utils.param_from_dxchange(params.file_name,
                                             pn, attr=None,
                                             scalar=True,
                                             char_array=False)
                    params.scintillator_thickness = float(val)
                    break
            log.info('  *** *** scintillator thickness = {:f}'.format(params.scintillator_thickness))
            possible_names = ['/measurement/instrument/detection_system/scintillator/name',
                            '/measurement/instrument/detection_system/scintillator/type',
                            '/measurement/instrument/detection_system/scintillator/description']
            scint_material_string = ''
            for pn in possible_names:
                if utils.check_item_exists_hdf(params.file_name, pn):
                    scint_material_string = utils.param_from_dxchange(params.file_name,
                                                pn, scalar = False, char_array = True)
                    break
            else:
                log.warning('  *** *** no scintillator material found')
                return(params)
            if scint_material_string.lower().startswith('luag'):
                params.scintillator_material = 'Lu3Al5O12'
                params.scintillator_density = 6.9
            elif scint_material_string.lower().startswith('lyso'):
                params.scintillator_material = 'LYSO_Ce'
            elif scint_material_string.lower().startswith('yag'):
                params.scintillator_material = 'YAG_Ce' 
            else:
                log.warning('  *** *** scintillator {:s} not recognized!'.format(scint_material_string))
            log.info('  *** *** using scintillator {:s}'.format(params.scintillator_material))
        return params 
