from tomocupy import config
from tomocupy import logging
from tomocupy import utils
import numpy as np
import numexpr as ne
import h5py
import os
import sys
import tifffile

log = logging.getLogger(__name__)


class ConfIO():
    '''
    Class for configuring processing sizes and read/write oeprations.
    '''

    def __init__(self, args):
        self.args = args
        self.init_sizes()
        if self.args.reconstruction_type[:3] == 'try':
            self.init_sizes_try()
            self.init_output_files_try()
        else:
            self.init_output_files()
        if self.args.lamino_angle!=0:
            self.init_sizes_lamino()

    def init_sizes(self):
        """Calculating and adjusting sizes for reconstruction by chunks"""

        file_in = h5py.File(self.args.file_name)
        data = file_in['/exchange/data']
        dark = file_in['/exchange/data_dark']
        flat = file_in['/exchange/data_white']
        theta = file_in['/exchange/theta'][:].astype('float32')/180*np.pi
        # determine sizes
        nz, ni = data.shape[1:]
        ndark = dark.shape[0]
        nflat = flat.shape[0]

        if self.args.binning > 0:
            in_dtype = self.args.dtype
        else:
            in_dtype = data.dtype

        if (self.args.end_row == -1):
            self.args.end_row = data.shape[1]
        if (self.args.end_proj == -1):
            self.args.end_proj = data.shape[0]

        # define chunk size for processing
        ncz = self.args.nsino_per_chunk
        if ncz==1 and self.args.reconstruction_algorithm=='fourierrec':
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

        stn = 0
        endn = data.shape[-1]

        if self.args.dtype == 'float16':
            center += (2**int(np.log2(ni))-ni)/2
            stn = (ni-2**int(np.log2(ni)))//2
            endn = stn+2**int(np.log2(ni))
            ni = 2**int(np.log2(ni))
            n = 2**int(np.log2(n))

            log.warning(
                f'Crop data to the power of 2 sizes to work with 16bit precision, output size in x dimension {ni}')

        # blocked views fix
        ids_proj = np.arange(len(theta))[
            self.args.start_proj:self.args.end_proj]
        theta = theta[ids_proj]

        if self.args.blocked_views:
            st = self.args.blocked_views_start
            end = self.args.blocked_views_end
            ids = np.where(((theta) % np.pi < st) +
                           ((theta-st) % np.pi > end-st))[0]
            theta = theta[ids]
            ids_proj = ids_proj[ids]

        nproj = len(theta)

        if self.args.end_row == -1:
            nz = nz-self.args.start_row
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
        self.stn = stn
        self.endn = endn
        self.data = data
        self.dark = dark
        self.flat = flat

    def init_sizes_try(self):
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
        self.idslice = int(self.args.nsino*(self.nz*2**self.args.binning-1) /
                           2**self.args.binning)*2**self.args.binning

    def init_sizes_lamino(self):
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
                self.data.shape[1]//2**self.args.binning, self.n, self.n), dtype=self.dtype)
            os.system(f'mkdir -p {fnameout[:-3]}_parts')
            for k in range(self.nzchunk):
                filename = f"{fnameout[:-3]}_parts/p{k:04d}.h5"
                vsource = h5py.VirtualSource(
                    filename, "/exchange/data", shape=(self.lzchunk[k], self.n, self.n), dtype=self.dtype)
                st = self.args.start_row//2**self.args.binning+k*self.ncz
                layout[st:st+self.lzchunk[k]] = vsource

            # Add virtual dataset to output file
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

    def downsample(self, data):
        """Downsample data"""

        data = data.astype(self.in_dtype, copy=False)
        for j in range(self.args.binning):
            x = data[:, :, ::2]
            y = data[:, :, 1::2]
            data = ne.evaluate('x + y')  # should use multithreading
        for k in range(self.args.binning):
            x = data[:, ::2]
            y = data[:, 1::2]
            data = ne.evaluate('x + y')
        return data

    def read_data_chunk(self, data_queue, k):
        """Reading the kth data chunk from hard disk"""

        item = {}
        st = self.args.start_row+k*self.ncz*2**self.args.binning
        end = self.args.start_row + \
            (k*self.ncz+self.lzchunk[k])*2**self.args.binning
        item['data'] = self.downsample(
            self.data[:,  st:end, self.stn:self.endn])
        if len(self.theta) != item['data'].shape[0]:
            item['data'] = item['data'][self.ids_proj]
        item['flat'] = self.downsample(
            self.flat[:,  st:end, self.stn:self.endn])
        item['dark'] = self.downsample(
            self.dark[:,  st:end, self.stn:self.endn])
        item['id'] = k
        data_queue.put(item)

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

    def read_data_to_queue(self, data_queue, read_threads):
        """Reading data from hard disk and putting it to a queue"""
        for k in range(self.nzchunk):
            ithread = utils.find_free_thread(read_threads)
            read_threads[ithread].run(self.read_data_chunk, (data_queue, k))

    def read_data_try(self):
        """Read one slice for try reconstruction"""

        log.info(
            f'Try rotation center reconstruction for slice {self.idslice}')
        data = self.data[:, self.idslice:self.idslice +
                         2**self.args.binning][self.ids_proj]
        dark = self.dark[:, self.idslice:self.idslice+2**self.args.binning]
        flat = self.flat[:, self.idslice:self.idslice+2**self.args.binning]
        data = self.downsample(data)
        flat = self.downsample(flat)
        dark = self.downsample(dark)
        return data, flat, dark

    def write_data_try(self, rec, cid):
        """Write tiff reconstruction with a given name"""

        if self.args.crop > 0:
            rec = rec[self.args.crop:-self.args.crop,
                      self.args.crop:-self.args.crop]
        tifffile.imwrite(f'{self.fnameout}_{cid:05.2f}.tiff', rec)

    def read_data(self, data, k, lchunk):
        """Read a chunk of projections with binning"""

        d = self.data[self.args.start_proj+k*lchunk:self.args.start_proj +
                      (k+1)*lchunk, self.args.start_row:self.args.end_row, self.stn:self.endn]
        data[k*lchunk:(k+1)*lchunk] = self.downsample(d)

    def read_flat_dark(self):
        """Read flat and dark"""
        # read dark and flat, binning
        dark = self.dark[:, self.args.start_row:self.args.end_row,
                         self.stn:self.endn]
        flat = self.flat[:, self.args.start_row:self.args.end_row,
                         self.stn:self.endn]
        flat = self.downsample(flat)
        dark = self.downsample(dark)
        return flat, dark

    def read_pairs(self, pairs):
        """Read pairs for checking rotation center"""

        d = self.data[pairs, self.args.start_row:self.args.end_row,
                      self.stn:self.endn]
        data = self.downsample(d)
        return data
