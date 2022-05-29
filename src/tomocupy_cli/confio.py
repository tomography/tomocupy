from tomocupy_cli import logging
import numpy as np
import numexpr as ne
import h5py
import dxchange
import os
import sys

log = logging.getLogger(__name__)


class ConfIO():
    '''
    Class for configuring processing sizes and read/write oeprations.
    '''

    def __init__(self, args):
        self.args = args
        self.init_sizes()
        self.init_output_files()

    def init_sizes(self):
        """Calculating and adjusting sizes for reconstruction by chunks"""

        file_in = h5py.File(self.args.file_name)
        # determine sizes
        nz, ni = file_in['/exchange/data'].shape[1:]
        ndark = file_in['/exchange/data_dark'].shape[0]
        nflat = file_in['/exchange/data_white'].shape[0]
        theta = file_in['/exchange/theta'][:].astype('float32')/180*np.pi
        in_dtype = file_in['exchange/data'].dtype
        if (self.args.end_row == -1):
            self.args.end_row = file_in['/exchange/data'].shape[1]
        if (self.args.end_proj == -1):
            self.args.end_proj = file_in['/exchange/data'].shape[0]

        # define chunk size for processing
        ncz = self.args.nsino_per_chunk
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
        endn = ni
        
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

        # calculate chunks
        if self.args.reconstruction_type == 'try':
            # reconstruct one slice with different centers
            shift_array = np.arange(-self.args.center_search_width,
                                    self.args.center_search_width, self.args.center_search_step*2**self.args.binning).astype('float32')/2**self.args.binning

            nchunk = int(np.ceil(len(shift_array)/(ncz)))
            lchunk = np.minimum(ncz, np.int32(
                len(shift_array)-np.arange(nchunk)*ncz))  # chunk sizes
            self.shift_array = shift_array
        else:
            if self.args.end_row == -1:
                nz = nz-self.args.start_row
            else:
                nz = self.args.end_row-self.args.start_row
            nchunk = int(np.ceil(nz/2**self.args.binning/ncz))
            lchunk = np.minimum(
                ncz, np.int32(nz/2**self.args.binning-np.arange(nchunk)*ncz))  # chunk sizes
        
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
        self.nchunk = nchunk
        self.lchunk = lchunk
        self.file_in = file_in
        self.in_dtype = in_dtype
        self.stn = stn
        self.endn = endn

    def init_output_files(self):
        """Constructing output file names and initiating the actual files"""

        if self.args.reconstruction_type == 'try':
            fnameout = os.path.dirname(
                self.args.file_name)+'_rec/try_center/'+os.path.basename(self.args.file_name)[:-3]
            os.system(f'mkdir -p {fnameout}')
            fnameout += '/recon'
        else:
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
                    self.file_in['exchange/data'].shape[1]//2**self.args.binning, self.n, self.n), dtype=self.args.dtype)
                os.system(f'mkdir -p {fnameout[:-3]}_parts')
                for k in range(self.nchunk):
                    filename = f"{fnameout[:-3]}_parts/p{k:04d}.h5"
                    vsource = h5py.VirtualSource(
                        filename, "/exchange/recon", shape=(self.lchunk[k], self.n, self.n), dtype=self.args.dtype)
                    st = self.args.start_row//2**self.args.binning+k*self.ncz
                    layout[st:st+self.lchunk[k]] = vsource
                # Add virtual dataset to output file
                rec_virtual = h5py.File(fnameout, "w")
                rec_virtual.create_virtual_dataset("/exchange/recon", layout)
                # saving command line for reconstruction
                rec_line = sys.argv
                # remove full path to the file
                rec_line[0] = os.path.basename(rec_line[0])
                s = ' '.join(rec_line).encode("utf-8")
                # save as an attribute in hdf5 file
                rec_virtual.attrs["rec_line"] = np.array(
                    s, dtype=h5py.string_dtype('utf-8', len(s)))

        self.fnameout = fnameout
        log.info(f'Output: {fnameout}')

    def read_data_to_queue(self, data_queue):
        """Reading data from hard disk and putting it to a queue"""

        # init references to data, no reading at this point
        data = self.file_in['exchange/data']
        dark = self.file_in['exchange/data_dark']
        flat = self.file_in['exchange/data_white']

        for k in range(self.nchunk):
            item = {}
            st = self.args.start_row+k*self.ncz*2**self.args.binning
            end = self.args.start_row + \
                (k*self.ncz+self.lchunk[k])*2**self.args.binning            
            item['data'] = self.downsample(data[:,  st:end, self.stn:self.endn])[
                self.ids_proj]
            item['flat'] = self.downsample(flat[:,  st:end, self.stn:self.endn])
            item['dark'] = self.downsample(dark[:,  st:end, self.stn:self.endn])

            data_queue.put(item)

    def write_data(self, rec, k):
        """Writing the kth data chunk to hard disk"""

        if self.args.crop > 0:
            rec_pinned0 = rec_pinned0[:, self.args.crop:-
                                      self.args.crop, self.args.crop:-self.args.crop]

        if self.args.save_format == 'tiff':
            dxchange.write_tiff_stack(rec, fname=self.fnameout, start=k *
                                      self.ncz+self.args.start_row//2**self.args.binning, overwrite=True)
        elif self.args.save_format == 'h5':
            filename = f"{self.fnameout[:-3]}_parts/p{k:04d}.h5"
            with h5py.File(filename, "w") as fid:
                fid.create_dataset("/exchange/recon", data=rec,
                                   chunks=(1, self.n, self.n))

    def downsample(self, data):
        """Downsample data"""

        data = data.astype(self.args.dtype)
        for j in range(self.args.binning):
            x = data[:, :, ::2]
            y = data[:, :, 1::2]
            data = ne.evaluate('x + y')  # should use multithreading
        for k in range(self.args.binning):
            x = data[:, ::2]
            y = data[:, 1::2]
            data = ne.evaluate('x + y')
        return data

    def write_h5(self, data, filename):
        """Save data to an hdf5"""

        with h5py.File(filename, "w") as fid:
            fid.create_dataset("/exchange/recon", data=data,
                               chunks=(1, self.n, self.n))

    def read_data_try(self):
        """Read one slice for try reconstruction"""

        fid = h5py.File(self.args.file_name, 'r')
        data = fid['exchange/data']
        dark = fid['exchange/data_dark']
        flat = fid['exchange/data_white']
        idslice = int(
            self.args.nsino*(data.shape[1]-1)/2**self.args.binning)*2**self.args.binning
        log.info(f'Try rotation center reconstruction for slice {idslice}')
        data = data[:, idslice:idslice+2**self.args.binning][self.ids_proj]
        dark = dark[:, idslice:idslice+2 **
                    self.args.binning].astype(self.args.dtype)
        flat = flat[:, idslice:idslice+2 **
                    self.args.binning].astype(self.args.dtype)
        data = np.append(data, data, 1)
        dark = np.append(dark, dark, 1)
        flat = np.append(flat, flat, 1)
        data = self.downsample(data)
        flat = self.downsample(flat)
        dark = self.downsample(dark)
        return data, flat, dark

    def write_data_try(self, rec, cid):
        """Write tiff reconstruction with a given name"""

        if self.args.crop > 0:
            rec = rec[self.args.crop:-self.args.crop,
                      self.args.crop:-self.args.crop]
        dxchange.write_tiff(
            rec, f'{self.fnameout}_{cid:05.2f}.tiff', overwrite=True)

    def read_data(self, data, k, lchunk):
        """Read a chunk of projection with binning"""
        d = self.file_in['exchange/data'][self.args.start_proj+k*lchunk:self.args.start_proj +
                                          (k+1)*lchunk, self.args.start_row:self.args.end_row,self.stn:self.endn]
        data[k*lchunk:self.args.start_proj+(k+1)*lchunk] = self.downsample(d)

    def read_flat_dark(self):
        """Read flat and dark"""
        # read dark and flat, binning
        dark = self.file_in['exchange/data_dark'][:,self.args.start_row:self.args.end_row,self.stn:self.endn].astype(self.args.dtype)
        flat = self.file_in['exchange/data_white'][:,self.args.start_row:self.args.end_row,self.stn:self.endn].astype(self.args.dtype)
        flat = self.downsample(flat)
        dark = self.downsample(dark)

        return flat, dark
