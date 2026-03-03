import sys
import pathlib
import dxchange
import tomocupy

from tomocupy.dataio import reader
from tomocupy.dataio import writer
from tomocupy.global_vars import args

def read_aps(fname):

    _, meta_dict = dxchange.read_hdf_meta(fname)

    params_dict = {}
    for section in tomocupy.config.RECON_STEPS_PARAMS:
        for key, value in tomocupy.config.SECTIONS[section].items():
            key = key.replace('-', '_')
            params_dict[key] = value['default']

    # create a parameter object identical to the one passed using the CLI
    global args
    args.__dict__.update(params_dict)
    
    # set only the parameters that are different from the default
    args.reconstruction_type          = 'try'
    args.file_name                    = fname
    args.rotation_axis_auto           = 'auto'
    args.out_path_name                = '/data/tmpfdc/' 
    args.clear_folder                 = True
    args.fbp_filter                   = 'shepp' 
    args.retrieve_phase_method        = None 
    args.remove_stripe_method         = 'vo'
    args.pixel_size                   = meta_dict['measurement_instrument_detection_system_objective_resolution'][0] * 1e-4
    args.propagation_distance         = meta_dict['measurement_instrument_detector_motor_stack_setup_z'][0]
    args.energy                       = meta_dict['measurement_instrument_monochromator_energy'][0]
    args.retrieve_phase_alpha         = 0.0008

def main():

    if len(sys.argv) == 1:
        print ('ERROR: Must provide an hdf file name')
        print ('Example:')
        print ('        python readrec_aps.py /data/aps_dataset.h5')
        sys.exit(1)
    else:
        file_name = sys.argv[1]
        p = pathlib.Path(file_name)
        if p.is_file():
            read_aps(file_name)
            
            cl_reader = reader.Reader()
            cl_writer = writer.Writer()

            clrotthandle = tomocupy.FindCenter(cl_reader)
            args.rotation_axis = clrotthandle.find_center()*2**args.binning
            print(f'set rotaion  axis {args.rotation_axis}')

            clpthandle = tomocupy.GPURec(cl_reader, cl_writer)
            clpthandle.recon_all()

            print('Done!')


        else:
            print('ERROR: %s does not exist' % p)
            sys.exit(1)


if __name__ == "__main__":
   main()