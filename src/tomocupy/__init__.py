from pkg_resources import get_distribution, DistributionNotFound

__version__ = '0.2'

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
    
# from tomocupy.config import *
# from tomocupy.lprec import *
# from tomocupy.fourierrec import *
# from tomocupy.line_summation import *
# from tomocupy.logging import *
# from tomocupy.rec import *
# from tomocupy.rec_steps import *
# from tomocupy.find_center import *
# from tomocupy.remove_stripe import *
# from tomocupy.retrieve_phase import *
# from tomocupy.utils import *
# from tomocupy.conf_io import *
# from tomocupy.tomo_functions import *

