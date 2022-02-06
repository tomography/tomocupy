from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
    
from tomocupy_cli.config import *
from tomocupy_cli.find_rotation import *
from tomocupy_cli.fourierrec import *
from tomocupy_cli.logging import *
from tomocupy_cli.lprec import *
from tomocupy_cli.rec import *
from tomocupy_cli.rec_steps import *
from tomocupy_cli.remove_stripe import *
from tomocupy_cli.retrieve_phase import *
from tomocupy_cli.utils import *

