from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
    
from tomocupyfp16_cli.config import *
from tomocupyfp16_cli.find_rotation import *
from tomocupyfp16_cli.fourierrec import *
from tomocupyfp16_cli.logging import *
from tomocupyfp16_cli.rec import *
from tomocupyfp16_cli.rec_steps import *
from tomocupyfp16_cli.remove_stripe import *
from tomocupyfp16_cli.retrieve_phase import *
from tomocupyfp16_cli.utils import *

