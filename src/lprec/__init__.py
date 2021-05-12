from pkg_resources import get_distribution, DistributionNotFound

from lprec.solver import LpRec

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass