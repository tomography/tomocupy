from pkg_resources import get_distribution, DistributionNotFound

from tomocupy_cli.lprec.lprec import LpRec
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass