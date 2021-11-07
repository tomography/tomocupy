from pkg_resources import get_distribution, DistributionNotFound

from h5gpurec.solver import H5GPURec
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass