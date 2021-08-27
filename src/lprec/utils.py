from itertools import islice
import numpy as np

#timing functions
def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
       return time.time() - startTime_for_tictoc