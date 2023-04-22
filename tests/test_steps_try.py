import unittest
import os
import numpy as np
import tifffile
import inspect

prefix = 'tomocupy recon_steps --file-name data/test_data.h5 --rotation-axis 782.5 --nsino-per-chunk 4'
cmd_dict = {
    f'{prefix} --reconstruction-type try --lamino-angle 1 --reconstruction-algorithm linerec': 14.023,
    f'{prefix} --reconstruction-type try_lamino --lamino-angle 1 --reconstruction-algorithm linerec': 37.060,
    f'{prefix} --reconstruction-type try --rotate-proj-angle 1 --nsino [0,0.5]': 28.004,
    f'{prefix} --reconstruction-type try --lamino-angle 1 --rotate-proj-angle 1 --nsino [0,0.5]': 26.180,    
    f'{prefix} --reconstruction-type try --binning 1 --rotate-proj-angle 1 --nsino [0,0.5]': 11.103,    
}


class SequentialTestLoader(unittest.TestLoader):
    def getTestCaseNames(self, testCaseClass):
        test_names = super().getTestCaseNames(testCaseClass)
        testcase_methods = list(testCaseClass.__dict__.keys())
        test_names.sort(key=testcase_methods.index)
        return test_names


class Tests(unittest.TestCase):
    def test_full_recon(self):
        for cmd in cmd_dict.items():
            os.system('rm -rf data_rec')
            print(f'TEST {inspect.stack()[0][3]}: {cmd[0]}')
            st = os.system(cmd[0])
            self.assertEqual(st, 0)
            ssum = 0
            for k in np.arange(758.5, 778.5, 0.5):
                try:
                    ssum += np.linalg.norm(tifffile.imread(
                        f'data_rec/try_center/test_data/recon_slice0010_center{k:05.2f}.tiff'))
                    ssum += np.linalg.norm(tifffile.imread(
                        f'data_rec/try_center/test_data/recon_slice0000_center{k:05.2f}.tiff'))
                except:
                    pass
            for k in np.arange(-4, 6, 0.25):
                try:
                    ssum += np.linalg.norm(tifffile.imread(
                        f'data_rec/try_center/test_data/recon_slice0010_center{k:05.2f}.tiff'))
                except:
                    pass
            self.assertAlmostEqual(ssum, cmd[1], places=1)            
                
if __name__ == '__main__':
    unittest.main(testLoader=SequentialTestLoader(), failfast=True)
