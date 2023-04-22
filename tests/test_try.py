import unittest
import os
import numpy as np
import tifffile
import inspect

prefix = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type try --rotation-axis 782.5 --nsino-per-chunk 4'
cmd_dict = {
    f'{prefix} ': 13.98,
    f'{prefix} --nsino [0,0.5] ': 28.36,
    f'{prefix} --reconstruction-algorithm lprec ': 13.82,
    f'{prefix} --reconstruction-algorithm linerec ': 14.00,
    f'{prefix} --binning 1': 5.47,
    f'{prefix} --nsino-per-chunk 2 --file-type double_fov': 7.67,
    f'{prefix} --center-search-width 30 --center-search-step 2': 12.70,
    f'{prefix} --dezinger 2': 13.98,
    f'{prefix} --flat-linear True': 13.98,
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
            self.assertAlmostEqual(ssum, cmd[1], places=0)


if __name__ == '__main__':
    unittest.main(testLoader=SequentialTestLoader(), failfast=True)
