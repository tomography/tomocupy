import unittest
import os
import numpy as np
import tifffile
import inspect
import h5py


class SequentialTestLoader(unittest.TestLoader):
    def getTestCaseNames(self, testCaseClass):
        test_names = super().getTestCaseNames(testCaseClass)
        testcase_methods = list(testCaseClass.__dict__.keys())
        test_names.sort(key=testcase_methods.index)
        return test_names


class Tests(unittest.TestCase):

    def test_imports(self):
        cmd = 'tomocupy recon -h'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0, 'Issues with import tomocupy')

    def test_try_recon(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --nsino-per-chunk 2'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in np.arange(758.5, 778.5, 0.5):
            ssum += np.sum(tifffile.imread(
                f'data_rec/try_center/test_data/recon_{k:05.2f}.tiff'))
        self.assertAlmostEqual(ssum,2766.41386, places=1)

    def test_try_recon_binning(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --binning 1 --nsino-per-chunk 2'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in np.arange(759, 779):
            ssum += np.sum(tifffile.imread(
                f'data_rec/try_center/test_data/recon_{k:05.2f}.tiff'))
        self.assertAlmostEqual(ssum, 653.3486022949219, places=1)

    def test_try_recon_double_fov(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --file-type double_fov --rotation-axis 130 --nsino-per-chunk 2'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in np.arange(120.5, 140.5, 0.5):
            ssum += np.sum(tifffile.imread(
                f'data_rec/try_center/test_data/recon_{k:05.2f}.tiff'))
        self.assertAlmostEqual(ssum, 4183.489646911621, places=1)

    def test_try_recon_center_steps(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --center-search-width 30 --center-search-step 2 --nsino-per-chunk 4'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in np.arange(740, 799, 2):
            ssum += np.sum(tifffile.imread(
                f'data_rec/try_center/test_data/recon_{k:05.2f}.tiff'))
        self.assertAlmostEqual(ssum, 2039.1368, places=1)

    
if __name__ == '__main__':
    unittest.main(testLoader=SequentialTestLoader(), failfast=True)
