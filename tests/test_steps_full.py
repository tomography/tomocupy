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

   
    def test_recon_step(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon_steps --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --nsino-per-chunk 4'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in range(22):
            ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff'))
        self.assertAlmostEqual(ssum, 1449.0038, places=1)

    def test_recon_step_retrieve_phase(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon_steps --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --retrieve-phase-method paganin --retrieve-phase-alpha 0.0001 --nsino-per-chunk 4'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in range(22):
            ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff'))
        self.assertAlmostEqual(ssum, 1445.8616, places=1)

    def test_recon_step_retrieve_phase_f16(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon_steps --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --retrieve-phase-method paganin --retrieve-phase-alpha 0.0001 --dtype float16 --nsino-per-chunk 4'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in range(22):
            ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff').astype('float32'))
        self.assertAlmostEqual(ssum, 1099.7876, places=-2)


if __name__ == '__main__':
    unittest.main(testLoader=SequentialTestLoader(), failfast=True)
