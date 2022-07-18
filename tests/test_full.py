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

    def test_full_recon(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --nsino-per-chunk 4'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in range(22):
            ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff'))
        self.assertAlmostEqual(ssum, 1449.0039, places=1)

    def test_full_recon_binning(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --binning 1 --nsino-per-chunk 2'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in range(11):
            ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff'))
        self.assertAlmostEqual(ssum, 362.4355, places=1)

    def test_full_recon_parts(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --start-row 3 --end-row 15 --start-proj 200 --end-proj 700 --nsino-per-chunk 2'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0, f"{cmd} failed to run")
        ssum = 0
        for k in range(3,15):
            ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff'))
        self.assertAlmostEqual(ssum, 756.3026, places=1)

    def test_full_recon_h5(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --save-format h5 --nsino-per-chunk 4'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0, f"{cmd} failed to run")
        with h5py.File('data_rec/test_data_rec.h5', 'r') as fid:
            data = fid['exchange/data']
            ssum = np.sum(data[:])
        self.assertAlmostEqual(ssum, 1449.0039, places=1)

    def test_full_recon_binning_h5(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --save-format h5 --binning 1 --nsino-per-chunk 4'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0, f"{cmd} failed to run")
        with h5py.File('data_rec/test_data_rec.h5', 'r') as fid:
            data = fid['exchange/data']
            ssum = np.sum(data[:])
            self.assertAlmostEqual(ssum, 362.4355, places=1)

    def test_full_recon_double_fov(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 170 --file-type double_fov --nsino-per-chunk 2'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0, f"{cmd} failed to run")
        ssum = 0
        for k in range(22):
            ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff'))
        self.assertAlmostEqual(ssum, 2425.952, places=1)

    def test_full_recon_double_fov_h5(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 170 --save-format h5 --file-type double_fov --nsino-per-chunk 2'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0, f"{cmd} failed to run")
        with h5py.File('data_rec/test_data_rec.h5', 'r') as fid:
            data = fid['exchange/data']
            ssum = np.sum(data)
        self.assertAlmostEqual(ssum, 2425.952, places=1)

    def test_full_recon_f16(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --dtype float16 --nsino-per-chunk 4'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in range(22):
            ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff'))
        self.assertAlmostEqual(ssum, 1152.5211, places=-1)

    def test_full_recon_f16_h5(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770  --save-format h5 --dtype float16 --nsino-per-chunk 4'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0, f"{cmd} failed to run")
        with h5py.File('data_rec/test_data_rec.h5', 'r') as fid:
            data = fid['exchange/data']
            ssum = np.sum(data.astype('float32'))
        self.assertEqual(data.shape, (22, 1024, 1024))
        self.assertAlmostEqual(ssum, 1152.5211, places=-1)

    def test_full_recon_blocked(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --blocked-views True --blocked-views-start 0.2 --blocked-views-end 1 --nsino-per-chunk 4'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in range(22):
            ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff'))
        self.assertAlmostEqual(ssum, 1199.6244, places=1)

    def test_full_recon_remove_stripe_fw(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --remove-stripe-method fw --nsino-per-chunk 4'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in range(22):
            ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff'))
        self.assertAlmostEqual(ssum, 1461.764, places=0)

    def test_full_recon_remove_stripe_fw_f16(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --remove-stripe-method fw --dtype float16 --nsino-per-chunk 4'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in range(22):
            ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff'))
        self.assertAlmostEqual(ssum, 1085.961, places=-2)

    
    def test_full_recon_crop(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --crop 512 --nsino-per-chunk 4'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in range(22):
            ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff').astype('float32'))
        self.assertAlmostEqual(ssum, 583.6758, places=1)


if __name__ == '__main__':
    unittest.main(testLoader=SequentialTestLoader(), failfast=True)
