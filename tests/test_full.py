import unittest
import os
import numpy as np
import tifffile
import inspect
import h5py

prefix = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 782.5 --nsino-per-chunk 4'
cmd_dict = {
    f'{prefix} ': 27.6,
    f'{prefix} --reconstruction-algorithm lprec ': 27.6,
    f'{prefix} --reconstruction-algorithm linerec ': 27.6,
    f'{prefix} --dtype float16': 23.8,
    f'{prefix} --reconstruction-algorithm lprec --dtype float16': 23.8,
    f'{prefix} --reconstruction-algorithm linerec --dtype float16': 25.2,
    f'{prefix} --binning 1': 11.9,
    f'{prefix} --reconstruction-algorithm lprec --binning 1': 11.9,
    f'{prefix} --reconstruction-algorithm linerec --binning 1': 11.9,
    f'{prefix} --start-row 3 --end-row 15 --start-proj 200 --end-proj 700': 17.1,
    f'{prefix} --save-format h5': 27.6,
    f'{prefix} --nsino-per-chunk 2 --double_fov True': 27.6,
    f'{prefix} --nsino-per-chunk 2 --blocked-views True --blocked-views-start 0.2 --blocked-views-end 1': 30.0,
    f'{prefix} --remove-stripe-method fw': 27.5,
    f'{prefix} --remove-stripe-method fw --dtype float16': 23.6,
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
            try:
                with h5py.File('data_rec/test_data_rec.h5', 'r') as fid:
                    data = fid['exchange/data']
                    ssum = np.sum(np.linalg.norm(data[:], axis=(1, 2)))
            except:
                pass
            for k in range(24):
                try:
                    ssum += np.linalg.norm(tifffile.imread(
                        f'data_rec/test_data_rec/recon_{k:05}.tiff'))
                except:
                    pass
            self.assertAlmostEqual(ssum, cmd[1], places=0)


if __name__ == '__main__':
    unittest.main(testLoader=SequentialTestLoader(), failfast=True)
