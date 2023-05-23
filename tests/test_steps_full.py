import unittest
import os
import numpy as np
import tifffile
import inspect
import h5py

prefix = 'tomocupy recon_steps --file-name data/test_data.h5 --rotation-axis 782.5 --nsino-per-chunk 4 --reconstruction-type full'
cmd_dict = {
    f'{prefix}': 28.307,
    f'{prefix} --retrieve-phase-method paganin --retrieve-phase-alpha 0.0001 --propagation-distance 60 --energy 20 --pixel-size 1.17': 21.437,
    f'{prefix} --retrieve-phase-method paganin --retrieve-phase-alpha 0.0001 --propagation-distance 60 --energy 20 --pixel-size 1.17 --dtype float16 ':17.577,
    f'{prefix} --reconstruction-algorithm linerec': 28.341,
    f'{prefix} --lamino-angle 1 --reconstruction-algorithm linerec': 24.804,
    f'{prefix} --lamino-angle 1 --reconstruction-algorithm linerec --retrieve-phase-method paganin --retrieve-phase-alpha 0.0001 --propagation-distance 60 --energy 20 --pixel-size 1.17': 18.610,        
    f'{prefix} --lamino-angle 1 --save-format h5': 24.263,
    f'{prefix} --rotate-proj-angle 1.5': 27.903,
    f'{prefix} --rotate-proj-angle 1.5 --rotate-proj-order 3': 28.216,
    f'{prefix} --start-column 200 --end-column 1000': 18.248
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
