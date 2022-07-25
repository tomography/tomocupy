import unittest
import os
import numpy as np
import tifffile
import inspect

prefix = 'tomocupy recon_steps --file-name data/test_data.h5 --rotation-axis 782.5 --nsino-per-chunk 4 --lamino-angle 1'
cmd_dict = {
    f'{prefix} --reconstruction-type try --reconstruction-algorithm linesummation': 13.7,
    f'{prefix} --reconstruction-type try_lamino --reconstruction-algorithm linesummation': 36.3,
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
                        f'data_rec/try_center/test_data/recon_{k:05.2f}.tiff'))
                except:
                    pass
            for k in np.arange(-4, 6, 0.25):
                try:
                    ssum += np.linalg.norm(tifffile.imread(
                        f'data_rec/try_center/test_data/recon_{k:05.2f}.tiff'))
                except:
                    pass
            self.assertAlmostEqual(ssum, cmd[1], places=1)            
                
#     def test_try_reconstep_linesummation(self):
#         os.system('rm -rf data_rec')
#         cmd = 'tomocupy recon_steps --reconstruction-algorithm linesummation --reconstruction-type try --file-name data/test_data.h5 --lamino-angle 1 --nsino-per-chunk 4 '
#         print(f'TEST {inspect.stack()[0][3]}: {cmd}')
#         st = os.system(cmd)
#         self.assertEqual(st, 0)
#         ssum = 0
#         for k in np.arange(758.5, 778.5, 0.5):
#             ssum += np.sum(tifffile.imread(
#                 f'data_rec/try_center/test_data/recon_{k:05.2f}.tiff').astype('float32'))
#         self.assertAlmostEqual(ssum,2027.0089836120605, places=1)

#     def test_try_lamino_reconstep_linesummation(self):
#         os.system('rm -rf data_rec')
#         cmd = 'tomocupy recon_steps --reconstruction-algorithm linesummation --reconstruction-type try_lamino --file-name data/test_data.h5 --lamino-angle 1 --nsino-per-chunk 4'
#         print(f'TEST {inspect.stack()[0][3]}: {cmd}')
#         st = os.system(cmd)
#         self.assertEqual(st, 0)
#         ssum = 0
#         for k in np.arange(-4, 6, 0.25):            
#             ssum += np.sum(tifffile.imread(
#                 f'data_rec/try_center/test_data/recon_{k:05.2f}.tiff'))
#         self.assertAlmostEqual(ssum,1133.0427651405334, places=1)

#     def test_full_reconstep_linesummation_double_fov(self):
#         os.system('rm -rf data_rec')
#         cmd = 'tomocupy recon_steps --reconstruction-algorithm linesummation --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 170 --file-type double_fov --lamino-angle 2 --nsino-per-chunk 2'
#         print(f'TEST {inspect.stack()[0][3]}: {cmd}')
#         st = os.system(cmd)
#         self.assertEqual(st, 0, f"{cmd} failed to run")
#         ssum = 0
#         for k in range(22):
#             ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff').astype('float32'))
#         self.assertAlmostEqual(ssum, 484.60846, places=1)
    
#     def test_full_reconstep_linesummation_h5_double_fov(self):
#         os.system('rm -rf data_rec')
#         cmd = 'tomocupy recon_steps --reconstruction-algorithm linesummation --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 170 --file-type double_fov --lamino-angle 2 --nsino-per-chunk 2 --save-format h5'
#         print(f'TEST {inspect.stack()[0][3]}: {cmd}')
#         st = os.system(cmd)
#         self.assertEqual(st, 0, f"{cmd} failed to run")
#         with h5py.File('data_rec/test_data_rec.h5', 'r') as fid:
#             data = fid['exchange/data']
#             ssum = np.sum(data)        
#         self.assertAlmostEqual(ssum, 484.60846, places=1)

#     def test_full_reconstep_linesummation_fp16_h5_double_fov(self):
#         os.system('rm -rf data_rec')
#         cmd = 'tomocupy recon_steps --reconstruction-algorithm linesummation --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 170 --file-type double_fov --lamino-angle 2 --nsino-per-chunk 2 --save-format h5 --dtype float16'
#         print(f'TEST {inspect.stack()[0][3]}: {cmd}')
#         st = os.system(cmd)
#         self.assertEqual(st, 0, f"{cmd} failed to run")
#         with h5py.File('data_rec/test_data_rec.h5', 'r') as fid:
#             data = fid['exchange/data']
#             ssum = np.sum(data.astype('float32'))        
#         self.assertAlmostEqual(ssum, 336.27917, places=-2)
    
#     def test_full_recon_write_threads(self):
#         os.system('rm -rf data_rec')
#         cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --nsino-per-chunk 4 --max-write-threads 4'
#         print(f'TEST {inspect.stack()[0][3]}: {cmd}')
#         st = os.system(cmd)
#         self.assertEqual(st, 0)
#         ssum = 0
#         for k in range(22):
#             ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff'))
#         self.assertAlmostEqual(ssum, 1449.0039, places=1)

#     def test_full_recon_write_threads(self):
#         os.system('rm -rf data_rec')
#         cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --nsino-per-chunk 4 --max-write-threads 1'
#         print(f'TEST {inspect.stack()[0][3]}: {cmd}')
#         st = os.system(cmd)
#         self.assertEqual(st, 0)
#         ssum = 0
#         for k in range(22):
#             ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff'))
#         self.assertAlmostEqual(ssum, 1449.0039, places=1)
    
#     def test_recon_step_write_threads(self):
#         os.system('rm -rf data_rec')
#         cmd = 'tomocupy recon_steps --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --nsino-per-chunk 4 --max-write-threads 1'
#         print(f'TEST {inspect.stack()[0][3]}: {cmd}')
#         st = os.system(cmd)
#         self.assertEqual(st, 0)
#         ssum = 0
#         for k in range(22):
#             ssum += np.sum(tifffile.imread(f'data_rec/test_data_rec/recon_{k:05}.tiff'))
#         self.assertAlmostEqual(ssum, 1449.0038, places=1)


if __name__ == '__main__':
    unittest.main(testLoader=SequentialTestLoader(), failfast=True)
