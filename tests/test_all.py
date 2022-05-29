import unittest
import os
import filecmp
import numpy as np
import dxchange
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
        cmd = 'tomocupy recon --file-name data/test_data.h5'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in np.arange(758.5,778.5,0.5):
            ssum+=np.sum(dxchange.read_tiff(f'data_rec/try_center/test_data/recon_{k:05.2f}.tiff'))
        self.assertAlmostEqual(ssum, 2766.41386,places=2)

    def test_try_recon_double_fov(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --file-type double_fov --rotation-axis 130'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in np.arange(120.5,140.5,0.5):
            ssum+=np.sum(dxchange.read_tiff(f'data_rec/try_center/test_data/recon_{k:05.2f}.tiff'))
        self.assertAlmostEqual(ssum, 4183.489646911621,places=2)
    
    def test_try_recon_center_steps(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --center-search-width 30 --center-search-step 2'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum = 0
        for k in np.arange(740,799,2):
            ssum+=np.sum(dxchange.read_tiff(f'data_rec/try_center/test_data/recon_{k:05.2f}.tiff'))
        self.assertAlmostEqual(ssum, 2039.1368,places=2)
    
    def test_full_recon(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum=np.sum(dxchange.read_tiff_stack(f'data_rec/test_data_rec/recon_00000.tiff', ind = range(0,22)))
        self.assertAlmostEqual(ssum, 1449.0039,places=2)
    
    def test_full_recon_parts(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --start-row 3 --end-row 15 --start-proj 200 --end-proj 700'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0, f"{cmd} failed to run")
        ssum=np.sum(dxchange.read_tiff_stack(f'data_rec/test_data_rec/recon_00003.tiff', ind = range(3,15)))
        self.assertAlmostEqual(ssum, 756.3026,places=2)
    
    def test_full_recon_h5(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --save-format h5'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0, f"{cmd} failed to run")
        with h5py.File('data_rec/test_data_rec.h5','r') as fid:            
            data = fid['exchange/recon']            
            ssum = np.sum(data)
        self.assertEqual(data.shape, (22,1536,1536))
        self.assertAlmostEqual(ssum, 1449.0039,places=2)
    
    def test_full_recon_double_fov(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 170 --file-type double_fov'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0, f"{cmd} failed to run")
        data = dxchange.read_tiff_stack(f'data_rec/test_data_rec/recon_00000.tiff', ind = range(0,22))
        ssum=np.sum(data)
        self.assertEqual(data.shape, (22,1536*2,1536*2))
        self.assertAlmostEqual(ssum, 2425.952,places=2)

    def test_full_recon_double_fov_h5(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 170 --save-format h5 --file-type double_fov'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0, f"{cmd} failed to run")
        with h5py.File('data_rec/test_data_rec.h5','r') as fid:            
            data = fid['exchange/recon']            
            ssum = np.sum(data)
        self.assertEqual(data.shape, (22,1536*2,1536*2))
        self.assertAlmostEqual(ssum, 2425.952,places=2)

    def test_full_recon_f16(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --dtype float16'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum=np.sum(dxchange.read_tiff_stack(f'data_rec/test_data_rec/recon_00000.tiff', ind = range(0,22)).astype('float32'))
        self.assertAlmostEqual(ssum, 1152.5211,places=-1)

    def test_full_recon_f16_h5(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770  --save-format h5 --dtype float16'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0, f"{cmd} failed to run")
        with h5py.File('data_rec/test_data_rec.h5','r') as fid:            
            data = fid['exchange/recon']            
            ssum = np.sum(data.astype('float32'))
        self.assertEqual(data.shape, (22,1024,1024))
        self.assertAlmostEqual(ssum, 1152.5211,places=-1)
    
    def test_full_recon_blocked(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --blocked-views True --blocked-views-start 0.2 --blocked-views-end 1'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum=np.sum(dxchange.read_tiff_stack(f'data_rec/test_data_rec/recon_00000.tiff', ind = range(0,22)))
        self.assertAlmostEqual(ssum, 1199.6244, places=2)
    
    def test_full_recon_remove_stripe_fw(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --remove-stripe-method fw'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum=np.sum(dxchange.read_tiff_stack(f'data_rec/test_data_rec/recon_00000.tiff', ind = range(0,22)))
        self.assertAlmostEqual(ssum, 1459.2646, places=0)

    def test_full_recon_remove_stripe_fw_f16(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy recon --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --remove-stripe-method fw --dtype float16'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum=np.sum(dxchange.read_tiff_stack(f'data_rec/test_data_rec/recon_00000.tiff', ind = range(0,22)).astype('float32'))
        self.assertAlmostEqual(ssum, 1080.2837, places=-2)
            
    def test_recon_step(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy reconstep --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum=np.sum(dxchange.read_tiff_stack(f'data_rec/test_data_rec/recon_00000.tiff', ind = range(0,22)))
        self.assertAlmostEqual(ssum, 1449.0038, places=2)

    def test_recon_step_retrieve_phase(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy reconstep --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --retrieve-phase-method paganin --retrieve-phase-alpha 0.0001'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum=np.sum(dxchange.read_tiff_stack(f'data_rec/test_data_rec/recon_00000.tiff', ind = range(0,22)))
        self.assertAlmostEqual(ssum, 1445.8616, places=2)

    def test_recon_step_retrieve_phase_f16(self):
        os.system('rm -rf data_rec')
        cmd = 'tomocupy reconstep --file-name data/test_data.h5 --reconstruction-type full --rotation-axis 770 --retrieve-phase-method paganin --retrieve-phase-alpha 0.0001 --dtype float16'
        print(f'TEST {inspect.stack()[0][3]}: {cmd}')
        st = os.system(cmd)
        self.assertEqual(st, 0)
        ssum=np.sum(dxchange.read_tiff_stack(f'data_rec/test_data_rec/recon_00000.tiff', ind = range(0,22)).astype('float32'))
        self.assertAlmostEqual(ssum, 1099.7876, places=-2)
    
if __name__ == '__main__':
    unittest.main(testLoader=SequentialTestLoader(),failfast=False)
    