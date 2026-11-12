import unittest
import shutil
import os
import inspect
from pathlib import Path

INFER_PATH = os.environ['INFER_PATH']
BIN_INFER_PATH = os.environ['BIN_INFER_PATH']

prefix = f'tomocupy recon --file-name /home/beams/TANGS/conda/tomocupy/tests/data/test_data.h5 --nsino-per-chunk 2 --fbp-filter shepp --rotation-axis-auto auto --rotation-axis-method ai --use-relative-bin-size'
prefix1 = f'--infer-model-path {INFER_PATH} --bin-infer-model-path {BIN_INFER_PATH} --save-test-results --bin-infer-save-intermediate-data --infer-save-intermediate-data --reconstruction-type try --ai-search-method full --bin-infer-bin-sizes 0.05,0.0125 --bin-infer-bin-counts 20,4'
prefix2 = f'--infer-model-path {INFER_PATH} --bin-infer-model-path {BIN_INFER_PATH} --save-test-results --bin-infer-save-intermediate-data --infer-save-intermediate-data --reconstruction-type full --ai-search-method full --bin-infer-bin-sizes 0.05,0.0125 --bin-infer-bin-counts 20,4 --save-format tiff'
prefix3 = f'--infer-model-path {INFER_PATH} --bin-infer-model-path {BIN_INFER_PATH} --save-test-results --bin-infer-save-intermediate-data --infer-save-intermediate-data --reconstruction-type try'
prefix4 = f'--infer-model-path {INFER_PATH} --bin-infer-model-path {BIN_INFER_PATH} --save-test-results --bin-infer-save-intermediate-data --infer-save-intermediate-data --reconstruction-type full'
prefix5 = f'--infer-model-path {INFER_PATH} --bin-infer-model-path {BIN_INFER_PATH} --bin-infer-save-intermediate-data --infer-save-intermediate-data --reconstruction-type try --ai-search-method full --bin-infer-bin-sizes 0.05,0.0125 --bin-infer-bin-counts 20,4'
prefix6 = f'--infer-model-path {INFER_PATH} --infer-save-intermediate-data --reconstruction-type try'
prefix7 = f'--infer-model-path {INFER_PATH} --reconstruction-type try'
prefix8 = f'--infer-model-path {INFER_PATH} --reconstruction-type try --binning 1'
prefix9 = f'--infer-model-path {INFER_PATH} --reconstruction-type try --binning 1 --infer-binning 1'
prefix10 = f'--infer-model-path {INFER_PATH} --bin-infer-model-path {BIN_INFER_PATH} --reconstruction-type try --binning 1 --infer-binning 1 --bin-infer-binning 1'
prefix11 = f'--infer-model-path {INFER_PATH} --bin-infer-model-path {BIN_INFER_PATH} --reconstruction-type try --binning 1 --bin-infer-binning 1'
prefix12 = f'--infer-model-path {INFER_PATH} --bin-infer-model-path {BIN_INFER_PATH} --reconstruction-type try --bin-infer-binning 1'
prefix13 = f'--infer-model-path {INFER_PATH} --bin-infer-model-path {BIN_INFER_PATH} --reconstruction-type try --binning 1 --infer-binning 1 --bin-infer-binning 1 --ai-search-method full'
prefix14 = f'--infer-model-path {INFER_PATH} --bin-infer-model-path {BIN_INFER_PATH} --reconstruction-type try --binning 1 --bin-infer-binning 1 --ai-search-method full'
prefix15 = f'--infer-model-path {INFER_PATH} --bin-infer-model-path {BIN_INFER_PATH} --reconstruction-type try --bin-infer-binning 1 --ai-search-method full'
cmd_dict = {
    # f'{prefix} {prefix1}': 4,
    # f'{prefix} {prefix2}': 5,
    f'{prefix} {prefix3}': 3,
    f'{prefix} {prefix4}': 3,
    # f'{prefix} {prefix5}': 3,
    # f'{prefix} {prefix6}': 2,
    # f'{prefix} {prefix7}': 1,
    # f'{prefix} {prefix8}': 1,
    # f'{prefix} {prefix9}': 1,
    # f'{prefix} {prefix10}': 1,
    # f'{prefix} {prefix11}': 1,
    # f'{prefix} {prefix12}': 1,
    # f'{prefix} {prefix13}': 1,
    # f'{prefix} {prefix14}': 1,
    # f'{prefix} {prefix15}': 1,
}

extensions = {".npz", ".txt", ".json"}

class SequentialTestLoader(unittest.TestLoader):
    def getTestCaseNames(self, testCaseClass):
        test_names = super().getTestCaseNames(testCaseClass)
        testcase_methods = list(testCaseClass.__dict__.keys())
        test_names.sort(key=testcase_methods.index)
        return test_names

class Tests(unittest.TestCase):
    def test_full_recon(self):
        for cmd in cmd_dict.items():

            shutil.rmtree('data_rec',ignore_errors=True)      
            print(f'TEST {inspect.stack()[0][3]}: {cmd[0]}')
            st = os.system(cmd[0])
            self.assertEqual(st, 0)
            
            num = sum([len(list(Path('data_rec/try_center/test_data').glob(f'*{e}'))) for e in extensions])
            num += sum([len(list(Path('data_rec').glob(f"*{e}"))) for e in extensions])
            if Path('data_rec/test_data_rec').exists():
                num += sum([len(list(Path('data_rec/test_data_rec').glob(f"*{e}"))) for e in extensions])
            self.assertEqual(num,cmd[1])

if __name__ == '__main__':
    unittest.main(testLoader=SequentialTestLoader(), failfast=True)