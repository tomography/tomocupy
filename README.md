# h5lprec

## Installation necessary packages
conda create -n h5lprec -c conda-forge python=3.9 dxchange cupy scikit-build cupy pywavelets

conda activate h5lprec

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/fbcotter/pytorch_wavelets

cd pytorch_wavelets

pip install .

## Installation h5lprec

cd h5lprec

python setup.py install

## Run test
cd tests

python test_syn.py

## Usage with real data/

h5lprec <h5 file> <rotation_center> <chunk_size>

Example:

h5lprec /local/ssd/tmp/286_2_spfp_019.h5 1023 8

