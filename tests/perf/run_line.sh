
nzt=(512 256 128 64 32 16)
nz=(64 32 8 4 2 2)
size=(512 1024 2048 4096 8192 16384)
# nz=(2 2 2)
# size=(4096 8192 16384)

for k in {0,1,2,3,4,5}; do
echo ${nz[$k]}
echo ${size[$k]}

echo 'fp32 line'
rm -rf /local/data_rec/*; 
tomocupy recon --file-name /local/data/tmp${size[$k]}.h5 --reconstruction-type full --nsino-per-chunk ${nz[$k]} --reconstruction-algorithm linesummation
done
