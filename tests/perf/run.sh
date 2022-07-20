
nzt=(512 256 128 64 32 16)
nz=(64 32 8 4 2 2)
size=(512 1024 2048 4096 8192 16384)
# nz=(2 2 2)
# size=(4096 8192 16384)

for k in {0,1,2}; do
echo ${nzt[$k]}
echo ${size[$k]}

echo 'fp32 tomopy'
rm -rf /local/data_rec/*; 
time tomopy recon --file-name /local/data/tmp${size[$k]}.h5 --reconstruction-type full --rotation-axis-auto manual --nsino-per-chunk ${nzt[$k]} --gridrec-padding True;
done
