
nzt=(512 256 128 64 32 16)
nz=(64 32 8 4 2 1)
size=(512 1024 2048 4096 8192 16384)
# nz=(2 2 2)
# size=(4096 8192 16384)

for k in 5; do
echo ${nz[$k]}
echo ${size[$k]}

# rm -rf /local/data_rec/*; 
# sleep 5
# echo 'fp16 fourierrec'
# tomocupy recon --file-name /local/data/tmp${size[$k]}.h5 --reconstruction-type full --dtype float16 --reconstruction-algorithm fourierrec --max-write-threads 8 --nsino-per-chunk ${nz[$k]}
# pkill -9 tomocupy 

rm -rf /local/data_rec/*;
sleep 5
echo 'fp16 lprec'
tomocupy recon --file-name /local/data/tmp${size[$k]}.h5 --reconstruction-type full --dtype float16 --reconstruction-algorithm lprec --max-write-threads 8 --nsino-per-chunk ${nz[$k]}
pkill -9 tomocupy 

# rm -rf /local/data_rec/*;
# sleep 5
# echo 'fp32 fourierrec'
# tomocupy recon --file-name /local/data//tmp${size[$k]}.h5 --reconstruction-type full --dtype float32 --reconstruction-algorithm fourierrec  --max-write-threads 8 --nsino-per-chunk ${nz[$k]}
# pkill -9 tomocupy 

# rm -rf /local/data_rec/*;
# sleep 5
# echo 'fp32 lprec'
# tomocupy recon --file-name /local/data//tmp${size[$k]}.h5 --reconstruction-type full --dtype float32 --reconstruction-algorithm lprec --max-write-threads 8 --nsino-per-chunk ${nz[$k]}
# pkill -9 tomocupy 

# echo 'fp32 tomopy'
# rm -rf /local/data_rec/*; 
# tomopy recon --file-name /local/data/tmp${size[$k]}.h5 --reconstruction-type full --rotation-axis-auto manual --nsino-per-chunk ${nzt[$k]} --gridrec-padding True;
done
