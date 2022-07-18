threads=(2 4)
size=(8192)

for k in 0; do
echo $k
echo ${threads[$k]}
echo ${size[$k]}
for j in {0,1}; do
echo $j
# python form_data.py $k;
rm -rf /local/data_rec/*${size[$k]}*; 
sleep 5
tomocupy recon --file-name /data/tmp/tmp${size[$k]}.h5 --reconstruction-type full --dtype float16 --reconstruction-algorithm fourierrec --max-write-threads 8 --nsino-per-chunk ${threads[$j]}
rm -rf /local/data_rec/*${size[$k]}*; 
sleep 5
tomocupy recon --file-name /data/tmp/tmp${size[$k]}.h5 --reconstruction-type full --dtype float16 --reconstruction-algorithm lprec --max-write-threads 8 --nsino-per-chunk ${threads[$j]}
rm -rf /local/data_rec/*${size[$k]}*; 
sleep 5
tomocupy recon --file-name /data/tmp//tmp${size[$k]}.h5 --reconstruction-type full --dtype float32 --reconstruction-algorithm fourierrec --max-write-threads 8 --nsino-per-chunk ${threads[$j]}
rm -rf /local/data_rec/*${size[$k]}*; 
sleep 5
tomocupy recon --file-name /data/tmp//tmp${size[$k]}.h5 --reconstruction-type full --dtype float32 --reconstruction-algorithm lprec --max-write-threads 8 --nsino-per-chunk ${threads[$j]}
done
done
