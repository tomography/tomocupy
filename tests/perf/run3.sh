read_threads=(1 2 4 8)
write_threads=(4 8 16)
# python form_data.py 16384;
for j in {0,1,2}; do
for k in {0,1,2,3}; do
rm -rf /local/data_rec/*16384*; 
sleep 5
echo ${read_threads[$k]}
echo ${write_threads[$j]}
tomocupy recon --file-name /local/data/tmp16384.h5 --reconstruction-type full --dtype float16 --reconstruction-algorithm lprec --max-read-threads ${write_threads[$j]} --max-write-threads ${read_threads[$k]} --nsino-per-chunk 1
done
done
