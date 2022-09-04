nz=(64 32 16 8 2 1)
size=(512 1024 2048 4096 8192 16384)
methods=(lprec fourierrec linerec)
dtypes=(float16 float32)
for im in 0; do
    for iz in 2; do    
        for id in 0; do
            for ir in {0..2}; do
                # rm -rf /local/data_rec/*; 
                rsync -a --delete /local/empty/ /local/data_rec/
                pkill -9 tomocupy
                sleep 5
                echo ${size[$iz]} ${nz[$iz]} ${methods[$im]} ${dtypes[$id]}
                tomocupy recon --file-name /local/data/tmp${size[$iz]}.h5 --reconstruction-type full --dtype ${dtypes[$id]} --reconstruction-algorithm ${methods[$im]} --max-write-threads 8 --nsino-per-chunk ${nz[$iz]}                
                cp time.npy times/time_${size[$iz]}_${nz[$iz]}_${methods[$im]}_${dtypes[$id]}_$ir.npy                
            done
        done
    done
done

