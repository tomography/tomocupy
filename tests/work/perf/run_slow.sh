nz=(64 32 16 8 2 1)
size=(512 1024 2048 4096 8192 16384)
methods=(linerec)
dtypes=(float16 float32)
for im in 0; do
    for iz in {0..5}; do    
        for id in {0,1}; do
            for ir in {0..2}; do
                rm -rf /local/data_rec/*; 
                # rsync -a --delete /local/empty/ /local/data_rec/
                pkill -9 tomocupy
                sleep 2
                echo ${size[$iz]} ${nz[$iz]} ${methods[$im]} ${dtypes[$id]}
                tomocupy recon --file-name /local/data/tmp${size[$iz]}.h5 --reconstruction-type full --dtype ${dtypes[$id]} --reconstruction-algorithm ${methods[$im]} --max-write-threads 8 --nsino-per-chunk ${nz[$iz]}                
                cp time.npy times/time_${size[$iz]}_${nz[$iz]}_${methods[$im]}_${dtypes[$id]}_$ir.npy                
            done
        done
    done
done



# source ~/.bashrc; conda activate tomopy;
# for iz in {0..5}; do
#     for ir in {0..2}; do
#         rm -rf /local/data_rec/*; 
#         sleep 5
#         echo ${size[$iz]} ${nz[$iz]} 'tomopy' 'float32'
#         tomopy recon --file-name /local/data/tmp${size[$iz]}.h5 --reconstruction-type full --nsino-per-chunk ${nzt[$iz]} --rotation-axis-auto manual
#         cp time.npy times/time_${size[$iz]}_${nz[$iz]}_tomopy_float32_$ir.npy                            
#     done                
# done

# source ~/.bashrc; conda activate tomopy;
# for iz in {0..5}; do
#     for ir in {0..2}; do
#         rm -rf /local/data_rec/*; 
#         sleep 5
#         echo ${size[$iz]} ${nz[$iz]} 'tomopypad' 'float32'
#         tomopy recon --file-name /local/data/tmp${size[$iz]}.h5 --reconstruction-type full --nsino-per-chunk ${nzt[$iz]} --rotation-axis-auto manual --gridrec-padding True
#         cp time.npy times/time_${size[$iz]}_${nz[$iz]}_tomopypad_float32_$ir.npy                            
#     done                
# done




