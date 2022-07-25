nz=(64 32 8 4 2 1)
nzt=(512 256 256 128 64 32)
size=(512 1024 2048 4096 8192 16384)

for iz in {0..5}; do
    for ir in {0..3}; do
        rm -rf /local/data_rec/*; 
        sleep 5
        echo ${size[$iz]} ${nz[$iz]} 'linesummation' 'float32'
        tomocupy recon --file-name /local/data/tmp${size[$iz]}.h5 --reconstruction-type full --dtype float32 --reconstruction-algorithm linesummation --max-write-threads 8 --nsino-per-chunk ${nz[$iz]}
        cp time.npy times/time_${size[$iz]}_${nz[$iz]}_linesummation_float32_$ir.npy                
        
    done        
done

source ~/.bashrc; conda activate tomopy;
for iz in {0..5}; do
    for ir in {0..3}; do
        rm -rf /local/data_rec/*; 
        sleep 5
        echo ${size[$iz]} ${nz[$iz]} 'tomopy' 'float32'
        tomopy recon --file-name /local/data/tmp${size[$iz]}.h5 --reconstruction-type full --nsino-per-chunk ${nzt[$iz]} --rotation-axis-auto manual
        cp time.npy times/time_${size[$iz]}_${nz[$iz]}_tomopy_float32_$ir.npy                            
    done                
done

source ~/.bashrc; conda activate tomopy;
for iz in {0..5}; do
    for ir in {0..3}; do
        rm -rf /local/data_rec/*; 
        sleep 5
        echo ${size[$iz]} ${nz[$iz]} 'tomopypad' 'float32'
        tomopy recon --file-name /local/data/tmp${size[$iz]}.h5 --reconstruction-type full --nsino-per-chunk ${nzt[$iz]} --rotation-axis-auto manual --gridrec-padding True
        cp time.npy times/time_${size[$iz]}_${nz[$iz]}_tomopypad_float32_$ir.npy                            
    done                
done




