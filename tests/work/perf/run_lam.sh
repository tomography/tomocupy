# size=(6144)
# nz=(2)
# methods=(fourierrec linerec)
# for im in 0; do
    
#     for iz in {0..1}; do    
#         for k in  {0..1}; do
#                 rsync -a --delete /local/empty/ /local/tmp_rec/
#                 pkill -9 tomocupy
#                 sleep 3
#                 echo ${size[$iz]} ${nz[$iz]} ${methods[$im]}
#                 tomocupy recon_steps --file-name /local/tmp/tmp${size[$iz]}.h5 --reconstruction-type full --reconstruction-algorithm ${methods[$im]}  --nsino-per-chunk ${nz[$iz]} --nproj-per-chunk  ${nz[$iz]} --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-end-row 128
#             done
#         done    
#     done


# rsync -a --delete /local/empty/ /local/tmp*
# mkdir /local/tmp
# python form_data.py 1024 1024
# tomocupy recon_steps --file-name /local/tmp/tmp1024.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 64 --nproj-per-chunk  64 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 337 --lamino-end-row 721
# tomocupy recon_steps --file-name /local/tmp/tmp1024.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 64 --nproj-per-chunk  64 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 337 --lamino-end-row 721

# rsync -a --delete /local/empty/ /local/tmp*
# mkdir /local/tmp
# python form_data.py 1536 1536
# tomocupy recon_steps --file-name /local/tmp/tmp1536.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 32 --nproj-per-chunk  32 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 505 --lamino-end-row 1017
# tomocupy recon_steps --file-name /local/tmp/tmp1536.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 32 --nproj-per-chunk  32 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 505 --lamino-end-row 1017

# rsync -a --delete /local/empty/ /local/tmp*
# mkdir /local/tmp
# python form_data.py 2048 2048
# tomocupy recon_steps --file-name /local/tmp/tmp2048.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 16 --nproj-per-chunk  16 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 674 --lamino-end-row 1442
# tomocupy recon_steps --file-name /local/tmp/tmp2048.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 16 --nproj-per-chunk  16 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 674 --lamino-end-row 1442

# rsync -a --delete /local/empty/ /local/tmp*
# mkdir /local/tmp
# python form_data.py 3072 3072
# tomocupy recon_steps --file-name /local/tmp/tmp3072.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 8 --nproj-per-chunk  8 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 1011 --lamino-end-row 2035
# tomocupy recon_steps --file-name /local/tmp/tmp3072.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 8 --nproj-per-chunk  8 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 1011 --lamino-end-row 2035

# rsync -a --delete /local/empty/ /local/tmp*
# mkdir /local/tmp
# python form_data.py 4096 4096
# tomocupy recon_steps --file-name /local/tmp/tmp4096.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 4 --nproj-per-chunk  4 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 1348 --lamino-end-row 2884
# tomocupy recon_steps --file-name /local/tmp/tmp4096.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 4 --nproj-per-chunk  4 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 1348 --lamino-end-row 2884

# rsync -a --delete /local/empty/ /local/tmp*
# mkdir /local/tmp
# python form_data.py 6144 2048
# tomocupy recon_steps --file-name /local/tmp/tmp6144.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 2 --nproj-per-chunk  2 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 2000 --lamino-end-row 4048
# tomocupy recon_steps --file-name /local/tmp/tmp6144.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 2 --nproj-per-chunk  2 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 2000 --lamino-end-row 4048

# rsync -a --delete /local/empty/ /local/tmp*
# mkdir /local/tmp
# python form_data.py 6144 3072
# tomocupy recon_steps --file-name /local/tmp/tmp6144.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 2 --nproj-per-chunk  2 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 2000 --lamino-end-row 4048
# tomocupy recon_steps --file-name /local/tmp/tmp6144.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 2 --nproj-per-chunk  2 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 2000 --lamino-end-row 4048

rsync -a --delete /local/empty/ /local/tmp_rec
rsync -a --delete /local/empty/ /local/tmp
mkdir /local/tmp
python form_data.py 8192 1024
tomocupy recon_steps --file-name /local/tmp/tmp8192.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 2 --nproj-per-chunk  2 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 2000 --lamino-end-row 2128 --start-row 128 --end-row 256
tomocupy recon_steps --file-name /local/tmp/tmp8192.h5 --reconstruction-type full --reconstruction-algorithm fourierrec  --nsino-per-chunk 2 --nproj-per-chunk  2 --lamino-angle 20 --retrieve-phase-method paganin --energy 20 --remove-stripe-method ti --pixel-size 1 --lamino-start-row 2000 --lamino-end-row 2128







