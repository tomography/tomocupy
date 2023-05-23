import os
import sys
import numpy as np
import dxchange

# lamino_start = 350*4
# lamino_end = 700*4
# for l in range(lamino_start,lamino_end):
#     print(f'{l=}')
#     data = dxchange.read_tiff(f'/data/2022-10/Nikitin/brain_s3_7p5x_31MP_090_rec0/recon_0{l}.tiff').copy()
#     for k in range(1536,6000,1536):
#         st = k
#         end = min(k+1536,5999)
#         print(st,end)
#         data += dxchange.read_tiff(f'/data/2022-10/Nikitin/brain_s3_7p5x_31MP_090_rec{k}/recon_0{l}.tiff')
#     print(data.shape)
#     dxchange.write_tiff(data,f'/data/2022-10/Nikitin/brain_s3_7p5x_31MP_090_rec_sum/recon_0{l}.tiff',overwrite=True)

lamino_start = 1150
lamino_end = 2350
for l in range(lamino_start,lamino_end):
    print(f'{l=}')
    data = dxchange.read_tiff(f'/data/2022-10/Nikitin_rec/mosaic_rec0/recon_0{l}.tiff').copy()
    for k in range(1536,9000,1536):
        st = k
        end = min(k+1536,9000)
        print(st,end)
        data += dxchange.read_tiff(f'/data/2022-10/Nikitin_rec/mosaic_rec{k}/recon_0{l}.tiff')
    print(data.shape)
    dxchange.write_tiff(data,f'/data/2022-10/Nikitin_rec/mosaic_rec_sum/recon_0{l}.tiff',overwrite=True)
#lamino_start = 350*4
#lamino_end = 650*4
#for l in range(lamino_start,lamino_end):
#    print(f'{l=}')
#    data = dxchange.read_tiff(f'/data/2022-10/Nikitin_rec/brain_s1568_7p5x_31MP_091_rec0/recon_0{l}.tiff').copy()
#    for k in range(1536,6000,1536):
#        st = k
#        end = min(k+1536,5999)
#        print(st,end)
#        data += dxchange.read_tiff(f'/data/2022-10/Nikitin_rec/brain_s1568_7p5x_31MP_091_rec{k}/recon_0{l}.tiff')
#    print(data.shape)
#    dxchange.write_tiff(data,f'/data/2022-10/Nikitin_rec/brain_s1568_7p5x_31MP_091_rec_sum/recon_0{l}.tiff',overwrite=True)
    

    
        
# lamino_start = 350*4
# lamino_end = 650*4
# for k in range(0,6000,1536):
#     st = k
#     end = min(k+1536,5999)
#     print(st,end)
#     os.system(f'tomocupy recon_steps --lamino-angle 19 --file-name /data/2022-10/Nikitin/brain_s1568_7p5x_31MP_091.h5 --remove-stripe-method fw --rotation-axis 3137 --fw-sigma 2 \
#               --nsino-per-chunk 4 --nproj-per-chunk 4 --reconstruction-type full --start-proj {st} --end-proj {end} --lamino-start-row {lamino_start} --lamino-end-row {lamino_end} --out-path-name /data/2022-10/Nikitin/brain_s1568_7p5x_31MP_091_rec{k}/')
    
# lamino_start = 1100
# lamino_end = 2400
# for k in range(0,9001,1536):
#     st = k
#     end = min(k+1536,9000)
#     print(st,end)
#     os.system(f'tomocupy recon_steps --lamino-angle -19 --file-name /data/2022-10/Nikitin/brain_tile2/mosaic/mosaic.h5 --remove-stripe-method fw --rotation-axis 4549 --fw-sigma 2 \
#               --nsino-per-chunk 2 --nproj-per-chunk 4 --reconstruction-type full --start-proj {st} --end-proj {end} --lamino-start-row {lamino_start} --lamino-end-row {lamino_end} --out-path-name /data/2022-10/Nikitin/brain_tile2/mosaic/mosaic_rec{k}/')
#    
