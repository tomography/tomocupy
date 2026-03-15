import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from tomocupy.ai.model_archs import ClassificationModel, _make_dinov2_model

def sample_patch_corner(mask,window_size,num_windows):
    sample_patch_probs = (mask / mask.sum()).reshape((-1,1)).squeeze().astype(np.float64)
    grid_indices = np.where(np.random.multinomial(1,sample_patch_probs/sample_patch_probs.sum(),num_windows))[1]
    patch_corners = []
    for grid_idx in grid_indices:
        grid_idx_ = []
        img_grids = np.indices(mask.shape)
        for d in range(len(list(mask.shape))):
            grid_idx_.append(img_grids[d].reshape((-1,1)).squeeze()[grid_idx])
        if grid_idx_[-1] == 0:
            grid_idx_ = grid_idx_[:-1]
        patch_corner = [grid_idx_[i]-window_size//2 for i in range(len(grid_idx_))]
        patch_corner = [max(0, pc) for pc in patch_corner]
        patch_corner = [min(pc, mask.shape[i] - window_size - 1) for i, pc in enumerate(patch_corner)]
        patch_corner = tuple(patch_corner)
        patch_corners.append(patch_corner)
    
    return patch_corners

def inference_pipeline(args, img_cache_original, center_of_rotation_cache, out_dir, preprocessed=False):
    
    use_8bits = args.infer_use_8bits
    downsample_factors = args.infer_downsample_factor
    nums_windows = args.infer_num_windows
    szs = args.infer_window_size
    assert isinstance(downsample_factors,list)
    assert isinstance(nums_windows,list)
    assert isinstance(szs,list)
    seed_number = args.infer_seed_number
    model_path = args.infer_model_path
    if len(nums_windows)>1:
        multi_instances = True
    elif len(nums_windows)==1 and nums_windows[0]>1:
        multi_instances = True
    else:
        multi_instances = False
    
    np.random.seed(seed_number)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    model_ = _make_dinov2_model()
    model = ClassificationModel(model_,embed_dim=model_.embed_dim,num_windows=nums_windows,multi_instances=multi_instances)
    states = torch.load(model_path, map_location='cpu')['state_dict']
    states = {(k.replace("module.", "") if "module." in k else k): v for k, v in states.items()}
    msg = model.load_state_dict(states,strict=False)
    model.to(device)

    print('starting model inference...')
    t_start3 = time.time()

    imgs_cache = []
    for downsample_factor in downsample_factors:
        if downsample_factor > 1:
            print(f"Resizing with downsample factor {downsample_factor}.")
        else:
            print(f"Downsample factor is {downsample_factor}. No resizing applied.")
        if use_8bits:
            print("Requantizing using 8 bits.")
        img_cache = []
        
        for img_ in img_cache_original:
            if not preprocessed:
                if downsample_factor>1:
                    
                    img_ = Image.fromarray(img_,mode='F')
                    img_array = np.array(img_.resize((img_.size[0]//downsample_factor,img_.size[1]//downsample_factor),Image.BILINEAR),dtype=np.float32)
                else:
                    
                    img_array = img_.copy().astype(np.float32)
            
                img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8))

                if use_8bits:
                    
                    img_array = (img_array * 255).astype(np.uint8)
                    img_array = img_array.astype(np.float32) / 255.
            else:
                img_array = img_.copy().astype(np.float32)
            img_cache.append(img_array[None,...])
        img_cache = np.concatenate(img_cache,axis=0)
        imgs_cache.append(img_cache)

    if multi_instances:
        patches_corners = []
        for img_cache,num_windows,sz in zip(imgs_cache,nums_windows,szs):
            row, col = img_cache.shape[1:]
            x_coords, y_coords = np.meshgrid(np.arange(col)-(col-1)/2, np.arange(row)-(row-1)/2)
            mask = (x_coords**2+y_coords**2) <= ((row-1) / 2)**2
            patch_corners = sample_patch_corner(mask,sz,num_windows)
            patches_corners.append(patch_corners)
    else:
        row, col = imgs_cache[0].shape[1:]
        sz = szs[0]
        patches_corners = [(row//2-sz//2, col//2-sz//2)]
        
    features = []
    
    
    for idx in range(imgs_cache[0].shape[0]):
        samples = []
        for img_cache,patch_corners,sz in zip(imgs_cache,patches_corners,szs):
            img_array = img_cache[idx]
            if multi_instances:
                imgs = []
                for patch_corner in patch_corners:
                    img = img_array[patch_corner[0]:patch_corner[0]+sz,patch_corner[1]:patch_corner[1]+sz]
                    img = torch.from_numpy(img).to(device=device,dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    imgs.append(img)
                sample = {'images':torch.cat(imgs,dim=1)}
            else:
                img = img_array[patch_corner[0]:patch_corner[0]+sz,patch_corner[1]:patch_corner[1]+sz]
                img = torch.from_numpy(img).to(device=device,dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                sample = {'images': img}
            samples.append(sample)
        with torch.no_grad():
            feature = model(samples)
        features.append(feature)
    t_stop3 = time.time()
    print(f"done. Elapsed time is {t_stop3-t_start3} s.")
    
    features_all = torch.cat(features,dim=0).detach().cpu().numpy()
    if args.infer_save_intermediate_data:
        np.savez(Path(out_dir)/'predicts_all',features_all,center_of_rotation_cache)
    scores = np.exp(features_all[:,1])/(np.exp(features_all[:,0])+np.exp(features_all[:,1]))
    centers_of_rotation = [center_of_rotation_cache[i] for i in np.where(scores==max(scores))[0]]
    with open(Path(out_dir)/'center_of_rotation.txt','a') as f:
        for cor in centers_of_rotation:
            f.write(f"{cor:.1f}\n")