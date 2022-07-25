import numpy as np
import matplotlib.pyplot as plt

nz=[64,32,8,4,2,1]
size=[512,1024,2048,4096,8192,16384]
methods=['fourierrec','lprec','linesummation']
dtypes=['float16','float32']
times = np.zeros([7,4,3])
for iz in range(0,6):
    for im in range(0,3):
        for id in range(0,2):
            try:
                for ir in range(0,4):
                    t = np.load(f'times/time_{size[iz]}_{nz[iz]}_{methods[im]}_{dtypes[id]}_{ir}.npy')*size[iz]#/size[iz]
                    times[iz,im,id] +=t
                    print(t,end=' ')
            except:
                pass
            print('\n')

times/=4
plt.figure(figsize=(12,3))
for im in range(0,3):
    for id in range(0,2):
        print(f'{methods[im]:<13},{dtypes[id]} ',end="")            
        for iz in range(0,7):
            print(f'& {times[iz,im,id]:.1e}',end="")            
        print('')
        plt.plot(times[:,im,id],label=f'{methods[im]},{dtypes[id]}')
plt.yscale('linear')
plt.legend()
plt.show()

