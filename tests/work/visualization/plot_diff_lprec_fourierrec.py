import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import dxchange
import numpy as np
[x1,x2] = np.meshgrid(np.arange(-1024,1024)/1024,np.arange(-1024,1024)/1024)
circ = x1**2+x2**2<1-8/2048

lp32 = dxchange.read_tiff('res/lfp32.tiff').astype('float32')*0.886#/2048/2048#*0.94
fp32 = dxchange.read_tiff('res/fp32.tiff').astype('float32')

# lp32 = dxchange.read_tiff('res/llp32.tiff').astype('float32')/2048/2048/2
# fp32 = dxchange.read_tiff('res/lfp32.tiff').astype('float32')/2048/2048/2
# print(np.sum(fpl32[1024:1024+32,1024:1024+32]))
# print(np.sum(fp32[1024:1024+32,1024:1024+32]))
# exit()


# lp32 = lp32*np.linalg.norm(fp32)/np.linalg.norm(lp32)
fig = plt.figure(constrained_layout=True, figsize=(10.7, 2.8))
grid = fig.add_gridspec(1, 3, height_ratios=[1])

mmin = -0.0003
mmax = 0.0003

stx = 1024-50-150-200
endx = 1024+50-150-200
sty = 1024-50-150-200
endy = 1024+50-150-200
w = 6
diff = (fp32-lp32)
print(np.linalg.norm(diff))
exit()

fp320 = fp32[sty:endy,stx:endx].copy()
lp320 = lp32[sty:endy,stx:endx].copy()
diff0 = fp320-lp320
fp32[sty-w:sty+w+1,stx:endx] = mmin
fp32[endy-w:endy+w+1,stx:endx] = mmin
fp32[sty:endy,stx-w:stx+w+1] = mmin
fp32[sty:endy,endx-w:endx+w+1] = mmin
lp32[sty-w:sty+w+1,stx:endx] = mmin
lp32[endy-w:endy+w+1,stx:endx] = mmin
lp32[sty:endy,stx-w:stx+w+1] = mmin
lp32[sty:endy,endx-w:endx+w+1] = mmin
diff[sty-w:sty+w+1,stx:endx] = mmin
diff[endy-w:endy+w+1,stx:endx] = mmin
diff[sty:endy,stx-w:stx+w+1] = mmin
diff[sty:endy,endx-w:endx+w+1] = mmin

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(fp32*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')
ax0.set_title('Fourier, 32-bit')

ax = ax0.inset_axes([0.125,-0.02,0.6,0.6])
im = ax.imshow(fp320, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax.add_artist(scalebar)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax.set_xticks([])
ax.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()


ax0 = fig.add_subplot(grid[1])
ax0.set_title('lprec, 32-bit')
im = ax0.imshow(lp32*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')
ax = ax0.inset_axes([0.391,-0.02,0.6,0.6])
im = ax.imshow(lp320, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax.add_artist(scalebar)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax.set_xticks([])
ax.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()



mmin = -0.00003
mmax = 0.00003

ax0 = fig.add_subplot(grid[2])
ax0.set_title('Fourier, difference')
im = ax0.imshow(diff*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')

ax = ax0.inset_axes([0.649,-0.02,0.6,0.6])
im = ax.imshow(diff0, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax.add_artist(scalebar)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax.set_xticks([])
ax.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()





plt.colorbar(im, cax=cax,format='%.0e')
plt.savefig('difference_fourier_lprec.png',dpi=300,bbox_inches='tight')