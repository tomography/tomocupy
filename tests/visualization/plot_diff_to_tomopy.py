import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import dxchange
import numpy as np
[x1,x2] = np.meshgrid(np.arange(-1024,1024)/1024,np.arange(-1024,1024)/1024)
circ = x1**2+x2**2<1-8/2048
fig = plt.figure(constrained_layout=True, figsize=(7.2, 5.8))
grid = fig.add_gridspec(2, 2, height_ratios=[1,1])
fp32f = dxchange.read_tiff('res/fp32.tiff').astype('float32')
fp32l = dxchange.read_tiff('res/lfp32.tiff').astype('float32')/2048/2048
fp32t = dxchange.read_tiff('res/tfp32.tiff').astype('float32')/4
fp32line = dxchange.read_tiff('res/linefp32.tiff').astype('float32')
# fp32l = np.roll(fp32l,1,axis=0)
# fp32line = np.roll(fp32line,1,axis=0)



mmin = -0.0003
mmax = 0.0003

stx = 1024-50-150-200
endx = 1024+50-150-200
sty = 1024-50-150-200
endy = 1024+50-150-200
w = 6
diff = (fp32f-fp32l)
diff1 = (fp32f-fp32t)
diff2 = (fp32f-fp32line)

fp32f0 = fp32f[sty:endy,stx:endx].copy()
fp32t0 = fp32t[sty:endy,stx:endx].copy()
fp32l0 = fp32l[sty:endy,stx:endx].copy()
fp32line0 = fp32line[sty:endy,stx:endx].copy()
diff0 = fp32f0-fp32l0
diff10 = fp32f0-fp32t0
diff20 = fp32f0-fp32line0

fp32f[sty-w:sty+w+1,stx:endx] = mmin
fp32f[endy-w:endy+w+1,stx:endx] = mmin
fp32f[sty:endy,stx-w:stx+w+1] = mmin
fp32f[sty:endy,endx-w:endx+w+1] = mmin
fp32l[sty-w:sty+w+1,stx:endx] = mmin
fp32l[endy-w:endy+w+1,stx:endx] = mmin
fp32l[sty:endy,stx-w:stx+w+1] = mmin
fp32l[sty:endy,endx-w:endx+w+1] = mmin
fp32line[sty-w:sty+w+1,stx:endx] = mmin
fp32line[endy-w:endy+w+1,stx:endx] = mmin
fp32line[sty:endy,stx-w:stx+w+1] = mmin
fp32line[sty:endy,endx-w:endx+w+1] = mmin

fp32t[sty-w:sty+w+1,stx:endx] = mmin
fp32t[endy-w:endy+w+1,stx:endx] = mmin
fp32t[sty:endy,stx-w:stx+w+1] = mmin
fp32t[sty:endy,endx-w:endx+w+1] = mmin

diff[sty-w:sty+w+1,stx:endx] = mmin
diff[endy-w:endy+w+1,stx:endx] = mmin
diff[sty:endy,stx-w:stx+w+1] = mmin
diff[sty:endy,endx-w:endx+w+1] = mmin

diff1[sty-w:sty+w+1,stx:endx] = mmin
diff1[endy-w:endy+w+1,stx:endx] = mmin
diff1[sty:endy,stx-w:stx+w+1] = mmin
diff1[sty:endy,endx-w:endx+w+1] = mmin
diff2[sty-w:sty+w+1,stx:endx] = mmin
diff2[endy-w:endy+w+1,stx:endx] = mmin
diff2[sty:endy,stx-w:stx+w+1] = mmin
diff2[sty:endy,endx-w:endx+w+1] = mmin

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(fp32t*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')
ax0.set_title('Tomopy Gridrec')

ax = ax0.inset_axes([0.255,0.05,0.6,0.6])
im = ax.imshow(fp32t0, cmap='gray',vmin=mmin, vmax=mmax)
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

ax0 = fig.add_subplot(grid[1])
ax0.set_title('Difference between Tomopy Gridrec \n and Tomocupy Fourier ', fontsize=9.5)
im = ax0.imshow(diff1*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')
ax = ax0.inset_axes([0.473,0.05,0.6,0.6])
im = ax.imshow(diff10, cmap='gray',vmin=mmin, vmax=mmax)
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
ax0.set_title('Difference between Tomocupy Log-polar \n and Tomocupy Fourier ', fontsize=9.5)
im = ax0.imshow(diff*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')

ax = ax0.inset_axes([0.255,-0.18,0.6,0.6])
im = ax.imshow(diff0, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax.add_artist(scalebar)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax.set_xticks([])
ax.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()


ax0 = fig.add_subplot(grid[3])
ax0.set_title('Difference between Tomocupy Line summation \n and Tomocupy Fourier ', fontsize=9.5)
im = ax0.imshow(diff2*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')

ax = ax0.inset_axes([0.473,-0.18,0.6,0.6])
im = ax.imshow(diff20, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax.add_artist(scalebar)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax.set_xticks([])
ax.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()


plt.colorbar(im, cax=cax,format='%.0e')
plt.savefig('difference_to_tomopy.png',dpi=300,bbox_inches='tight')