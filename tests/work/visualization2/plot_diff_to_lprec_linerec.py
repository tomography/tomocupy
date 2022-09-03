import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import dxchange
import numpy as np
from skimage.metrics import structural_similarity as ssim
[x1,x2] = np.meshgrid(np.arange(-1024,1024)/1024,np.arange(-1024,1024)/1024)
circ = x1**2+x2**2<1-8/2048
fig = plt.figure(constrained_layout=True, figsize=(7.2, 5.8))
grid = fig.add_gridspec(2, 2, height_ratios=[1,1])
fp32f = dxchange.read_tiff('res/fp32.tiff').astype('float32')
fp32t = dxchange.read_tiff('res/lfp32.tiff').astype('float32')
fp32tpad = dxchange.read_tiff('res/linefp32.tiff').astype('float32')
fp32tine = dxchange.read_tiff('res/linefp32.tiff').astype('float32')
# fp32t = np.roll(fp32t,1,axis=0)
# fp32tine = np.roll(fp32tine,1,axis=0)

s1 = ssim(fp32f*1000,fp32t*1000)
s2 = ssim(fp32f*1000,fp32tpad*1000)

mmin = -0.001
mmax = 0.001

stx = 1024-75-150-200
endx = 1024+75-150-200
sty = 1024-75-150-200
endy = 1024+75-150-200
w = 6
diff = (fp32f-fp32t)
diff1 = (fp32f-fp32tpad)
diff2 = (fp32f-fp32tine)

fp32f0 = fp32f[sty:endy,stx:endx].copy()
fp32tpad0 = fp32tpad[sty:endy,stx:endx].copy()
fp32t0 = fp32t[sty:endy,stx:endx].copy()
fp32tine0 = fp32tine[sty:endy,stx:endx].copy()
#print(np.linalg.norm(fp32f0)/np.linalg.norm(fp32tpad0))
# print(ssim(fp32f0*1000,fp32t0*1000))
# print(ssim(fp32f0*1000,fp32tpad0*1000))

diff0 = fp32f0-fp32t0
diff10 = fp32f0-fp32tpad0
diff20 = fp32f0-fp32tine0

fp32f[sty-w:sty+w+1,stx:endx] = mmin
fp32f[endy-w:endy+w+1,stx:endx] = mmin
fp32f[sty:endy,stx-w:stx+w+1] = mmin
fp32f[sty:endy,endx-w:endx+w+1] = mmin
fp32t[sty-w:sty+w+1,stx:endx] = mmin
fp32t[endy-w:endy+w+1,stx:endx] = mmin
fp32t[sty:endy,stx-w:stx+w+1] = mmin
fp32t[sty:endy,endx-w:endx+w+1] = mmin
fp32tine[sty-w:sty+w+1,stx:endx] = mmin
fp32tine[endy-w:endy+w+1,stx:endx] = mmin
fp32tine[sty:endy,stx-w:stx+w+1] = mmin
fp32tine[sty:endy,endx-w:endx+w+1] = mmin

fp32tpad[sty-w:sty+w+1,stx:endx] = mmin
fp32tpad[endy-w:endy+w+1,stx:endx] = mmin
fp32tpad[sty:endy,stx-w:stx+w+1] = mmin
fp32tpad[sty:endy,endx-w:endx+w+1] = mmin

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
ax0.set_title('Tomocupy LpRec')

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





mmin = -0.0001
mmax = 0.0001
ax0 = fig.add_subplot(grid[1])
ax0.text(-350,-100,'b)',fontsize=20)
ax0.text(65,1955,f'SSIM: {s1:.3}',fontsize=11.5,backgroundcolor='0.5')
ax0.set_title('Difference between \n LpRec and FourierRec ', fontsize=9.5)
im = ax0.imshow(diff*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')
ax = ax0.inset_axes([0.473,0.05,0.6,0.6])
im = ax.imshow(diff0, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax.add_artist(scalebar)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax.set_xticks([])
ax.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()



mmin = -0.001
mmax = 0.001

ax0 = fig.add_subplot(grid[2])
ax0.set_title('Tomocupy LineRec')
im = ax0.imshow(fp32tpad*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')

ax = ax0.inset_axes([0.255,-0.18,0.6,0.6])
im = ax.imshow(fp32tpad0, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax.add_artist(scalebar)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax.set_xticks([])
ax.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()


mmin = -0.0001
mmax = 0.0001

ax0 = fig.add_subplot(grid[3])
ax0.set_title('Difference between \n LineRec and FourierRec ', fontsize=9.5)
ax0.text(65,1955,f'SSIM: {s2:.3}',fontsize=11.5,backgroundcolor='0.5')
im = ax0.imshow(diff1*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')

ax = ax0.inset_axes([0.473,-0.18,0.6,0.6])
im = ax.imshow(diff10, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax.add_artist(scalebar)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax.set_xticks([])
ax.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()


plt.colorbar(im, cax=cax,format='%.0e')

plt.savefig('difference_to_lprec_linerec.png',dpi=300,bbox_inches='tight')