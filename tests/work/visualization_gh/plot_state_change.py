import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import dxchange
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib.patches import Ellipse
[x1,x2] = np.meshgrid(np.arange(-1224,1224)/1224,np.arange(-1224,1224)/1224)
circ = x1**2+x2**2<1

st0 = dxchange.read_tiff('res/213.tiff').astype('float32')
st1 = dxchange.read_tiff('res/206.tiff').astype('float32')
st2 = dxchange.read_tiff('res/249.tiff').astype('float32')
st3 = dxchange.read_tiff('res/482.tiff').astype('float32')

s1=ssim(st0*circ*1000,st1*circ*1000)
# print(ssim(st1f*1000,st1tpad*1000))
# exit()

fig = plt.figure(constrained_layout=True, figsize=(7.5, 6))
grid = fig.add_gridspec(2, 2, height_ratios=[1,1])

mmin = -0.001
mmax = 0.0015

stx = 1224-250-150-300
endx = 1224+250-150-300
sty = 1224-150-150-300+20+90
endy = 1224+350-150-300+20+90
w = 6
diff = (st1-st0)

st10 = st1[sty:endy,stx:endx].copy()
st00 = st0[sty:endy,stx:endx].copy()
diff0 = st10-st00
print(ssim(st00*1000,st10*1000))

st1[sty-w:sty+w+1,stx:endx] = mmin
st1[endy-w:endy+w+1,stx:endx] = mmin
st1[sty:endy,stx-w:stx+w+1] = mmin
st1[sty:endy,endx-w:endx+w+1] = mmin
st0[sty-w:sty+w+1,stx:endx] = mmin
st0[endy-w:endy+w+1,stx:endx] = mmin
st0[sty:endy,stx-w:stx+w+1] = mmin
st0[sty:endy,endx-w:endx+w+1] = mmin


stx = 1224-200-150-200+200+40
endx = 1224+450-150-200+200-40
sty = 1224-300-150-300-400
endy = 1224+350-150-300-400-80
st20 = st2[sty:endy,stx:endx].copy()
st30 = st3[sty:endy,stx:endx].copy()
st2[sty-w:sty+w+1,stx:endx] = mmin
st2[endy-w:endy+w+1,stx:endx] = mmin
st2[sty:endy,stx-w:stx+w+1] = mmin
st2[sty:endy,endx-w:endx+w+1] = mmin
st3[sty-w:sty+w+1,stx:endx] = mmin
st3[endy-w:endy+w+1,stx:endx] = mmin
st3[sty:endy,stx-w:stx+w+1] = mmin
st3[sty:endy,endx-w:endx+w+1] = mmin

diff[sty-w:sty+w+1,stx:endx] = mmin
diff[endy-w:endy+w+1,stx:endx] = mmin
diff[sty:endy,stx-w:stx+w+1] = mmin
diff[sty:endy,endx-w:endx+w+1] = mmin

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(st1*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')
ax0.set_title('Before water redistribution (1.1x)',fontsize=14)

ax = ax0.inset_axes([0.26,0.045,0.6,0.6])
im = ax.imshow(st10, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25,location='lower right')
ax.add_artist(scalebar)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax.arrow(70, 70, 60, 10, width = 20,facecolor='white')
ax.set_xticks([])
ax.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()




ax0 = fig.add_subplot(grid[1])
ax0.set_title('After water redistribution (1.1x)',fontsize=14)
im = ax0.imshow(st0*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')
ax = ax0.inset_axes([0.48,0.045,0.6,0.6])
im = ax.imshow(st00, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.13, "um", length_fraction=0.25,location='lower right')
ax.add_artist(scalebar)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax.arrow(70, 70, 60, 10, width = 20,facecolor='white')
ax.set_xticks([])
ax.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()


mmin = -0.0003
mmax = 0.00018
st2[sty-w:sty+w+1,stx:endx] = mmin
st2[endy-w:endy+w+1,stx:endx] = mmin
st2[sty:endy,stx-w:stx+w+1] = mmin
st2[sty:endy,endx-w:endx+w+1] = mmin
st3[sty-w:sty+w+1,stx:endx] = mmin
st3[endy-w:endy+w+1,stx:endx] = mmin
st3[sty:endy,stx-w:stx+w+1] = mmin
st3[sty:endy,endx-w:endx+w+1] = mmin

st2[sty-w:sty+w+1,stx:endx] = mmin
st2[endy-w:endy+w+1,stx:endx] = mmin
st2[sty:endy,stx-w:stx+w+1] = mmin
st2[sty:endy,endx-w:endx+w+1] = mmin
st3[sty-w:sty+w+1,stx:endx] = mmin
st3[endy-w:endy+w+1,stx:endx] = mmin
st3[sty:endy,stx-w:stx+w+1] = mmin
st3[sty:endy,endx-w:endx+w+1] = mmin

ax0 = fig.add_subplot(grid[2])
im = ax0.imshow(st2*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.45/5, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')
ax0.set_title('After water redistribution (5x)',fontsize=14)

ax = ax0.inset_axes([0.26,-0.18,0.6,0.6])
im = ax.imshow(st20, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.45/5, "um", length_fraction=0.25,location='lower right')
ax.add_artist(scalebar)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax.set_xticks([])
ax.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()


ax0 = fig.add_subplot(grid[3])
ax0.set_title('Formed gas hydrate (5x)',fontsize=14)
im = ax0.imshow(st3*circ, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.45/5, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')
ax = ax0.inset_axes([0.48,-0.18,0.6,0.6])
im = ax.imshow(st30, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(3.45/5, "um", length_fraction=0.25,location='lower right')
ax.add_artist(scalebar)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax.set_xticks([])
ax.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()


plt.colorbar(im, cax=cax,format='%.0e')
plt.savefig('state_change.png',dpi=300,bbox_inches='tight')