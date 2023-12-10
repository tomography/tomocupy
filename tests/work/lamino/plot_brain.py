import dxchange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

rec_linerec = dxchange.read_tiff_stack('/data/2023-04-Nikitin_rec/brain_lam_20deg_middle_421_rec_linerec/recon_00000.tiff', ind=range(800,801))[0]
rec = -dxchange.read_tiff_stack('/data/2023-04-Nikitin_rec/brain_lam_20deg_middle_421_rec/recon_00000.tiff', ind=range(800+50,801+50))[0]


mmin = -0.001
mmax = 0.001
rec_linerec[0,0]= mmin
rec_linerec[0,1]= mmax
rec_linerecp = rec_linerec[1150:1650,550:1050]
rec_linerecp[0,0]= mmin
rec_linerecp[0,1]= mmax
rec_linerecpp = rec_linerecp[200:400,200:400]
rec_linerecpp[0,0]= mmin
rec_linerecpp[0,1]= mmax

rec[0,0]= mmin
rec[0,1]= mmax
recp = rec[1150:1650,550:1050]
recp[0,0]= mmin
recp[0,1]= mmax
recpp = recp[200:400,200:400]
recpp[0,0]= mmin
recpp[0,1]= mmax




fig = plt.figure(constrained_layout=True, figsize=(10.7, 2.8))
grid = fig.add_gridspec(1, 3, height_ratios=[1])

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(rec_linerec, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(1.38, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')
plt.savefig('rec_linerec.png',dpi=300,bbox_inches='tight')


fig = plt.figure(constrained_layout=True, figsize=(10.7, 2.8))
grid = fig.add_gridspec(1, 3, height_ratios=[1])

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(rec_linerecp, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(1.38, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
ax0.set_xticks([])
ax0.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()
plt.savefig('rec_linerecp.png',dpi=300,bbox_inches='tight')


fig = plt.figure(constrained_layout=True, figsize=(10.7, 2.8))
grid = fig.add_gridspec(1, 3, height_ratios=[1])

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(rec_linerecpp, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(1.38, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
# plt.colorbar(im, cax=cax,format='%.0e')
ax0.set_xticks([])
ax0.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()
plt.savefig('rec_linerecpp.png',dpi=300,bbox_inches='tight')
# ax = ax0.inset_axes([-0.26,-0.17,1.5,1])
# im = ax.imshow(rec_linerecp, cmap='gray',vmin=mmin, vmax=mmax)
# scalebar = ScaleBar(1.38, "um", length_fraction=0.25)
# ax.add_artist(scalebar)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)
# ax.set_xticks([])
# ax.set_yticks([])
# cb = plt.colorbar(im, cax=cax,format='%.0e')
# cb.remove()

fig = plt.figure(constrained_layout=True, figsize=(10.7, 2.8))
grid = fig.add_gridspec(1, 3, height_ratios=[1])

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(rec, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(1.38, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax,format='%.0e')
plt.savefig('rec.png',dpi=300,bbox_inches='tight')


fig = plt.figure(constrained_layout=True, figsize=(10.7, 2.8))
grid = fig.add_gridspec(1, 3, height_ratios=[1])

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(recp, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(1.38, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
# plt.colorbar(im, cax=cax,format='%.0e')
ax0.set_xticks([])
ax0.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()
plt.savefig('recp.png',dpi=300,bbox_inches='tight')


fig = plt.figure(constrained_layout=True, figsize=(10.7, 2.8))
grid = fig.add_gridspec(1, 3, height_ratios=[1])

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(recpp, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(1.38, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
# plt.colorbar(im, cax=cax,format='%.0e')
ax0.set_xticks([])
ax0.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()
plt.savefig('recpp.png',dpi=300,bbox_inches='tight')


rec_linerec = dxchange.read_tiff_stack('/data/2023-04-Nikitin_rec/brain_lam_20deg_middle_421_rec_linerec/recon_00000.tiff', ind=range(400,1201))
rec = -dxchange.read_tiff_stack('/data/2023-04-Nikitin_rec/brain_lam_20deg_middle_421_rec/recon_00000.tiff', ind=range(400+50,1201+50))



mmin = -0.001
mmax = 0.001

rec_linerecv = rec_linerec[:,1400]
rec_linerecv[0,0]= mmin
rec_linerecv[0,1]= mmax
rec_linerecvp = rec_linerecv[100:600,550:1050]
rec_linerecvp[0,0]= mmin
rec_linerecvp[0,1]= mmax
rec_linerecvpp = rec_linerecvp[150:350,150:350]
rec_linerecvpp[0,0]= mmin
rec_linerecvpp[0,1]= mmax

recv = rec[:,1400]
recv[0,0]= mmin
recv[0,1]= mmax
recvp = recv[100:600,550:1050]
recvp[0,0]= mmin
recvp[0,1]= mmax
recvpp = recvp[150:350,150:350]
recvpp[0,0]= mmin
recvpp[0,1]= mmax


fig = plt.figure(constrained_layout=True, figsize=(10.7, 2.8))
grid = fig.add_gridspec(1, 3, height_ratios=[1])

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(rec_linerecv, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(1.38, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
# plt.colorbar(im, cax=cax,format='%.0e')
# ax0.set_xticks([])
# ax0.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()
plt.savefig('rec_linerecv.png',dpi=300,bbox_inches='tight')

fig = plt.figure(constrained_layout=True, figsize=(10.7, 2.8))
grid = fig.add_gridspec(1, 3, height_ratios=[1])

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(rec_linerecvp, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(1.38, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
# plt.colorbar(im, cax=cax,format='%.0e')
ax0.set_xticks([])
ax0.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()
plt.savefig('rec_linerecvp.png',dpi=300,bbox_inches='tight')

fig = plt.figure(constrained_layout=True, figsize=(10.7, 2.8))
grid = fig.add_gridspec(1, 3, height_ratios=[1])

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(rec_linerecvpp, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(1.38, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
# plt.colorbar(im, cax=cax,format='%.0e')
ax0.set_xticks([])
ax0.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()
plt.savefig('rec_linerecvpp.png',dpi=300,bbox_inches='tight')




fig = plt.figure(constrained_layout=True, figsize=(10.7, 2.8))
grid = fig.add_gridspec(1, 3, height_ratios=[1])

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(recv, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(1.38, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
# plt.colorbar(im, cax=cax,format='%.0e')
# ax0.set_xticks([])
# ax0.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()
plt.savefig('recv.png',dpi=300,bbox_inches='tight')

fig = plt.figure(constrained_layout=True, figsize=(10.7, 2.8))
grid = fig.add_gridspec(1, 3, height_ratios=[1])

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(recvp, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(1.38, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
# plt.colorbar(im, cax=cax,format='%.0e')
ax0.set_xticks([])
ax0.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()
plt.savefig('recvp.png',dpi=300,bbox_inches='tight')


fig = plt.figure(constrained_layout=True, figsize=(10.7, 2.8))
grid = fig.add_gridspec(1, 3, height_ratios=[1])

ax0 = fig.add_subplot(grid[0])
im = ax0.imshow(recvpp, cmap='gray',vmin=mmin, vmax=mmax)
scalebar = ScaleBar(1.38, "um", length_fraction=0.25)
ax0.add_artist(scalebar)
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.1)
# plt.colorbar(im, cax=cax,format='%.0e')
ax0.set_xticks([])
ax0.set_yticks([])
cb = plt.colorbar(im, cax=cax,format='%.0e')
cb.remove()
plt.savefig('recvpp.png',dpi=300,bbox_inches='tight')

