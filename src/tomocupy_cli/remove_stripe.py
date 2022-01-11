
import cupy as cp
import torch
from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)


def remove_stripe_fw(data, sigma, wname, level):
    """Remove stripes with wavelet filtering"""

    [nproj, nz, ni] = data.shape

    nproj_pad = nproj + nproj // 8
    xshift = int((nproj_pad - nproj) // 2)

    # Accepts all wave types available to PyWavelets
    xfm = DWTForward(J=1, mode='symmetric', wave=wname).cuda()
    ifm = DWTInverse(mode='symmetric', wave=wname).cuda()

    # Wavelet decomposition.
    cc = []
    sli = torch.zeros([nz, 1, nproj_pad, ni], device='cuda')

    sli[:, 0, (nproj_pad - nproj)//2:(nproj_pad + nproj) //
        2] = torch.as_tensor(data.swapaxes(0, 1), device='cuda')
    for k in range(level):
        sli, c = xfm(sli)
        cc.append(c)
        # FFT
        fcV = torch.fft.fft(cc[k][0][:, 0, 1], axis=1)
        _, my, mx = fcV.shape
        # Damping of ring artifact information.
        y_hat = torch.fft.ifftshift((torch.arange(-my, my, 2).cuda() + 1) / 2)
        damp = -torch.expm1(-y_hat**2 / (2 * sigma**2))
        fcV *= torch.transpose(torch.tile(damp, (mx, 1)), 0, 1)
        # Inverse FFT.
        cc[k][0][:, 0, 1] = torch.fft.ifft(fcV, my, axis=1).real

    # Wavelet reconstruction.
    for k in range(level)[::-1]:
        shape0 = cc[k][0][0, 0, 1].shape
        sli = sli[:, :, :shape0[0], :shape0[1]]
        sli = ifm((sli, cc[k]))

    data = cp.asarray(sli[:, 0, (nproj_pad - nproj) //
                      2:(nproj_pad + nproj)//2, :ni])
    data = data.swapaxes(0, 1)

    return data
