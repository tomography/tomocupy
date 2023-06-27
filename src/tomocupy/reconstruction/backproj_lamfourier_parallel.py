#!/usr/bin/env python
# -*- coding: utf-8 -*-

# *************************************************************************** #
#                  Copyright © 2022, UChicago Argonne, LLC                    #
#                           All Rights Reserved                               #
#                         Software Name: Tomocupy                             #
#                     By: Argonne National Laboratory                         #
#                                                                             #
#                           OPEN SOURCE LICENSE                               #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
#    this list of conditions and the following disclaimer.                    #
# 2. Redistributions in binary form must reproduce the above copyright        #
#    notice, this list of conditions and the following disclaimer in the      #
#    documentation and/or other materials provided with the distribution.     #
# 3. Neither the name of the copyright holder nor the names of its            #
#    contributors may be used to endorse or promote products derived          #
#    from this software without specific prior written permission.            #
#                                                                             #
#                                                                             #
# *************************************************************************** #
#                               DISCLAIMER                                    #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS         #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT           #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS           #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT    #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,      #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED    #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR      #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF      #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING        #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS          #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                #
# *************************************************************************** #

from tomocupy import utils
from tomocupy import logging
from tomocupy import writer
from tomocupy import cfunc_usfft1d
from tomocupy import cfunc_usfft2d
from tomocupy import cfunc_fft2d
from tomocupy.reconstruction import fbp_filter
from threading import Thread
import cupy as cp
import numpy as np


class BackprojLamFourierParallel():

    def __init__(self, cl_conf, cl_writer):
        
        self.n0 = cl_conf.rh
        self.n1 = cl_conf.n
        self.n2 = cl_conf.n
        self.ntheta = cl_conf.nproj
        self.nthetac = cl_conf.ncproj
        self.detw = cl_conf.n
        self.deth = cl_conf.nz
        self.n1c = cl_conf.ncz
        self.dethc = cl_conf.ncz
        self.center = cl_conf.center
        self.ne = 4*self.detw

        self.cl_usfft1d = cfunc_usfft1d.cfunc_usfft1d(self.n0, self.n1c, self.n2, self.deth)
        self.cl_usfft2d = cfunc_usfft2d.cfunc_usfft2d(
            self.dethc, self.n1, self.n2, self.ntheta, self.detw, self.dethc)  # n0 becomes deth
        self.cl_fft2d = cfunc_fft2d.cfunc_fft2d(self.nthetac, self.detw, self.deth)
        
        pinned_block_size = max(self.n1*self.n0*self.n2, self.n1*self.deth*self.n2, self.ntheta*self.deth*self.detw)
        gpu_block_size = max(self.n1c*self.n0*self.n2, self.n1c*self.deth*self.n2, self.n1*self.dethc*self.n2,self.dethc*self.ntheta*self.detw,self.nthetac*self.deth*self.detw)
        
        # reusable pinned memory blocks
        self.pab0 = utils.pinned_array(np.empty(pinned_block_size,dtype='complex64'))
        self.pab1 = utils.pinned_array(np.empty(pinned_block_size,dtype='complex64'))
        self.pab2 = utils.pinned_array(np.empty(pinned_block_size,dtype='complex64'))
        # pointers (no memory allocation)
        self.pa0 =  self.pab0[:self.n1*self.n0*self.n2].reshape(self.n1, self.n0, self.n2)
        self.pa1 =  self.pab1[:self.n1*self.deth*self.n2].reshape(self.n1,self.deth,self.n2)
        self.pa2 =  self.pab0[:self.ntheta*self.deth*self.detw].reshape(self.ntheta,self.deth,self.detw)
        self.pa3 =  self.pab1[:self.ntheta*self.deth*self.detw].reshape(self.ntheta,self.deth,self.detw)
        
        # reusable gpu memory blocks
        self.gb0 = cp.empty(2*gpu_block_size,dtype='complex64')
        self.gb1 = cp.empty(2*gpu_block_size,dtype='complex64')
        self.gb2 = cp.empty(2*gpu_block_size,dtype='complex64')
        
        # pointers (no memory allocation)
        self.ga0 = self.gb0[:2*self.n1c*self.n0*self.n2].reshape(2,self.n1c,self.n0,self.n2)
        self.ga1 = self.gb1[:2*self.n1c*self.deth*self.n2].reshape(2,self.n1c,self.deth,self.n2)
        self.ga2 = self.gb0[:2*self.n1*self.dethc*self.n2].reshape(2,self.n1,self.dethc,self.n2)
        self.ga3 = self.gb1[:2*self.dethc*self.ntheta*self.detw].reshape(2,self.ntheta,self.dethc,self.detw)
        self.ga4 = self.gb0[:2*self.nthetac*self.deth*self.detw].reshape(2,self.nthetac,self.deth,self.detw)
        self.ga5 = self.gb1[:2*self.nthetac*self.deth*self.detw].reshape(2,self.nthetac,self.deth,self.detw)

        # streams for overlapping data transfers with computations
        self.stream1 = cp.cuda.Stream(non_blocking=False)
        self.stream2 = cp.cuda.Stream(non_blocking=False)
        self.stream3 = cp.cuda.Stream(non_blocking=False)

        # threads for data writing to disk
        self.write_threads = []
        for k in range(cl_conf.args.max_write_threads):
            self.write_threads.append(utils.WRThread())
        
        self.cl_filter = fbp_filter.FBPFilter(
                self.ne, self.deth, self.nthetac, cl_conf.args.dtype)  # note filter is applied on projections, not sinograms as in another methods
        self.wfilter = self.cl_filter.calc_filter(cl_conf.args.fbp_filter)

        self.cl_conf = cl_conf
        self.cl_writer = cl_writer
        self.rec_fun = self.rec_lam
        
    def usfft1d_chunks(self, out_t, inp_t, out_gpu, inp_gpu, phi, direction='fwd'):               
        nchunk = int(np.ceil(self.n1/self.n1c))
        
        for k in range(nchunk+2):
            if(k > 0 and k < nchunk+1):
                with self.stream2:
                    self.cl_usfft1d.adj(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr, phi, self.stream2.ptr)
            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    st, end = (k-2)*self.n1c, min(self.n1,(k-1)*self.n1c)
                    s = end-st
                    out_gpu[(k-2)%2,:s].get(out=out_t[st:end])# contiguous copy, fast  # not swapaxes
            if(k<nchunk):
                with self.stream1:  # cpu->gpu copy
                    st, end = k*self.n1c, min(self.n1,(k+1)*self.n1c)
                    s = end-st
                    inp_gpu[k%2,:s].set(inp_t[st:end])# contiguous copy, fast
            self.stream1.synchronize()
            self.stream2.synchronize()
            self.stream3.synchronize()
            
    def usfft2d_chunks(self, out, inp, out_gpu, inp_gpu, theta, phi, direction='fwd'):
        theta = cp.array(theta)        
        nchunk = int(np.ceil(self.deth/self.dethc))
        
        for k in range(nchunk+2):            
            if(k > 0 and k < nchunk+1):
                with self.stream2:
                    self.cl_usfft2d.adj(out_gpu[(k-1)%2].data.ptr, inp_gpu[(k-1)%2].data.ptr,theta.data.ptr, phi, k-1, self.deth, self.stream2.ptr)
            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    for j in range(out.shape[0]):# non-contiguous copy, slow but comparable with gpu computations
                        st, end = (k-2)*self.dethc, min(self.deth,(k-1)*self.dethc)
                        s = end-st
                        out_gpu[(k-2)%2,j,:s].get(out=out[j,st:end])   #added
            if(k<nchunk):
                with self.stream1:  # cpu->gpu copy           
                    for j in range(inp.shape[0]):
                        st, end = k*self.dethc, min(self.deth,(k+1)*self.dethc)
                        s = end-st
                        inp_gpu[k%2,j,:s].set(inp[j,st:end])# non-contiguous copy, slow but comparable with gpu computations)                    
                        
            self.stream1.synchronize()
            self.stream2.synchronize()
            self.stream3.synchronize()                                    
            
    def fft2_chunks(self, out, inp, out_gpu, inp_gpu, direction='fwd'):
        nchunk = int(np.ceil(self.ntheta/self.nthetac))
        for k in range(nchunk+2):
            if(k > 0 and k < nchunk+1):
                with self.stream2:
                    data0 = inp_gpu[(k-1)%2].real
                    data0 = self.fbp_filter_center(
                        data0, cp.tile(np.float32(0), [data0.shape[0], 1])).astype('complex64')                    
                    self.cl_fft2d.fwd(out_gpu[(k-1)%2].data.ptr,data0.data.ptr,self.stream2.ptr)
            if(k > 1):
                with self.stream3:  # gpu->cpu copy        
                    st, end = (k-2)*self.nthetac, min(self.ntheta,(k-1)*self.nthetac)
                    s = end-st
                    out_gpu[(k-2)%2, :s].get(out=out[st:end])# contiguous copy, fast                                        
                    
            if(k<nchunk):
                with self.stream1:  # cpu->gpu copy
                    st, end = k*self.nthetac, min(self.ntheta,(k+1)*self.nthetac)
                    s = end-st
                    inp_gpu[k%2, :s].set(inp[st:end])# contiguous copy, fast
                    
            self.stream1.synchronize()
            self.stream2.synchronize()
            self.stream3.synchronize()
     
    def fbp_filter_center(self, data, sht=0):
        """FBP filtering of projections with applying the rotation center shift wrt to the origin"""

        tmp = cp.pad(
            data, ((0, 0), (0, 0), (self.ne//2-self.n2//2, self.ne//2-self.n2//2)), mode='edge')
        t = cp.fft.rfftfreq(self.ne).astype('float32')
        w = self.wfilter*cp.exp(-2*cp.pi*1j*t*(-self.center +
                                    sht[:, cp.newaxis]+self.n2/2))  # center fix
        # tmp = cp.fft.irfft(
            # w*cp.fft.rfft(tmp, axis=2), axis=2).astype(self.args.dtype)  # note: filter works with complex64, however, it doesnt take much time
        self.cl_filter.filter(tmp, w, cp.cuda.get_current_stream())
        data[:] = tmp[:, :, self.ne//2-self.n2//2:self.ne//2+self.n2//2]

        return data  # reuse input memory

    def rec_lam(self, data):
        data = data.astype('complex64')        
        self.copy(data,self.pa3)
        #steps 1,2,3 of the fwd operator but in reverse order
        self.fft2_chunks(self.pa2, self.pa3, self.ga4, self.ga5, direction='fwd')
        self.usfft2d_chunks(self.pa1, self.pa2, self.ga2, self.ga3, self.cl_conf.theta, np.pi/2+self.cl_conf.lamino_angle/180*np.pi, direction='adj')
        self.usfft1d_chunks(self.pa0,self.pa1,self.ga0,self.ga1, np.pi/2+self.cl_conf.lamino_angle/180*np.pi, direction='adj')        
        
        u = self.copyTransposed(self.pa0).real
        self.write_parallel(u)
        
    def _copy(self, res, u, st, end):
        res[st:end] = u[st:end]
        
    def copy(self, u, res=[], nthreads=8):
        if res==[]:
            res = np.empty([u.shape[0],u.shape[1],u.shape[2]],dtype=u.dtype)
        nchunk = int(np.ceil(u.shape[0]/nthreads))
        mthreads = []
        for k in range(nthreads):
            th = Thread(target=self._copy,args=(res,u,k*nchunk,min((k+1)*nchunk,u.shape[0])))
            mthreads.append(th)
            th.start()
        for th in mthreads:
            th.join()
        return res

    def _copyTransposed(self, res, u, st, end):
        res[st:end] = u[:,st:end].swapaxes(0,1)        
        
    def copyTransposed(self, u, res=[], nthreads=8):
        if res==[]:
            res = np.empty([u.shape[1],u.shape[0],u.shape[2]],dtype=u.dtype)
        nchunk = int(np.ceil(u.shape[1]/nthreads))
        mthreads = []
        for k in range(nthreads):
            th = Thread(target=self._copyTransposed,args=(res,u,k*nchunk,min((k+1)*nchunk,u.shape[1])))
            mthreads.append(th)
            th.start()
        for th in mthreads:
            th.join()        
        return res
    
    def write_parallel(self,u,nthreads=8):
        nchunk = int(np.ceil(u.shape[0]/nthreads))
        mthreads = []
        for k in range(nthreads):
            th = Thread(target=self.cl_writer.write_data_chunk,args=(u[k*nchunk:min((k+1)*nchunk,u.shape[0])],k*nchunk,min((k+1)*nchunk,u.shape[0]),k))
            mthreads.append(th)
            th.start()
        for th in mthreads:
            th.join()