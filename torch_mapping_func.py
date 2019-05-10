#!/usr/bin/env python
# coding=utf-8

from .mp_utils import get_torch_pool
from .mapping import numpy_gfusedlasso
import numpy as np
import torch
# import time

def backward_gfusedmax_torch_1D(output, edge_list, grad_output, needs_input_grad):
    '''
    output: [N], torch tensor, float
    edge_list: [E,2], numpy ndarray, int
    grad_output: [N], torch tensor, float
    needs_input_grad: [bool]*?

    return grad_input: [N], torch tensor, float
    '''
    grad_input = [None]*len(needs_input_grad)
    grad_x = torch.zeros_like(grad_output) if (len(needs_input_grad)>0 and needs_input_grad[0]) else None
    grad_lam = torch.zeros([], dtype=grad_output.dtype, device=grad_output.device) if (len(needs_input_grad)>3 and needs_input_grad[3]) else None
    if grad_x is None and grad_lam is None:
        return grad_input
    if grad_lam is not None:
        # method 0
        # edge_list = torch.tensor(edge_list,device=output.device)
        # sign = torch.sign(output[edge_list[:,0]] - output[edge_list[:,1]])
        # sign = torch.cat([sign,-sign],dim=0)
        # aggr_sign = torch.zeros_like(output)
        # aggr_sign.scatter_add_(0,torch.cat([edge_list[:,0],edge_list[:,1]],dim=0),sign)

        # method 1
        edge_list = torch.tensor(edge_list,device=output.device)
        edge_list = torch.cat([edge_list,edge_list[:,[1,0]]],dim=0)
        sign = torch.sign(output[edge_list[:,0]] - output[edge_list[:,1]])
        aggr_sign = torch.zeros_like(output)
        aggr_sign.scatter_add_(0,edge_list[:,0],sign)

    # method 0
    # unique_output = torch.unique(output)
    # for uo in unique_output.unbind():
        # mask = output == uo
        # if grad_x is not None:
            # grad_x[mask] = torch.sum(grad_output[mask])/torch.sum(mask)
        # if grad_lam is not None:
            # grad_lam -= torch.sum(grad_output[mask])*torch.sum(aggr_sign[mask])/torch.sum(mask)

    # method 1
    unique_output, mask_index = torch.unique(output, return_inverse=True)
    grad_output_sum = torch.zeros_like(unique_output)
    grad_output_sum.scatter_add_(0,mask_index,grad_output)
    mask_sum = torch.zeros_like(unique_output)
    mask_sum.scatter_add_(0,mask_index,torch.ones_like(grad_output))
    if grad_x is not None:
        grad_x = torch.gather(grad_output_sum/mask_sum,0,mask_index)
    if grad_lam is not None:
        aggr_sign_sum = torch.zeros_like(unique_output)
        aggr_sign_sum.scatter_add_(0,mask_index,aggr_sign)
        grad_lam = -torch.sum(grad_output_sum*aggr_sign_sum/mask_sum)

    if grad_x is not None:
        grad_input[0] = grad_x
    if grad_lam is not None:
        grad_input[3] = grad_lam

    return grad_input

class torch_sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dim=-1, gamma=1.0):
        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma,device=inp.device,dtype=inp.dtype)

        reshape_size = [1]*len(inp.size())
        reshape_size[dim] = -1

        inp_div = inp / gamma
        inp_sorted,_ = torch.sort(inp_div, dim=dim, descending=True)
        cumsum = torch.cumsum(inp_sorted,dim=dim)
        mask = (1+torch.arange(1,inp_div.size()[dim]+1,device=inp.device,dtype=inp.dtype)
                .reshape(reshape_size)*inp_sorted) > cumsum
        mask = mask.type_as(inp)
        tau = (torch.sum(inp_sorted*mask,dim=dim,keepdim=True)-1.)/torch.sum(mask,dim=dim,keepdim=True,dtype=inp.dtype)
        output = torch.clamp(inp_div-tau,min=0)

        ctx.dim = dim
        ctx.save_for_backward(inp, gamma, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, gamma, output = ctx.saved_tensors
        dim = ctx.dim

        mask = (output > 0).type_as(inp)
        masked_grad_output = grad_output*mask
        masked_grad_output -= mask * (torch.sum(masked_grad_output,dim=dim,keepdim=True)\
                            / (torch.sum(mask,dim=dim,keepdim=True)+1e-5))

        grad_inp = None
        if ctx.needs_input_grad[0]:
            grad_inp = masked_grad_output / gamma
        if len(ctx.needs_input_grad) < 2:
            return grad_inp

        if ctx.needs_input_grad[1]:
            raise ValueError('No gradient is defined for dim argument of sparsemax')
        if len(ctx.needs_input_grad) < 3:
            return grad_inp,None

        grad_gamma = None
        if ctx.needs_input_grad[2]:
            grad_gamma = -torch.sum(masked_grad_output*inp*mask)/gamma/gamma

        return grad_inp, None, grad_gamma

class torch_gfusedlasso(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, edge_list, graph_size_list, lam=1.0):
        '''
        inp: [N*B], torch tensor, float
        edge_list: [[E,2]]*B, numpy array, int
        graph_size_list: [N_1, N_2, ..., N_B], list, int
        lam: [], torch tensor, float

        return: [N*B], torch tensor, float
        '''
        if torch.is_tensor(lam):
            lam = float(lam.detach().cpu().numpy())

        cuda_flag = inp.is_cuda
        if cuda_flag:
            inp = inp.cpu()
        inp = inp.detach().numpy() #[N*M]
        inp_list = np.split(inp, np.cumsum(graph_size_list)[:-1]) #[M]*N
        pool = get_torch_pool()
        output_list = pool.starmap(numpy_gfusedlasso,zip(inp_list,edge_list,[lam]*len(inp_list))) #[M]*N
        output = torch.from_numpy(np.concatenate(output_list)) #[N*M]
        if cuda_flag:
            output = output.cuda()

        ctx.graph_size_list = graph_size_list
        ctx.edge_list = edge_list
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # tt = time.time()

        if len(ctx.needs_input_grad) < 1:
            return
        if len(ctx.needs_input_grad) > 1 and ctx.needs_input_grad[1]:
            raise ValueError('Gradients for edge_list in the gfusedlasso is not implemented')
        if len(ctx.needs_input_grad) > 2 and ctx.needs_input_grad[2]:
            raise ValueError('Gradients for graph_size_list in the gfusedlasso is not implemented')

        edge_list, graph_size_list = ctx.edge_list, ctx.graph_size_list
        output, = ctx.saved_tensors

        # method 0
        # grad_inp_list = [backward_gfusedmax_torch_1D(*arg) for arg in
                        # zip(torch.split(output, graph_size_list),
                            # edge_list,
                            # torch.split(grad_output, graph_size_list),
                            # [ctx.needs_input_grad]*len(edge_list))]
        # grad_inp = [None]*len(ctx.needs_input_grad)
        # if (len(ctx.needs_input_grad)>0 and ctx.needs_input_grad[0]):
            # grad_inp[0] = torch.cat([gi[0] for gi in grad_inp_list],dim=0)
        # if (len(ctx.needs_input_grad)>3 and ctx.needs_input_grad[3]):
            # grad_inp[3] = torch.sum(torch.stack([gi[3] for gi in grad_inp_list],dim=0),dim=0)

        # method 1
        start_index = [0,]+list(np.cumsum(graph_size_list)[:-1])
        edge_list = np.concatenate([el+si for el,si in zip(edge_list,start_index)],axis=0)
        grad_inp = backward_gfusedmax_torch_1D(output, edge_list, grad_output, ctx.needs_input_grad)

        # print('backward time:', time.time()-tt)
        return tuple(grad_inp)

def torch_topkmax(x, ratio=0.5):
    '''
    x: [N] torch.tensor, float
    ratio: float
    '''
    _,topk_index = torch.topk(x, int(x.size(-1)*ratio), dim=-1, sorted=False)
    mask = torch.zeros_like(x)
    mask[topk_index] = 1
    masked_x = torch.tanh(x*mask)
    return masked_x/x.size(-1)


