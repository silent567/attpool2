#!/usr/bin/env python
# coding=utf-8

from .torch_mapping_func import get_torch_pool, torch_sparsemax, torch_gfusedlasso
import torch

class Gfusedmax(torch.nn.Module):
    def __init__(self,gamma=1.0,lam=1.0):
        super(Gfusedmax,self).__init__()
        self.gamma = gamma
        self.lam = lam
    def forward(self,x,graph_size_list,edge_list):
        '''
        x: [N*B], torch tensor, float
        edge_list: [[E,2]]*B, numpy array, int
        graph_size_list: [N_1, N_2, ..., N_B]

        return: [N*B], torch tensor, float
        '''
        fused_x = torch_gfusedlasso.apply(x,edge_list,graph_size_list,self.lam)
        output = [torch_sparsemax.apply(*arg) for arg in zip(torch.split(fused_x,graph_size_list),[-1]*len(graph_size_list),[self.gamma]*len(graph_size_list))]
        return torch.cat(output)

class Sparsemax(torch.nn.Module):
    def __init__(self,gamma=1.0,lam=None):
        super(Sparsemax,self).__init__()
        self.gamma = gamma
    def forward(self,x,graph_size_list,edge_list=None):
        '''
        x: [N*B], torch tensor, float
        graph_size_list: [N_1, N_2, ..., N_B]
        edge_list: None, existing for consistent API

        return: [N*B], torch tensor, float
        '''
        output = [torch_sparsemax.apply(*arg) for arg in zip(torch.split(x,graph_size_list),[-1]*len(graph_size_list),[self.gamma]*len(graph_size_list))]
        return torch.cat(output)

class Softmax(torch.nn.Module):
    def __init__(self,gamma=None,lam=None):
        super(Softmax,self).__init__()
    def forward(self,x,graph_size_list,edge_list=None):
        '''
        x: [N*B], torch tensor, float
        graph_size_list: [N_1, N_2, ..., N_B]
        edge_list: None, existing for consistent API

        return: [N*B], torch tensor, float
        '''
        # pool = get_torch_pool()
        # output = pool.map(torch.nn.functional.softmax,torch.split(x,graph_size_list))
        # with torch.multiprocessing.Pool(8) as p:
            # output = p.map(torch.nn.functional.softmax,torch.split(x,graph_size_list))
        output = [torch.nn.functional.softmax(xx,dim=-1) for xx in torch.split(x,graph_size_list)]
        return torch.cat(output)

class GfusedmaxN(torch.nn.Module):
    def __init__(self,gamma=1.0,lam=1.0):
        super(GfusedmaxN,self).__init__()
        self.gamma = gamma
        self.lam = lam
    def forward(self,x,graph_size_list,edge_list):
        '''
        x: [N*B], torch tensor, float
        edge_list: [[E,2]]*B, numpy array, int
        graph_size_list: [N_1, N_2, ..., N_B]

        return: [N*B], torch tensor, float
        '''
        fused_x = torch_gfusedlasso.apply(x,edge_list,graph_size_list,self.lam)
        output = [gs*torch_sparsemax.apply(*arg) for arg,gs in zip(zip(torch.split(fused_x,graph_size_list),[-1]*len(graph_size_list),[self.gamma]*len(graph_size_list)),graph_size_list)]
        return torch.cat(output)

class SparsemaxN(torch.nn.Module):
    def __init__(self,gamma=1.0,lam=None):
        super(SparsemaxN,self).__init__()
        self.gamma = gamma
    def forward(self,x,graph_size_list,edge_list=None):
        '''
        x: [N*B], torch tensor, float
        graph_size_list: [N_1, N_2, ..., N_B]
        edge_list: None, existing for consistent API

        return: [N*B], torch tensor, float
        '''
        output = [gs*torch_sparsemax.apply(*arg) for arg,gs in zip(zip(torch.split(x,graph_size_list),[-1]*len(graph_size_list),[self.gamma]*len(graph_size_list)),graph_size_list)]
        return torch.cat(output)

class SoftmaxN(torch.nn.Module):
    def __init__(self,gamma=None,lam=None):
        super(SoftmaxN,self).__init__()
    def forward(self,x,graph_size_list,edge_list=None):
        '''
        x: [N*B], torch tensor, float
        graph_size_list: [N_1, N_2, ..., N_B]
        edge_list: None, existing for consistent API

        return: [N*B], torch tensor, float
        '''
        output = [torch.nn.functional.softmax(xx,dim=-1)*gs for xx,gs in zip(torch.split(x,graph_size_list),graph_size_list)]
        return torch.cat(output)

class Sigmoid(torch.nn.Module):
    def __init__(self,gamma=None,lam=None):
        super(Sigmoid,self).__init__()
    def forward(self,x,graph_size_list,edge_list=None):
        '''
        x: [N*B], torch tensor, float
        graph_size_list: [N_1, N_2, ..., N_B]
        edge_list: None, existing for consistent API

        return: [N*B], torch tensor, float
        '''
        return torch.sigmoid(x)

class Tanh(torch.nn.Module):
    def __init__(self,gamma=None,lam=None):
        super(Tanh,self).__init__()
    def forward(self,x,graph_size_list,edge_list=None):
        '''
        x: [N*B], torch tensor, float
        graph_size_list: [N_1, N_2, ..., N_B]
        edge_list: None, existing for consistent API

        return: [N*B], torch tensor, float
        '''
        return torch.tanh(x)

class Sum(torch.nn.Module):
    def __init__(self,gamma=None,lam=None):
        super(Sum,self).__init__()
    def forward(self,x,graph_size_list,edge_list=None):
        '''
        x: [N*B], torch tensor, float
        graph_size_list: [N_1, N_2, ..., N_B]
        edge_list: None, existing for consistent API

        return: [N*B], torch tensor, float
        '''
        return torch.ones_like(x)

class Mean(torch.nn.Module):
    def __init__(self,gamma=None,lam=None):
        super(Mean,self).__init__()
    def forward(self,x,graph_size_list,edge_list=None):
        '''
        x: [N*B], torch tensor, float
        graph_size_list: [N_1, N_2, ..., N_B]
        edge_list: None, existing for consistent API

        return: [N*B], torch tensor, float
        '''
        return torch.cat([torch.ones([gs,],dtype=x.dtype,device=x.device)/gs for gs in graph_size_list])

class Max(torch.nn.Module):
    def __init__(self,gamma=None,lam=None):
        super(Max,self).__init__()
    def forward(self,x,graph_size_list,edge_list=None):
        '''
        x: [N*B], torch tensor, float
        graph_size_list: [N_1, N_2, ..., N_B]
        edge_list: None, existing for consistent API

        return: [N*B], torch tensor, float
        '''
        output = [torch.eye(xx.size(-1), device=x.device, dtype=x.dtype)[torch.argmax(xx,dim=-1)] for xx in torch.split(x,graph_size_list)]
        return torch.cat(output).type(x.dtype)

if __name__ == '__main__':
    size = 10
    import numpy as np
    numpy_a = np.array([ 0.1761,  0.1761,  0.1761,  0.1761,  0.1761,  0.1761,  0.1761,  0.1761, 0.9138,  0.1761,  0.1761,  0.1761,  1.9040, -0.7119,  0.1761,  0.1761])
    a = torch.tensor(numpy_a,requires_grad=True,dtype=torch.float)
    # a = torch.rand(size,requires_grad=True)
    lam = torch.tensor(10.0,requires_grad=True,dtype=torch.float)
    torch_sparse = torch_sparsemax.apply(a,-1,lam)
    numpy_sparse = sparsemax(a.detach().numpy(),lam.item())
    torch_sparse.backward(torch.arange(a.size()[-1],dtype=a.dtype))
    print(a,torch.sum(torch_sparse),np.sum(numpy_sparse))
    print(a.grad, lam.grad)

    # b = a * 30
    # A = (torch.rand(size,size)>0.9).type_as(a)
    # lam = lam.detach()
    # torch_gfusedmax = Gfusedmax(lam,lam)(b,A,-1)
    # numpy_gfusedmax = gfusedmax(b.detach().numpy(),A.numpy(),lam.item(),lam.item())
    # torch_gfusedmax.backward(torch.arange(size,dtype=a.dtype))
    # print(b,torch_gfusedmax,numpy_gfusedmax)
    # print(a.grad,)
