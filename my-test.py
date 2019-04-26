#!/usr/bin/env python
# coding=utf-8

from torch_mapping import Gfusedmax, Sparsemax, Softmax
from torch_attention import FastFlexAddAttention, MultipleAttention
import numpy as np
import torch

def random_graph():
    graph_size_list = [np.random.randint(5,10) for _ in range(10)]
    edge_list = [np.stack(np.mask_indices(gs,lambda n,k:
                np.triu((1-np.eye(gs))*np.random.rand(gs,gs))>=0.5),axis=-1).astype('int')
                for gs in graph_size_list]
    print(graph_size_list)
    print(len(edge_list),[e.shape for e in edge_list])
    return graph_size_list,edge_list

def test_mapping():
    graph_size_list, edge_list = random_graph()
    x = torch.rand(np.sum(graph_size_list)) + 1
    print(x.size(), x.dtype, x.device, x)

    w = Gfusedmax(lam=0.1)(x,graph_size_list,edge_list)
    print(w.size(),w)
    w = Sparsemax(lam=0.1)(x,graph_size_list,edge_list)
    print(w.size(),w)
    w = Softmax(lam=0.1)(x,graph_size_list,edge_list)
    print(w.size(),w)

def test_attention(att, x, graph_size_list, edge_list):
    if x.is_cuda:
        att.cuda()
    att.share_memory()
    y = att(x, graph_size_list, edge_list)
    print(y.size(), y.dtype, y.device)
    loss = torch.sum(y*y)
    loss.backward()
    grad_x = x.grad
    print(grad_x.size(), grad_x.dtype, grad_x.device)

def test():
    graph_size_list, edge_list = random_graph()
    x = torch.rand([np.sum(graph_size_list),10],requires_grad=True,device='cuda')
    print(x.size(), x.dtype, x.device)
    print('test softmax-att')
    att = FastFlexAddAttention(10, 5, 'softmax', True)
    test_attention(att, x, graph_size_list, edge_list)
    print('test sparsemax-att')
    att = FastFlexAddAttention(10, 5, 'sparsemax', True)
    test_attention(att, x, graph_size_list, edge_list)
    print('test gfusedmax-att')
    att = FastFlexAddAttention(10, 5, 'gfusedmax', True)
    test_attention(att, x, graph_size_list, edge_list)

if __name__ == '__main__':
    test()


