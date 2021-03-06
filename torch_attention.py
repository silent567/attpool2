#!/usr/bin/env python
# coding=utf-8

from .torch_mapping import Gfusedmax, Sparsemax, Softmax, GfusedmaxN, SparsemaxN, SoftmaxN, Mean, Sum, Max, Sigmoid, Tanh, TopK, TopKN
import torch

def standardize(x):
    x = (x - torch.mean(x,dim=-1,keepdim=True))/(torch.std(x,dim=-1,keepdim=True)+1e-7)
    return x

def identity(x):
    return x

class FastFlexAddAttention(torch.nn.Module):
    def __init__(self,input_size,output_size,max_type='softmax',norm_flag=True,lam=1.0,gamma=1.0
                 ,train_gamma_flag=False,train_lam_flag=False,score_activation_func=identity,proj_activation_func=identity):
        super(FastFlexAddAttention,self).__init__()
        self.max_type = max_type
        if max_type == 'softmax':
            self.mapping_func = Softmax()
        elif max_type == 'sparsemax':
            self.register_parameter('gamma',torch.nn.Parameter(torch.tensor(gamma or 1.0,dtype=torch.float),requires_grad=train_gamma_flag))
            self.mapping_func = Sparsemax(gamma=self.gamma)
        elif max_type == 'gfusedmax':
            self.register_parameter('gamma',torch.nn.Parameter(torch.tensor(gamma or 1.0,dtype=torch.float),requires_grad=train_gamma_flag))
            self.register_parameter('lam',torch.nn.Parameter(torch.tensor(lam or 1.0,dtype=torch.float),requires_grad=train_lam_flag))
            self.mapping_func = Gfusedmax(gamma=self.gamma,lam=self.lam)
        elif max_type == 'softmax-n':
            self.mapping_func = SoftmaxN()
        elif max_type == 'sparsemax-n':
            self.register_parameter('gamma',torch.nn.Parameter(torch.tensor(gamma or 1.0,dtype=torch.float),requires_grad=train_gamma_flag))
            self.mapping_func = SparsemaxN(gamma=self.gamma)
        elif max_type == 'gfusedmax-n':
            self.register_parameter('gamma',torch.nn.Parameter(torch.tensor(gamma or 1.0,dtype=torch.float),requires_grad=train_gamma_flag))
            self.register_parameter('lam',torch.nn.Parameter(torch.tensor(lam or 1.0,dtype=torch.float),requires_grad=train_lam_flag))
            self.mapping_func = GfusedmaxN(gamma=self.gamma,lam=self.lam)
        elif max_type == 'sigmoid':
            self.mapping_func = Sigmoid()
        elif max_type == 'tanh':
            self.mapping_func = Tanh()
        elif max_type == 'sum':
            self.mapping_func = Sum()
        elif max_type == 'mean':
            self.mapping_func = Mean()
        elif max_type == 'max':
            self.mapping_func = Max()
        elif max_type == 'topk':
            self.mapping_func = TopK()
        elif max_type == 'topk-n':
            self.mapping_func = TopKN()
        else:
            raise ValueError('Wrong max_type: %s'%max_type)

        self.output_size = output_size
        self.proj_func = torch.nn.Linear(input_size,output_size)
        self.score_func = torch.nn.Linear(input_size,1)
        self.proj_activation_func = proj_activation_func
        self.score_activation_func = score_activation_func
        if norm_flag:
            self.score_norm = standardize
        else:
            self.score_norm = identity
    def forward(self,x,graph_size_list,edge_list):
        '''
        x: [N*B,d], torch tensor, float
        graph_size_list: [N_1, N_2, ..., N_B]
        edge_list: [[E,2]]*B, numpy array, int, each node index starting from 0

        return: [B,d'], torch tensor, float
        '''
        proj_x = self.proj_activation_func(self.proj_func(x)) #[N*B,d']
        score_x = self.score_activation_func(self.score_func(x).squeeze()) #[N*B]
        score_x_norm = torch.cat([self.score_norm(sx) for sx in torch.split(score_x,graph_size_list)])

        # cuda_flag = score_x_norm.is_cuda
        # if cuda_flag:
            # score_x_norm = score_x_norm.cpu()
        weights = self.mapping_func(score_x_norm,graph_size_list,edge_list).unsqueeze(-1) #[N*B,]
        # if cuda_flag:
            # weights = weights.cuda()

        output = torch.stack([torch.sum(px*w,dim=0) for px,w in zip(torch.split(proj_x,graph_size_list),torch.split(weights,graph_size_list))],dim=0)
        return output

class MultipleAttention(torch.nn.Module):
    def __init__(self,att_module,output_size,head_cnt):
        super(MultipleAttention,self).__init__()
        self.head_cnt = head_cnt
        self.output_size = output_size
        # self.attentions = []
        for i in range(head_cnt):
            self.__setattr__('attention%d'%i,att_module(int((i+1)*self.output_size/self.head_cnt)- int(i*self.output_size/self.head_cnt)))
            # self.attentions.append(self.__getattr__('attention%d'%i))
        # self.attentions = [att_module(int((i+1)*self.output_size/self.head_cnt)
                                      # - int(i*self.output_size/self.head_cnt)
                                      # ) for i in range(head_cnt)]
        # for i,att in enumerate(self.attentions):
            # # self.add_module('attention%d'%i,att)
            # self.__setattr__('attention%d'%i,att)
    def forward(self,*args,**kwargs):
        # output = [att(*args,**kwargs) for att in self.attentions]
        output = [self.__getattr__('attention%d'%i)(*args,**kwargs) for i in range(self.head_cnt)]
        return torch.cat(output,dim=-1)
