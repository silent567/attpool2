Requires: [gfl](https://gitlab.com/hao.tang/gfl), pytorch >= 1.0, numpy, python3

Hao.Tang's repo: https://github.com/silent567/attpool2

# Quick Setup on Ubuntu
- Install [gfl](https://gitlab.com/hao.tang/gfl). Make sure that libgsl is installed and lib-path is in the `LD_LIBRARY_PATH`
- Install pytorch

# File description 
import line: torch_attention.py --> torch_mapping.py --> torch_mapping_func.py --> mapping.py, mp_utils.py
- mp_utils.py: unifying all usage of multiprocessing, so that it's easy to change settings in one file
- mapping.py: wrapper of pygfl.easy.solve_gfl for API translation
- torch_mapping_func.py: all util functions. Logically, it should be combined with torch_mapping.py, and they are separated for isolation of multiprocessing usage for better debugging.
- torch_mapping.py: definition of Softmax, Sparsemax, and Gfusedmax in the same API
- torch_attention.py: attentional pooling, and multi-head attentional pooling
- my-test.py: simple test file

# Some issues related to multiprocessing
## Four different types of multiprocessing: 
- multiprocessing: python3 package, process parallelization
- multiprocessing.dummy: python3 package, thread parallelization (actually API translation of threading)
- pytorch.multiprocessing: pytorch wrapper of multiprocessing, process parallelization, sharing CUDA tensors
- pytorch.DataParallel: GPU parallelization, no idea of the parallelization level for CPU, numpy etc.

## Implementation issues:
- multiprocessing: NO. not compatible with CUDA tensors
- multiprocessing.dummy: YES. works well in any settings
- pytorch.multiprocessing: YES/NO. 
  - No autograd support, and therefore can't be used for multiple graphs' sparsemax.
  - Runtime error when utilized in the backward function of gfusedmax, no idea of the reasons
  - Works well in the forward pass of gfusedmax
- pytorch.DataParallel: NO. model-level parallelization, it is actually implemented in the [find-clique repo](https://github.com/silent567/find_clique). Refer there for more details. Quick message is that it doesn't improve the speed a lot (~6s/iter->5.s/iter).

## Intuition from previous experiments:
- The bottleneck is gfusedlasso. Sparsemax+graph construction together takes ~0.1s/iter while Gfusedmax takes ~6s/iter.
- multiprocessing.dummy (process number = 8): small improvement, from ~6s/iter --> ~5s/iter
- DataParallel: little improvement, from ~5s/iter --> ~4.5s/iter
- pytorch.multiprocessing: hard to implement, no enough experiment yet.
- **Currently, it takes >= 100 hours for one training. To make one training finished within one day, we need <= 1s/iter**

## Previous successful solution on [DGCNN](https://github.com/silent567/pytorch_DGCNN):
- Global multiprocessing.Pool and pytorch.multiprocessing.Pool
- mp.Pool for numpy functions and torchmp.Pool for torch functions
- 2-times speed up with 10 processes.
- same implementation has troubles in the [find-clique repo](https://github.com/silent567/find_clique).
- Please refer to [attpool](https://github.com/silent567/attpool) for more details

# FAQ
- Where exactly do you use multi-processing?
  - Only two files are involved in multiprocessing: torch_mapping_func.py and mp_utils.py
  - In mp_utils.py: it basically sets up the configuration for multiprocessing
  - In torch_mapping_func.py, all mp are utilized in Gfusedlasso:
    - Forward pass: line [99](https://github.com/silent567/attpool2/blob/8800e6ee168c010e2986ef1e47d16af3e3ebe9ab/torch_mapping_func.py#L99) to line [102](https://github.com/silent567/attpool2/blob/8800e6ee168c010e2986ef1e47d16af3e3ebe9ab/torch_mapping_func.py#L102)
    - Backward pass: line [126](https://github.com/silent567/attpool2/blob/8800e6ee168c010e2986ef1e47d16af3e3ebe9ab/torch_mapping_func.py#L126) to line [132](https://github.com/silent567/attpool2/blob/8800e6ee168c010e2986ef1e47d16af3e3ebe9ab/torch_mapping_func.py#L132)
