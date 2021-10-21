//#include <torch/torch.h>
#include <torch/extension.h>
#include "fusion_cuda.h"


PYBIND11_MODULE (TORCH_EXTENSION_NAME, fusion)
{
  fusion.def ("merge_cuda_forward", &merge_cuda_forward, "forward: remove repeated and merge features (CUDA)");
  fusion.def ("merge_cuda_backward", &merge_cuda_backward, "backward: remove repeated and mrege features (CUDA)");
  fusion.def ("get_index_cuda", &get_index_cuda, "query and get index from source (CUDA)");
  fusion.def ("knn_cuda", &knn_cuda, "knn (CUDA)");
}

