void get_index_cuda(at::Tensor coords, at::Tensor order, at::Tensor query, at::Tensor index);
void merge_cuda_backward(at::Tensor grad_output, at::Tensor out_order, at::Tensor order, at::Tensor features, at::Tensor grad_input);
void merge_cuda_forward(at::Tensor coords, at::Tensor features, at::Tensor order);
void knn_cuda(at::Tensor known, at::Tensor unknown, at::Tensor batch,  at::Tensor dist, at::Tensor idx, const int k, const int batchsize);

