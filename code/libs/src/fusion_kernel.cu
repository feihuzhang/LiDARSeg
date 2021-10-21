#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include "fusion_cuda.h"


#define CUDA_NUM_THREADS 256
#define THREADS_PER_BLOCK 64 

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])
/*
#ifdef __cplusplus
    extern "C" {
#endif
*/
typedef thrust::tuple<int, int, int, int, int> IntTuple;

struct less 
{
    __host__ __device__
    bool operator()(const IntTuple& t1, const IntTuple& t2)
    {

     if (t1.get<3>() != t2.get<3>())
         return t1.get<3>() < t2.get<3>();
     if (t1.get<0>() != t2.get<0>())
         return t1.get<0>() < t2.get<0>();
     if (t1.get<1>() != t2.get<1>())
         return t1.get<1>() < t2.get<1>();
     return t1.get<2>() < t2.get<2>();
    }
}; 

__global__ void remove_repeat(const int n, const int *x, const int*y, const int* z, const int*batch, const int* idx,  int* dst) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }
    if(index==0){
       dst[index]=idx[index];
       return;
    }
    int pre = index -1;
    if(x[pre]==x[index]&&y[pre]==y[index]&&z[pre]==z[index]&&batch[pre]==batch[index])
        dst[index]=-1-idx[index];
    else
        dst[index]=idx[index];

}
__global__ void merge_feature(const int n, const int * order, const int channel, float* feature) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n-1) {
        return;
    }
    if(order[index]<0)
        return;

    int next = index + 1;
//    if(next>=n)
//        return;
    if(order[next]>=0)
        return;
    while(next<n){
        if(order[next]>=0)
            break;
        int loc_next = -1 - order[next];
        for(int i=0;i<channel;i++){
            feature[order[index]*channel+i] += feature[loc_next*channel+i];
        }
        next ++;
    }
    for(int i=0;i<channel; i++){
        feature[order[index]*channel+i] /= (next-index);
    }

}
__global__ void merge_feature_max(const int n, const int * order, const int channel, float* feature) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n-1) {
        return;
    }
    if(order[index]<0)
        return;

    int next = index + 1;
//    if(next>=n)
//        return;
    int cur = order[index];
    while(next<n){
        if(order[next]>=0)
            break;
        int loc_next = -1 - order[next];
        for(int i=0;i<channel;i++){
            if(feature[cur*channel+i] < feature[loc_next*channel+i])
                feature[cur*channel+i] = feature[loc_next*channel+i];
        }
        next ++;
    }

}

/*
order_out: dense mapping order of the output/selected tensor (repeated removed).
*/
__global__ void feature_backward1(const int n, const float* grad_out, const int * order_out, const int channel, float* grad_feature) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }
    int source = order_out[index];
    for(int i=0;i<channel;i++){
        grad_feature[source*channel+i]=grad_out[index*channel+i];
    }
}
/*
order: sorted mapping order of the input tensor (including repeated points/coords).
Fill the gradient of repeated points/coords.
*/
__global__ void feature_backward2(const int n, const int * order, const int channel, float* grad_feature) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n-1) {
        return;
    }
    
    int source = order[index];
    if(source<0)
        return;
    int next = index +1;
//    if(next>=n)
//        return;
    if(order[next]>=0)
        return;
    while(next<n){
        if(order[next]>=0)
            break;
        int loc_next = -1 - order[next];
 
        for(int i=0;i<channel;i++){
            grad_feature[loc_next*channel+i]=grad_feature[source*channel+i];
        }
        next ++;
    }
    float norm = next - index;
    for(int k =index; k<next; k++){
        for(int i=0;i<channel;i++){
            grad_feature[order[k]*channel+i] /= norm;
        }
    }

}

__global__ void feature_backward_max(const int n, const int * order, const float* feature, const int channel, float* grad_feature) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n-1) {
        return;
    }
    
    int source = order[index];
    if(source<0)
        return;
    int next = index +1;
//    if(next>=n)
//        return;
    if(order[next]>=0)
        return;
    while(next<n){
        if(order[next]>=0)
            break;
        next ++;
    }

    for(int i=0;i<channel;i++){
        int cur = order[index];
        for(int k =index; k<next; k++){
            int loc_next = -1 - order[k];
            if(feature[cur*channel+i] < feature[loc_next*channel+i])
                cur = loc_next;
        }
        float temp = grad_feature[order[index]*channel+i];
        grad_feature[order[index]*channel+i] = 0;
        grad_feature[cur*channel+i] = temp;
    }

}
 __device__ int compare(const int x, const int y, const int z, const int batch, const int * que) {
     if ((batch==que[3])&&(x==que[0])&&(y==que[1])&&(z==que[2]))
         return -1;
     if (batch != que[3])
         return int(batch < que[3]);
     if (x != que[0])
         return int(x < que[0]);
     if (y != que[1])
         return int(y < que[1]);
     return int(z < que[2]);
}
__global__ void search(const int n, const int length, const int *x, const int*y, const int* z, const int*batch, const int* order, const int* que, int* idx) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    int left = 0;
    int right = length-1;
    int cur = (left+right)/2;
    while(left <= right){
        int flag = compare(x[cur], y[cur], z[cur], batch[cur], que + index*4);
        if(flag==-1){
            idx[index] = order[cur];
            return;
        }
        if(flag==0)
            right = cur-1;
        if(flag==1)
            left = cur +1;
        cur = (left + right)/2;
    }
    idx[index] = -1;

}     

void merge_cuda_forward(at::Tensor coords, at::Tensor features, at::Tensor order){
    int num = coords.size(1);
    int channel = coords.size(0);
    int fea_channel = features.size(1);
	
    if(channel!=4){
        printf("error in coords shape!\n");
        exit(0);
    }
    float *feas = features.data<float>();
    int *locs = coords.data<int>();
    int *dst_order = order.data<int>();

    thrust::device_ptr<int> dev_ptr(locs);
    thrust::device_vector<int> x (dev_ptr, dev_ptr + num);
    thrust::device_vector<int> y(dev_ptr+num, dev_ptr +2*num);
    thrust::device_vector<int> z(dev_ptr+2*num, dev_ptr+3*num);
    thrust::device_vector<int> batch(dev_ptr+3*num, dev_ptr+4*num);

    thrust::device_ptr<int> dev_idx(dst_order);
    thrust::device_vector<int> idx(dev_idx, dev_idx + num);

    sort(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), batch.begin(), idx.begin())), thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(), z.end(), batch.end(), idx.end())), less());

    const int* ptr_x = thrust::raw_pointer_cast(&x[0]);
    const int* ptr_y = thrust::raw_pointer_cast(&y[0]);
    const int* ptr_z = thrust::raw_pointer_cast(&z[0]);
    const int* ptr_batch = thrust::raw_pointer_cast(&batch[0]);
    const int* ptr_idx = thrust::raw_pointer_cast(&idx[0]);


    int threads = (num + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    remove_repeat<<< threads, CUDA_NUM_THREADS >>>(num, ptr_x, ptr_y, ptr_z, ptr_batch, ptr_idx, dst_order);
    merge_feature<<< threads, CUDA_NUM_THREADS >>>(num, dst_order, fea_channel, feas);

    x.clear();
    thrust::device_vector<int>().swap(x);
    y.clear();
    thrust::device_vector<int>().swap(y);
    z.clear();
    thrust::device_vector<int>().swap(z);
    batch.clear();
    thrust::device_vector<int>().swap(batch);
    idx.clear();
    thrust::device_vector<int>().swap(idx);

}
void merge_cuda_backward(at::Tensor grad_output, at::Tensor out_order, at::Tensor order, at::Tensor features, at::Tensor grad_input){
    int num1 = order.size(0);
    int num2 = out_order.size(0);

    int channel = grad_output.size(1);

    float *feas = features.data<float>();
    int *out_idx = out_order.data<int>();
    int *idx = order.data<int>();
    float * gradout = grad_output.data<float>();
    float * gradin = grad_input.data<float>();
    
   
    int threads = (num2 + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    feature_backward1<<< threads, CUDA_NUM_THREADS >>>(num2, gradout, out_idx, channel, gradin);
    threads = (num1 + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    feature_backward2<<< threads, CUDA_NUM_THREADS >>>(num1, idx, channel, gradin);

}

void get_index_cuda(at::Tensor coords, at::Tensor order, at::Tensor query, at::Tensor index){
    int num1 = coords.size(1);
    int channel1 = coords.size(0);

    int num2 = query.size(0);
    int channel2 = query.size(1);
	
    if(channel1!=channel2 && channel1!=4){
        printf("%d %d %d %d\n", channel1, channel2, num1, num2);
        printf("error in coords shape!\n");
        exit(0);
    }
    int *que = query.data<int>();
    int *locs = coords.data<int>();
    int *src_order = order.data<int>();
    int *dst_idx = index.data<int>();

    thrust::device_ptr<int> dev_ptr(locs);
    thrust::device_vector<int> x(dev_ptr, dev_ptr + num1);
    thrust::device_vector<int> y(dev_ptr+num1, dev_ptr +2*num1);
    thrust::device_vector<int> z(dev_ptr+2*num1, dev_ptr+3*num1);
    thrust::device_vector<int> batch(dev_ptr+3*num1, dev_ptr+4*num1);

    thrust::device_ptr<int> dev_idx(src_order);
    thrust::device_vector<int> idx(dev_idx, dev_idx + num1);

    sort(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), batch.begin(), idx.begin())), thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end(), z.end(), batch.end(), idx.end())), less());
    const int* ptr_x = thrust::raw_pointer_cast(&x[0]);
    const int* ptr_y = thrust::raw_pointer_cast(&y[0]);
    const int* ptr_z = thrust::raw_pointer_cast(&z[0]);
    const int* ptr_batch = thrust::raw_pointer_cast(&batch[0]);
    const int* ptr_idx = thrust::raw_pointer_cast(&idx[0]);

    int threads = (num2 + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    search<<< threads, CUDA_NUM_THREADS >>>(num2, num1, ptr_x, ptr_y, ptr_z, ptr_batch, ptr_idx, que, dst_idx);

    x.clear();
    thrust::device_vector<int>().swap(x);
    y.clear();
    thrust::device_vector<int>().swap(y);
    z.clear();
    thrust::device_vector<int>().swap(z);
    batch.clear();
    thrust::device_vector<int>().swap(batch);
    idx.clear();
    thrust::device_vector<int>().swap(idx);
}
__global__ void knn_kernel(const int n, const int m, const int k, const int start, const float * known, const float* unknown, float * dist2, int * idx) {
    // unknown: (N, 3)
    // known: (M, 3)
    // output:
    // dist2: (N, k)
    // idx: (N, k)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= n) return;

    unknown +=  pt_idx * 3;
//   known += bs_idx * m * 3;
    dist2 += pt_idx * k;
    idx += pt_idx * k;

    float ux = unknown[0];
    float uy = unknown[1];
    float uz = unknown[2];
//    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
//    int besti1 = 0, besti2 = 0, besti3 = 0;
    for(int j = 0; j < k; j++)
        dist2[j]  = 1e10;
    for (int i = 0; i < m; ++i) {
        float x = known[i * 3 + 0];
        float y = known[i * 3 + 1];
        float z = known[i * 3 + 2];
        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
        for(int j = 0; j < k; j++){
            if(d < dist2[j]){
//                memcpy(dist2+j+1, dist2+j, (k-j-1)*sizeof(float));
//                memcpy(idx+j+1, idx+j, (k-j-1)*sizeof(int));
#if 1
                for(int l=k-1;l>j;l--){
                    dist2[l]=dist2[l-1];
                    idx[l] = idx[l-1];
                }
                dist2[j] = d;
                idx[j]=i+start;
                break;
#else
                if(j==k-1){
                    dist2[j]=d;
                    idx[j]=i+start;
                    break;
                }
                else{
                    cudaMemcpyAsync(dist2+j+1, dist2+j, (k-j-1)*sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaMemcpyAsync(idx+j+1, idx+j, (k-j-1)*sizeof(float), cudaMemcpyDeviceToDevice);
//                   memcpy(dist2+j+1, dist2+j, (k-j-1)*sizeof(float));
//                    memcpy(idx+j+1, idx+j, (k-j-1)*sizeof(int));
                    dist2[j] = d;
                    idx[j]=i+start;
                    break;
                }
#endif
            }
        }
    }
}
__global__ void knn_kernel2(const int n, const int k, const int *batches, const int *end, const float * known, const float* unknown, float * dist2, int * idx) {
    // unknown: (N, 3)
    // known: (M, 3)
    // output:
    // dist2: (N, k)
    // idx: (N, k)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= n) return;

    unknown +=  pt_idx * 3;
//   known += bs_idx * m * 3;
    dist2 += pt_idx * k;
    idx += pt_idx * k;

    float ux = unknown[0];
    float uy = unknown[1];
    float uz = unknown[2];
//    double best1 = 1e40, best2 = 1e40, best3 = 1e40;
//    int besti1 = 0, besti2 = 0, besti3 = 0;
    for(int j = 0; j < k; j++)
        dist2[j]  = 1e10;
    int cur_batch = batches[pt_idx];
    int start = 0;
    int stop = end[cur_batch];
    if(cur_batch>0) start = end[cur_batch-1];
    
    for (int i = start; i < stop; i++) {
        float x = known[i * 3 + 0];
        float y = known[i * 3 + 1];
        float z = known[i * 3 + 2];
        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
        for(int j = 0; j < k; j++){
            if(d < dist2[j]){
                for(int l=k-1;l>j;l--){
                    dist2[l]=dist2[l-1];
                    idx[l] = idx[l-1];
                }
                dist2[j] = d;
                idx[j]=i;
                break;

            }
        }
    }
}
__global__ void locate_kernel(const int n, const int length, const int*batch, int* locs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    int left = 0;
    int right = length-1;
    int cur = (left+right)/2;
    while(left <= right){

        if(index<batch[cur])
            right = cur-1;
        if(index>=batch[cur])
            left = cur +1;
        cur = (left + right)/2;
    }
    locs[index] = left;
 //   printf("%d %d\n", index, locs[index]);

}
/*
int locate(const int *batch, const int length, const int index) {

    int left = 0;
    int right = length-1;
    int cur = (left+right)/2;
    while(left <= right){

        if(index<batch[cur])
            right = cur-1;
        if(index>=batch[cur])
            left = cur +1;
        cur = (left + right)/2;
    }
    return left;
 //   printf("%d %d\n", index, locs[index]);

}
*/
 __device__ void inv_index(const int n, const int k, const int * idx, int * inv_idx, int* end) {
     
     for(int i=0;i<n*k;i++){
         int j = idx[i];
         end[j+1] ++;
     }
     for(int i=0;i<n*k;i++)
         end[i+1] += end[i];
     for(int i=0;i<n*k;i++){
         int j = idx[i];
         inv_idx[end[j]] = i/k;
         end[j]++;
     }
     

}
void knn_cuda(at::Tensor known, at::Tensor unknown, at::Tensor batch,  at::Tensor dist, at::Tensor idx, const int k, const int batchsize) {
    // unknown: (B, N, 3)
    // known: (B, M, 3)
    // output: 
    //      dist2: (B, N, 3)
    //      idx: (B, N, 3)
//    int clc = clock();
    int N = unknown.size(0);
    int M = known.size(0);
//    int batchsize = nums.size(0)
    const float * ref = known.data<float>();
    const float *que = unknown.data<float>();
    const int *batch_idx = batch.data<int>();
//    int * end = nums.data<int>();
    float * dist2 = dist.data<float>();
    int * index = idx.data<int>();
    cudaError_t err;
    int *end;
    cudaMalloc((void **)&end, batchsize * sizeof(int));
    int threads = (batchsize + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
//    printf("%d start ...\n", batchsize);
    locate_kernel<<<threads, CUDA_NUM_THREADS >>>(batchsize, N, batch_idx, end);
    threads = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    knn_kernel2<<<threads, CUDA_NUM_THREADS>>>(N, k, batch_idx, end, ref, que, dist2, index);
//     printf("locate done ...\n");
//    printf("%dms\n", (clock()-clc) * 1000 / CLOCKS_PER_SEC);
#if 0
    int * h_end = (int*)malloc(batchsize * sizeof(int));
    cudaMemcpy ( h_end, end, sizeof(int)*batchsize, cudaMemcpyDeviceToHost);
//    for(int i=0;i<batchsize;i++) printf("%d\n", h_end[i]);
//    printf("%dms\n", (clock()-clc) * 1000 / CLOCKS_PER_SEC);
 //   dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
 //   dim3 threads(THREADS_PER_BLOCK);
    for(int i =0;i<batchsize; i++){
//        N = h_end[i];
//        M = h_end[i];
        int start =0;
        if(i>0) start=h_end[i-1];
        N = h_end[i]-start;
        M = N;
        threads = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
        knn_kernel<<<threads, CUDA_NUM_THREADS>>>(N, M, k, start, ref+start*3, que+start*3, dist2+start*k, index+start*k);
    }
    free(h_end);
    h_end = NULL;
#endif
    cudaFree(end);
    end = NULL;

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
//    printf("%dms\n", (clock()-clc) * 1000 / CLOCKS_PER_SEC);
    
}

/* 
#ifdef __cplusplus
    }
#endif
*/
