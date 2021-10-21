import torch
from torch.autograd import Function
from ..build.lib import fusion
from torch.autograd import Variable

class KnnFunction(Function):
    @staticmethod
    def forward(ctx, batches, points, knn, batchsize):
        assert(batches.is_contiguous() == True and points.is_contiguous() == True)
        with torch.cuda.device_of(points):
            num, channel = points.size()
#            order = batches.int()
            index = points.new().resize_(num, knn).zero_().int()
            dist = points.new().resize_(num, knn).zero_().float()
#            nums = points.new().resize_(batchsize).zero_().int()
#            print('knn start ...')
#            for i in range(batchsize):
#                print(torch.sum(batches==i).item())

            fusion.knn_cuda(points, points, batches.int(), dist, index, knn, batchsize)
#        print(dist)
        index = index.long().contiguous()
            
        return index
    @staticmethod
    def backward(ctx, temp1, temp2):
        return None, None, None, None

class QueryFunction(Function):
    @staticmethod
    def forward(ctx, coords, query):
        assert(coords.is_contiguous() == True and query.is_contiguous() == True)
        with torch.cuda.device_of(query):
            num, channel = coords.size()

            order = torch.arange(num, device=query.device).int()
#            coords = coords.to(query.device)
            coords = coords.transpose(0,1).int().contiguous()
            index = query.new().resize_(query.size(0)).zero_().int()
            query = query.int()
            fusion.get_index_cuda(coords, order, query, index)
        index = index.contiguous()
        temp = index<0
        if (temp.sum()>0):
            print("{} invalid points not found.".format(temp.sum().item()))
 #           print(temp.sum())
            
        return index
    @staticmethod
    def backward(ctx, gradOutput):
        return None, None

class MergeFunction(Function):
    @staticmethod
    def forward(ctx, coords, features, data=False):
        ctx.data = data
        assert(coords.is_contiguous() == True and features.is_contiguous() == True)
        with torch.cuda.device_of(features):
            coords = coords.to(features.device)
            num, channel = coords.size()

            order = torch.arange(num, device=features.device).int()
            coords_trans = coords.transpose(0,1).int().contiguous()

            new_feas = features.clone().float().contiguous()
            fusion.merge_cuda_forward(coords_trans, new_feas, order)
            
            idx = order[order>=0].contiguous()
            coords = coords[idx.long(), :]
            new_feas = new_feas[idx.long(), :]
        coords = coords.contiguous()
        new_feas = new_feas.contiguous()

        ctx.save_for_backward(features, order, idx)
        return coords, new_feas
    @staticmethod
    def backward(ctx, temp, gradOutput):
#        print(gradOutput)
        if ctx.data:
            return None, None, None
        features, order, idx = ctx.saved_tensors
        gradOutput = gradOutput.float()
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            grad_feas = gradOutput.new().resize_(features.size()).zero_().float()
            fusion.merge_cuda_backward(gradOutput, idx, order, features.float(), grad_feas)
        grad_feas = grad_feas.contiguous()
#        print(grad_feas)
        return None, grad_feas, None

