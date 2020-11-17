import time

import os
import ray
import numpy as np
import cupy as cp
import cupy.cuda.nccl as nccl
import torch


# @ray.remote(num_gpus=1, resources={'machine': 1})
@ray.remote(num_gpus=1)
class BenchmarkWorker:
    def __init__(self,
                 world_size,
                 world_rank,
                 object_size):
        self.world_size = world_size
        self.world_rank = world_rank
        self.object_size = object_size

        self.comm = None
        self.uid = None

        self.send = None
        self.recv = None

    def gen_unique_id(self):
        if self.world_rank != 0:
            raise ValueError('should not reach here...')
        uid = nccl.get_unique_id()
        self.uid = uid
        return uid

    def setup(self, uid):
        # cp.cuda.Stream.null.synchronize()
        self.comm = nccl.NcclCommunicator(
            self.world_size, uid, self.world_rank)

    def put_object(self, tensor_type='cupy'):
        cp.cuda.Stream.null.synchronize()
        if tensor_type == 'cupy':
            self.send = cp.ones((round(self.object_size / 4),), dtype=cp.float32)
            self.recv = cp.zeros((round(self.object_size / 4),), dtype=cp.float32)
        elif tensor_type == 'pytorch':
            self.send = torch.cuda.FloatTensor(self.object_size // 4).fill_(1.0)
            self.recv = torch.cuda.FloatTensor(self.object_size // 4).fill_(0.0)
        else:
            raise ValueError('unrecognized tensor backend')
        print(self.world_rank, self.send)
        cp.cuda.Stream.null.synchronize()
        return True

    def allreduce(self):
        if isinstance(self.send, cp.core.core.ndarray):
            send_ptr = self.send.data.ptr
        elif isinstance(self.send, torch.Tensor):
            send_ptr = self.send.data_ptr()
        else:
            raise ValueError('unrecognized tensor backend')

        if isinstance(self.recv, cp.core.core.ndarray):
            recv_ptr = self.recv.data.ptr
        elif isinstance(self.recv, torch.Tensor):
            recv_ptr = self.recv.data_ptr()
        else:
            raise ValueError('unrecognized tensor backend')

        start = time.time()
        self.comm.allReduce(send_ptr,
                            recv_ptr,
                            round(self.object_size / 4),
                            cp.cuda.nccl.NCCL_FLOAT32,
                            0,
                            cp.cuda.Stream.null.ptr)

        cp.cuda.Stream.null.synchronize()
        duration = time.time() - start
        print(self.world_rank, self.recv)
        return duration

    def destroy(self):
        self.comm.destroy()
        return True


class BenchmarkWorkerPool:
    def __init__(self,
                 world_size,
                 object_size):
        self.actors = []
        leading_actor = BenchmarkWorker.remote(world_size, 0, object_size)
        uid = ray.get(leading_actor.gen_unique_id.remote())
        self.actors.append(leading_actor)

        for world_rank in range(1, world_size):
            actor = BenchmarkWorker.remote(world_size, world_rank, object_size)
            self.actors.append(actor)
        ray.wait([actor.setup.remote(uid) for actor in self.actors])

    def put_objects(self):
        rets = []
        for i, w in enumerate(self.actors):
            if i % 2 == 0:
                rets.append(w.put_object.remote(tensor_type='cupy'))
            else:
                rets.append(w.put_object.remote(tensor_type='pytorch'))
        ray.wait(rets, num_returns=len(rets), timeout=None)
        return True

    def release_actors(self):
        ray.wait([w.destroy.remote() for w in self.actors])
        ray.wait([w.__ray_terminate__.remote() for w in self.actors])
        for w in self.actors:
            ray.kill(w)

    def __getitem__(self, item):
        return self.actors[item]

    def __len__(self):
        return len(self.actors)

def cupy_pytorch_allreduce(world_size, object_size):
    workers = BenchmarkWorkerPool(world_size, object_size)
    workers.put_objects()
    durations = ray.get([w.allreduce.remote() for w in workers])
    workers.release_actors()
    return max(durations)

if __name__ == "__main__":
    ray.init(address='auto')

    ## Hao: This prototype proves that we can use cupy.nccl to reduce pytorch and cupy tensors together
    ## Hence interoperatorability between cupy and torch.cuda.Tensor, or numpy and torch.Tensor
    world_size = 4
    object_size = 2 ** 10
    duration = cupy_pytorch_allreduce(world_size, object_size)
    print(duration)
    ray.shutdown()
