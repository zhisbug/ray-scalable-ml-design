import time
import numpy as np

import torch
import torch.distributed as dist

import ray


@ray.remote(resources={'machine:1'})
class PyTorchBenchmarkWorker:
    def __init__(self,
                 world_size,
                 world_rank,
                 object_size,
                 backend='nccl',
                 tensor_type='gpu'):
        self.world_size = world_size
        self.world_rank = world_rank
        self.object_size = object_size
        self.backend = backend
        self.tensor_type = tensor_type
        self.message = []

        self.setup()

    def setup(self):
        dist.init_process_group(backend=self.backend,
                                init_method="file:///users/hzhang2/projects/pytorch_benchmark",
                                world_size=self.world_size,
                                rank=self.world_rank)

    def put_object(self):
        self.message = self.create_tensor()
        return True

    def create_tensor(self):
        num_floats = self.object_size // 4
        if self.tensor_type == 'gpu':
            t = torch.cuda.FloatTensor(num_floats).fill_(1.0)
        else:
            t = torch.FloatTensor(num_floats).fill_(1.0)
        return t

    def broadcast(self, src_rank):
        start = time.time()
        torch.distributed.broadcast(self.message, src_rank)
        duration = time.time() - start
        return duration

    def reduce(self, dst_rank):
        start = time.time()
        torch.distributed.reduce(self.message, dst_rank)
        duration = time.time() - start
        return duration

    def allreduce(self):
        start = time.time()
        torch.distributed.all_reduce(self.message)
        duration = time.time() - start
        return duration

    def allgather(self):
        recv = [self.create_tensor() for _ in range(self.world_size)]
        start = time.time()
        torch.distributed.all_gather(recv, self.message)
        duration = time.time() - start
        return duration

    def gather(self, dst_rank):
        if self.rank == dst_rank:
            recv = [self.create_tensor() for _ in range(self.world_size)]
        else:
            recv = None
        start = time.time()
        torch.distributed.gather(self.message, recv, dst=dst_rank)
        duration = time.time() - start
        return duration


class PyTorchBenchmarkWorkerPool:
    def __init__(self, world_size, object_size):
        self.actors = []
        for world_rank in range(world_size):
            self.actors.append(
                PyTorchBenchmarkWorker.remote(world_size,
                                              world_rank,
                                              object_size)
            )

    def put_objects(self):
        rets = [w.put_object.remote() for w in self.actors]
        ray.wait(rets, num_returns=len(rets), timeout=None)
        return True

    def __getitem__(self, item):
        return self.actors[item]

    def __del__(self):
        for w in self.actors:
            ray.kill(w)

    def __len__(self):
        return len(self.actors)


def pytorch_broadcast(world_size, object_size):
    workers = PyTorchBenchmarkWorkerPool(world_size, object_size)
    src_rank = 0
    object_id = workers[src_rank].put_object.remote()
    ray.wait([object_id], num_returns=1, timeout=None)
    durations = ray.get([w.broadcast.remote(src_rank) for w in workers])
    return max(durations)

def pytorch_reduce(world_size, object_size):
    workers = PyTorchBenchmarkWorkerPool(world_size, object_size)
    dst_rank = 0
    workers.put_objects()
    durations = ray.get([w.reduce(dst_rank) for w in workers])
    return max(durations)

def pytorch_gather(world_size, object_size):
    workers = PyTorchBenchmarkWorkerPool(world_size, object_size)
    dst_rank = 0
    workers.put_objects()
    durations = ray.get([w.gather(dst_rank) for w in workers])
    return max(durations)

def pytorch_allreduce(world_size, object_size):
    workers = PyTorchBenchmarkWorkerPool(world_size, object_size)
    workers.put_objects()
    durations = ray.get([w.allreduce() for w in workers])
    return max(durations)

def pytorch_allgather(world_size, object_size):
    workers = PyTorchBenchmarkWorkerPool(world_size, object_size)
    workers.put_objects()
    durations = ray.get([w.allgather() for w in workers])
    return max(durations)