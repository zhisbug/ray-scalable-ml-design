import string
import time
import numpy as np
import os
import random
import socket

import torch
import torch.distributed as dist

import ray

#pytorch_cluster_file = "tcp://10.117.1.2:34567"
pytorch_cluster_file = "file:///users/hzhang2/projects/ray-scalable-ml-design/pytorch/microbenchmark/primitives/pytorch/tmp/"


#@ray.remote(resources={'machine': 1})
@ray.remote(num_gpus=1)
class PyTorchBenchmarkWorker:
    def __init__(self,
                 world_size,
                 world_rank,
                 object_size,
                 exp_name,
                 backend='nccl'):
        self.world_size = world_size
        self.world_rank = world_rank
        self.object_size = object_size
        self.backend = backend
        self.cluster_file = pytorch_cluster_file + exp_name
        #self.cluster_file = pytorch_cluster_file
        self.message = None

    def gen_init_method(self):
        if self.world_rank != 0:
            raise ValueError('should not reach here...')
        my_ip = socket.gethostbyname(socket.gethostname())
        method = 'tcp://' + my_ip + ':34567'
        return method
    
    def setup(self, init_method):
        dist.init_process_group(backend=self.backend,
                                #init_method=self.cluster_file,
                                init_method=init_method,
                                world_size=self.world_size,
                                rank=self.world_rank)
        return True

    def put_object(self):
        self.message = self.create_tensor()
        return True

    def create_tensor(self, value=1.0):
        num_floats = self.object_size // 4
        if self.backend == 'nccl':
            t = torch.cuda.FloatTensor(num_floats).fill_(self.world_rank + 0.11111111121)
        else:
            t = torch.FloatTensor(num_floats).fill_(self.world_rank + 0.11111111121)
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
        if self.world_rank == dst_rank:
            recv = [self.create_tensor() for _ in range(self.world_size)]
        else:
            recv = None
        start = time.time()
        torch.distributed.gather(self.message, recv, dst=dst_rank)
        duration = time.time() - start
        return duration

    def send(self, dst_rank):
        start = time.time()
        torch.distributed.send(self.message, dst_rank)
        duration = time.time() - start
        return duration

    def recv(self, src_rank):
        recv_buffer = self.create_tensor()
        start = time.time()
        torch.distributed.recv(recv_buffer, src_rank)
        duration = time.time() - start
        return duration


class PyTorchBenchmarkWorkerPool:
    def __init__(self,
                 world_size,
                 object_size,
                 exp_name,
                 backend='nccl'):
        self.actors = []

        leading_actor = PyTorchBenchmarkWorker.remote(world_size, 0, object_size, exp_name, backend=backend)
        method = ray.get(leading_actor.gen_init_method.remote())
        self.actors.append(leading_actor)

        for world_rank in range(1, world_size):
            actor = PyTorchBenchmarkWorker.remote(world_size, world_rank, object_size, exp_name, backend=backend)
            self.actors.append(actor)
        ray.wait([actor.setup.remote(method) for actor in self.actors])

    def put_objects(self):
        rets = [w.put_object.remote() for w in self.actors]
        ray.wait(rets, num_returns=len(rets), timeout=None)
        return True

    def release_actors(self):
        ray.wait([w.__ray_terminate__.remote() for w in self.actors])
        for w in self.actors:
            ray.kill(w)

    def __getitem__(self, item):
        return self.actors[item]

    #def __del__(self):
    #    for w in self.actors:
    #        ray.kill(w)

    def __len__(self):
        return len(self.actors)


def pytorch_broadcast(world_size,
                      object_size,
                      backend,
                      exp_name):
    workers = PyTorchBenchmarkWorkerPool(world_size, object_size, exp_name, backend=backend)
    src_rank = 0
    workers.put_objects()
    object_id = workers[src_rank].put_object.remote()
    ray.wait([object_id], num_returns=1, timeout=None)
    durations = ray.get([w.broadcast.remote(src_rank) for w in workers])
    workers.release_actors()
    return max(durations)

def pytorch_reduce(world_size,
                   object_size,
                   backend,
                   exp_name):
    workers = PyTorchBenchmarkWorkerPool(world_size, object_size, exp_name, backend=backend)
    dst_rank = 0
    workers.put_objects()
    durations = ray.get([w.reduce.remote(dst_rank) for w in workers])
    workers.release_actors()
    return max(durations)


def pytorch_gather(world_size,
                   object_size,
                   backend,
                   exp_name):
    workers = PyTorchBenchmarkWorkerPool(world_size, object_size, exp_name, backend=backend)
    dst_rank = 0
    workers.put_objects()
    durations = ray.get([w.gather.remote(dst_rank) for w in workers])
    workers.release_actors()
    return max(durations)


def pytorch_allreduce(world_size,
                      object_size,
                      backend,
                      exp_name):
    workers = PyTorchBenchmarkWorkerPool(world_size, object_size, exp_name, backend=backend)
    workers.put_objects()
    durations = ray.get([w.allreduce.remote() for w in workers])
    workers.release_actors()
    return max(durations)


def pytorch_allgather(world_size,
                      object_size,
                      backend,
                      exp_name):
    workers = PyTorchBenchmarkWorkerPool(world_size, object_size, exp_name, backend=backend)
    workers.put_objects()
    durations = ray.get([w.allgather.remote() for w in workers])
    workers.release_actors()
    return max(durations)


def pytorch_sendrecv(world_size,
                     object_size,
                     backend,
                     exp_name):
    workers = PyTorchBenchmarkWorkerPool(world_size, object_size, exp_name, backend=backend)
    workers.put_objects()
    src_rank = 0
    dst_rank = 1
    durations = ray.get([workers[src_rank].send.remote(dst_rank), workers[dst_rank].recv.remote(src_rank)])
    workers.release_actors()
    return max(durations)

