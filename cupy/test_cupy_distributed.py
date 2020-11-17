import time

import os
import ray
import numpy as np
import cupy as cp
import cupy.cuda.nccl as nccl

os.environ["NCCL_DEBUG"] = "INFO"

# @ray.remote(num_gpus=1, resources={'machine': 1})
@ray.remote(num_gpus=1)
class CupyBenchmarkWorker:
    def __init__(self,
                 world_size,
                 world_rank,
                 object_size):
        self.world_size = world_size
        self.world_rank = world_rank
        self.object_size = object_size

        self.comm = None
        self.uid = None

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

    def put_object(self):
        cp.cuda.Stream.null.synchronize()
        self.send = cp.ones((round(self.object_size / 4),), dtype=cp.float32)
        self.recv = cp.zeros((round(self.object_size / 4),), dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        return True

    def allreduce(self):
        start = time.time()
        self.comm.allReduce(self.send.data.ptr,
                            self.recv.data.ptr,
                            round(self.object_size / 4),
                            cp.cuda.nccl.NCCL_FLOAT32,
                            1,
                            cp.cuda.Stream.null.ptr)
        cp.cuda.Stream.null.synchronize()
        duration = time.time() - start
        return duration

    def destroy(self):
        self.comm.destroy()
        return True

class CupyBenchmarkWorkerPool:
    def __init__(self,
                 world_size,
                 object_size):
        self.actors = []
        leading_actor = CupyBenchmarkWorker.remote(world_size, 0, object_size)
        uid = ray.get(leading_actor.gen_unique_id.remote())
        self.actors.append(leading_actor)

        for world_rank in range(1, world_size):
            actor = CupyBenchmarkWorker.remote(world_size, world_rank, object_size)
            self.actors.append(actor)
        ray.wait([actor.setup.remote(uid) for actor in self.actors])

    def put_objects(self):
        rets = [w.put_object.remote() for w in self.actors]
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


def cupy_allreduce(world_size, object_size):
    workers = CupyBenchmarkWorkerPool(world_size, object_size)
    workers.put_objects()
    durations = ray.get([w.allreduce.remote() for w in workers])
    workers.release_actors()
    return max(durations)


if __name__ == "__main__":
    ray.init(address='auto')

    # Hao: several caveats:
    # (1) when I setup up this program on my Lambda cluster, it won't run; NCCL INFO compains "unable to connect",
    # unless I explicitly append NCCL_SOCKET_IFNAME=enp179s0f0 as a env before ray start, and enforce NCCL to use
    # this Ethernet device; This shouldn't happen as NCCL is supposed to find the right roundtrip Ethernet by itself.
    # The problem is either caused by (1) the lambda-cluster is weird, or (2) Ray controls Ethernet assignments?

    # (2) There might be some deadlocks when scheduling roundtrip send/recv using ray.get() or ray.wait()
    world_size = 4
    object_size = 2 ** 10
    duration = cupy_allreduce(world_size, object_size)
    print(duration)
    ray.shutdown()
