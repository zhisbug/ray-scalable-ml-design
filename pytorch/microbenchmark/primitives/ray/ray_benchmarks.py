import time
import numpy as np
import random
import ray

import torch

# @ray.remote(resources={'machine': 1})
@ray.remote(num_gpus=1)
class RayBenchmarkWorker:
    def __init__(self,
                 world_size,
                 world_rank,
                 object_size,
                 backend='gpu'):
        self.world_size = world_size
        self.world_rank = world_rank
        self.object_size = object_size
        self.backend = backend

    def barrier(self):
        # We sleep for a while to make sure all later
        # function calls are already queued in the execution queue.
        # This will make timing more precise.
        time.sleep(1)
        # barrier(self.notification_address, self.notification_port, self.world_size)

    def create_tensor(self, value=1.0):
        num_floats = self.object_size // 4
        if self.backend == 'gpu':
            t = torch.cuda.FloatTensor(num_floats).fill_(value)
        elif self.backend == 'cpu':
            t = torch.FloatTensor(num_floats).fill_(value)
        else:
            raise ValueError('Unrecognized backend: {}'.format(self.backend))
        return t

    def put_object(self, value=1.0):
        t = self.create_tensor(value)
        return ray.put(t)

    def get_objects(self, object_ids):
        object_ids = ray.get(object_ids)

        # timing
        start = time.time()
        _ = ray.get(object_ids)
        duration = time.time() - start
        return duration

    def get_objects_with_creation_time(self, object_ids):
        start = time.time()
        object_ids = ray.get(object_ids)
        _ = ray.get(object_ids)
        duration = time.time() - start
        return duration

    @ray.method(num_returns=2)
    def reduce_objects(self, object_ids):
        object_ids = ray.get(object_ids)

        reduce_result = self.create_tensor(0.0)
        start = time.time()
        for object_id in object_ids:
            array = ray.get(object_id)
            reduce_result += array
        duration = time.time() - start
        result_id = ray.put(reduce_result)
        return result_id, duration


class RayBenchmarkActorPool:
    def __init__(self,
                 world_size,
                 object_size,
                 backend='gpu'):
        self.actors = []
        for world_rank in range(world_size):
            self.actors.append(
                RayBenchmarkWorker.remote(world_size,
                                          world_rank,
                                          object_size,
                                          backend=backend))

    def barrier(self):
        return [w.barrier.remote() for w in self.actors]

    def prepare_objects(self):
        object_ids = [w.put_object.remote() for w in self.actors]
        # wait until we put all objects
        ray.wait(object_ids, num_returns=len(object_ids), timeout=None)
        return object_ids

    def __getitem__(self, k):
        return self.actors[k]

    def __del__(self):
        for w in self.actors:
            ray.kill(w)

    def __len__(self):
        return len(self.actors)


def ray_broadcast(world_size, object_size, backend):
    actor_pool = RayBenchmarkActorPool(world_size, object_size, backend)
    # Caution (Hao):
    # Ideally, I don't have to put objects on all actors for `broadcast`, but unfortunately if I don't do so,
    # the first time when a cuda tensor is launched on an actors, a big overhead will be incurred,
    # and be count in this benchmarking code
    object_ids = actor_pool.prepare_objects()
    # object_id = actor_pool[0].put_object.remote()
    # # wait until we have put that object
    # ray.wait([object_id], num_returns=1, timeout=None)
    durations = ray.get([w.get_objects.remote([object_ids[0]]) for w in actor_pool.actors])
    del actor_pool
    return max(durations)


def ray_reduce(world_size, object_size, backend):
    actor_pool = RayBenchmarkActorPool(world_size, object_size, backend)
    object_ids = actor_pool.prepare_objects()
    reduction_id, duration_id = actor_pool[0].reduce_objects.remote(object_ids)
    duration = ray.get(duration_id)
    del actor_pool
    return duration


# TODO: the timing is not precise
def ray_allreduce(world_size, object_size, backend):
    # Apparently this is not a typical implementation of allreduce, but the communication load is same.
    # Also not the implementing ring allreduce using Ray's object store might be even slower than this one, cuz
    # likely more ray tasks will be launched.
    actor_pool = RayBenchmarkActorPool(world_size, object_size, backend)
    object_ids = actor_pool.prepare_objects()
    # actor_pool.barrier()
    reduction_id, duration_id = actor_pool[0].reduce_objects.remote(object_ids)
    results = [duration_id]
    for i in range(1, len(actor_pool)):
        results.append(actor_pool[i].get_objects_with_creation_time.remote([reduction_id]))
    durations = ray.get(results)
    del actor_pool
    return max(durations)


def ray_gather(world_size, object_size, backend):
    actor_pool = RayBenchmarkActorPool(world_size, object_size, backend)
    object_ids = actor_pool.prepare_objects()
    duration = ray.get(actor_pool[0].get_objects.remote(object_ids))
    del actor_pool
    return duration


# TODO: the timing is not precise
def ray_allgather(world_size, object_size, backend):
    actor_pool = RayBenchmarkActorPool(world_size, object_size, backend)
    object_ids = actor_pool.prepare_objects()
    # actor_pool.barrier()
    results = [w.get_objects.remote(object_ids) for w in actor_pool.actors]
    durations = ray.get(results)
    del actor_pool
    return max(durations)


def ray_sendrecv(world_size, object_size, backend):
    actor_pool = RayBenchmarkActorPool(world_size, object_size, backend)
    object_ids = actor_pool.prepare_objects()
    src_rank, dst_rank = random.sample(range(world_size), 2)
    duration = ray.get(actor_pool[dst_rank].get_objects.remote([object_ids[src_rank]]))
    del actor_pool
    return duration


