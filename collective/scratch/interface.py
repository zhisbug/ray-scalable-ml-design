import ray


# interface 1


@ray.remote(num_gpus=1)
class MyActor:
    def __init__(self):
        pass
    def do_compute(self):
        ray.collective.allreduce(tensor)


@ray.remote(num_gpus=1)
def MyTask(x):
    pass



actors = [MyActor.remote() for i in range(4)]
options = {'world_size': 4, 'group_name': 'my_group', rank = [2,3,1,0]}
ray.collective.create_collective_group(options, actors)


def create_collective_group(options, actors):
    # find the actor with rank = 0, actor0
    id = actor0.init_collectve_group(*args)
    for actor in actors:
        if rank != 0
            ncclUniqueID = ray.get(id)







































obj_ref = ray.put(obj)



def init_collective_group(name='default_group'):
    if rank == 0:
        unique_obj = hash(name=group_name)
        obj_ref = ray.put(unique_obj)
    else:
        unique_obj = ray.get(key=group_name)


actors





ray.collective.init_collective_group(init_method='tcp://192.168.0.1:128',
                                     *args)
ray.collective.allreduce(tensor)






## PyTorch program: main.py
torch.distributed.init_process_group(init_method='tcp://192.168.0.1:128',
                                     world_size=4)
# tensors...
torch.distributed.allreduce(tensors)
