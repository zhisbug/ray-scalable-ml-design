import argparse
import numpy as np
import time

import pytorch_benchmarks
import ray

parser = argparse.ArgumentParser(description="PyTorch microbenchmarks")
parser.add_argument('--test_name', type=str, default='auto', required=False,
                    help='Name of the test (broadcast, reduce, allreduce, gather, allgather)')
parser.add_argument('-n', '--world-size', type=int, required=False,
                    help='Size of the collective processing group')
parser.add_argument('-s', '--object-size', type=int, required=False,
                    help='The size of the object')
parser.add_argument('-b', '--backend', type=str, default='nccl', required=False,
                    help='The communication backend to use (nccl/gloo)')
args = parser.parse_args()


def test_with_mean_std(repeat_times,
                       test_name,
                       world_size,
                       object_size,
                       backend='nccl'):
    results = []
    for i in range(repeat_times):
        print('Test case {}......'.format(i))
        test_case = pytorch_benchmarks.__dict__[test_name]
        duration = test_case(world_size, object_size, backend)
        results.append(duration)
        time.sleep(5)
    return np.mean(results), np.std(results)


if __name__ == "__main__":
    ray.init(num_cpus=16, num_gpus=2)
    test_name = 'pytorch_' + args.test_name
    assert test_name in pytorch_benchmarks.__dict__ or args.test_name == 'auto'
    if args.test_name != 'auto':
        assert args.world_size is not None and args.object_size is not None
        mean, std = test_with_mean_std(5, test_name, args.world_size, args.object_size)
        print(f"{args.test_name},{args.world_size},{args.object_size},{mean},{std}")
    else:
        assert args.world_size is None and args.object_size is None
        backends = ['nccl', 'gloo']
        for backend in backends:
            print("==== Testing backend {} ====".format(backend))
            write_to = 'pytorch-microbenchmark-' + backend + '.csv'
            with open(write_to, "w") as f:
                if backend == 'nccl':
                    algorithms = ['pytorch_broadcast', 'pytorch_reduce', 'pytorch_allreduce', 'pytorch_allgather']
                elif backend == 'gloo':
                    algorithms = ['pytorch_broadcast', 'pytorch_gather', 'pytorch_reduce', 'pytorch_allreduce',
                                  'pytorch_allgather', 'pytorch_sendrecv']
                else:
                    raise ValueError('Cannot recognize the backend: {}'.format(args.backend))
                world_sizes = [2]
                object_sizes = [2 ** 10, 2 ** 15, 2 ** 20, 2 ** 25, 2 ** 30]
                for algorithm in algorithms:
                    for world_size in world_sizes:
                        for object_size in object_sizes:
                            mean, std = test_with_mean_std(5, algorithm, world_size, object_size, backend=backend)
                            print(f"{backend}, {algorithm}, {world_size}, {object_size}, {mean}, {std}")
                            f.write(f"{algorithm},{world_size},{object_size},{mean},{std}\n")
