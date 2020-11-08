import argparse
import time

import numpy as np
import ray

import ray_benchmarks

parser = argparse.ArgumentParser(description='Ray microbenchmarks')
parser.add_argument('--test_name', type=str, default='auto', required=False,
                    help='Name of the test (broadcast, reduce, allreduce, gather, allgather)')
parser.add_argument('-n', '--world-size', type=int, required=False,
                    help='Size of the collective processing group')
parser.add_argument('-s', '--object-size', type=int, required=False,
                    help='The size of the object')
parser.add_argument('-b', '--backend', type=str, default='gpu', required=False,
                    help='The communication backend to use (gpu/cpu)')
args = parser.parse_args()


def test_with_mean_std(repeat_times,
                       test_name,
                       world_size,
                       object_size,
                       backend='gpu'):
    results = []
    for i in range(repeat_times):
        print('Test case {}......'.format(i))
        test_case = ray_benchmarks.__dict__[test_name]
        duration = test_case(world_size, object_size, backend)
        results.append(duration)
        time.sleep(5)
    return np.mean(results), np.std(results)


if __name__ == "__main__":
    ray.init(num_cpus=4, num_gpus=2)
    test_name = 'ray_' + args.test_name
    assert test_name in ray_benchmarks.__dict__ or args.test_name == 'auto'
    if args.test_name != 'auto':
        assert args.world_size is not None and args.object_size is not None
        mean, std = test_with_mean_std(5, test_name, args.world_size, args.object_size, backend=args.backend)
        print(f"{args.test_name},{args.world_size},{args.object_size},{mean},{std}")
    else:
        assert args.world_size is None and args.object_size is None
        backends = ['gpu', 'cpu']
        for backend in backends:
            write_to = 'ray-microbenchmark-' + backend + '.csv'
            with open(write_to, "w") as f:
                algorithms = ['ray_broadcast', 'ray_gather', 'ray_reduce', 'ray_allreduce', 'ray_allgather', 'ray_sendrecv']
                # algorithms = ['ray_sendrecv']
                world_sizes = [2]
                object_sizes = [2 ** 10, 2 ** 15, 2 ** 20, 2 ** 25, 2 ** 30]
                for algorithm in algorithms:
                    for world_size in world_sizes:
                        for object_size in object_sizes:
                            mean, std = test_with_mean_std(5, algorithm, world_size, object_size, backend=backend)
                            print(f"{algorithm}, {world_size}, {object_size}, {mean}, {std}")
                            f.write(f"{algorithm},{world_size},{object_size},{mean},{std}\n")
