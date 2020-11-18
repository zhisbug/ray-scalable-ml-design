# RFC-202011-collective-in-ray

| Status        | Proposed      |
:-------------- |:---------------------------------------------------- |
| **Author(s)** | Hao Zhang, Lianmin Zheng |
| **Sponsor**   | Ion Stoica                 |
| **Updated**   | YYYY-MM-DD                                           |


## Objective

This RFC proposes to add a set of *Python-based* collective and send/recv communication APIs into Ray, for *both CPU and GPU* tensors, based on several established collective communication backends, including [NCCL](https://github.com/NVIDIA/nccl), (MPI)[https://github.com/mpi4py/mpi4py], as well as the original Ray object store (hence gRPC backend). 

This set of APIs will enable Ray users to conveniently perform collective communication of several types of Python tensors in Ray actors and tasks, for their performance-critical distributed applications, such as (GPU-based) distributed machine learning, or other HPC applications. 

This set of APIs also serve as a shared infrastructure for the ongoing two Ray-derived projects: [NumS](https://github.com/nums-project/nums) and [RayML](https://github.com/zhisbug/ray-scalable-ml-design/).

### Non-goals

- For now, these APIs aim to only support **python Tensors** (e.g. Numpy, Cupy, PyTorch tensors), but **not** arbitrary Ray Objects or ObjectRefs.
 - These APIs are not obligated to perform communication through Ray object store. Instead, an option is provided in the APIs to allow choosing Ray object store (hence gPRC) as the backend.

## Motivation

We want to flesh out several value propositions that drive this project, with a few concrete use cases.

### Improve Programming Convenience 

- **General collective communication APIs in Ray**: Collective communication (CC) pattern (e.g.`allreduce`, `allgather`) naturally emerge in many distributed computing applications in Ray. While in practice Ray users can compose CC functions using Ray's generic APIs as a series of RPC calls, directly providing such CC APIs would add significant convenience.

- **Distributed NCCL APIs in Python**: NCCL relies on MPI or other socket tools to broadcast the `ncclUniqueId` to distributed processes, to run on distributed environments. This requirement makes it hard to implement distributed NCCL-based applications in Python. Existing libraries that use NCCL mostly reply on MPI or self-implemented distributed store to setup cross-process coordination in distributed environments. Examples include Horovod, which depends on `mpirun [args]` to invoke distributed processes; Or PyTorch, which implements a [Distributed Key-value Store](https://github.com/pytorch/pytorch/blob/master/torch/lib/c10d/Store.hpp) to enable NCCL to run in distributed environments. 
Also of note that NCCL itself does not have a high-level Python binding. The only way users can use NCCL for their distributed Python applications are: (1) writing C++ instead of Python; (2) constraint their code within PyTorch/TensorFlow/Horovod which natively support NCCL; (3) using Cupy plus some other distributed stitching tools like MPI(which involves low-level CUDA code such as stream management, etc., and is error-prone). 
Essentially, there is a gap to write distributed Python applications using NCCL. Ray naturally fills this gap -- Ray has a distributed store and high-level sticking APIs. Putting NCCL into Ray and wraps it into high-level Python APIs makes it easy to use NCCL, which in turn benefits Ray.

- **Interoperatorability between Python tensor types**: Nowadays ML frameworks or computing libraries expose some sorts of collective communication APIs for their framework-native tensor types, such as `torch.distributed.allreduce(x: torch.Tensor)` in PyTorch, or `tf.collective_ops` in TensorFlow. The collective APIs built on top of Ray can slightly relax this constraint on tensor types and allow collective communication of tensors from different computing libraries, such as a Cupy tensor and a PyTorch tensor, as long as the tensor classes expose a `data_ptr()` method which points to the address of the tensor (which is true for Numpy, Cupy, PyTorch).

- **Auxiliary applications**: we observe there is an emerging need for Ray to offer some native high-level application interfaces, such as a *sharded parameter server*, or *collective allreduce strategy*, similar to `tf.distributed.CollectiveAllReduceStrategy`. The targeted users of these interfaces are those who conduct distributed ML loads either without using TensorFlow or PyTorch, or mixing the usage of TensorFlow or PyTorch. For example, [Spacy](https://spacy.io/) has a community of users who build ML applications using their in-house framework [Thinc](https://thinc.ai/), which replies on Ray to provide distributed computing support. A set of high-performance collective APIs are the cornerstone of these applications.

### Improve Performance

- For CPU tensors, composing collective communication functions using Ray's object store is feasible, but the performance is suboptimal, compared to highly optimized, vender-specific CC libraries like MPI, NCCL, [GLOO](https://github.com/facebookincubator/gloo), [OneCCL](https://github.com/oneapi-src/oneCCL). See a detailed performance benchmark in this [report](https://github.com/zhisbug/ray-scalable-ml-design/tree/main/pytorch/microbenchmark/primitives/results). Bring these CC libraries into Ray can improve the perfermance of such communication patterns.

- Ray's object store has limited awareness of GPUs. As a consequence, using Ray's generic APIs to move GPU tensors (collectives or point-to-point) faces severe performance degeneration. See [Hao's benchmarking report](https://github.com/zhisbug/ray-scalable-ml-design/tree/main/pytorch/microbenchmark/primitives/results) and [Lianmin's benchmarking report](https://github.com/merrymercy/gpu-comm/tree/master/round-trip-benchmark) for details. While this RFC does NOT aim to add GPU awareness to Ray's object store, it can address the performance issues for collective and P2P communication of GPU tensors.

- Some Ray users (e.g. Spacy) rely on Ray as a primary tool to provide distributed communication support. They face some performance challenges caused by collective communication. These APIs will address their performance issues.

## User Benefit

With this project, Ray users can access a set of readily available collective and send/recv APIs in pure Python, with MPI/NCCL/Ray object store as optional backends, supporting both CPU and GPU tensors in Numpy, Cupy, or PyTorch.


## Design Proposal

### Architecture
An architecture diagram is shown below:

<p align="center"><img src="arch.png" width=600 /></p>

The intended functionalities of several key classes are briefly explained below: 

#### `Communicator`
CC-backend-specific implementations covering a few essential functionalities:
- Communicator creation, management, reuse, etc.
- Thread/CUDA stream creation, management, reuse, when needed
- tensor type detection, conversion
- Invocation of the third-party communication APIs
- Other backend-specific implementations such as type matching, stream synchronization, etc.

#### `CollectiveGroup`
A `CollectiveGroup` coordinates a set of processes (e.g. Ray actors or tasks) participating into collective communication. It accounts for managing the communicators of multiple processes.

The `CollectiveGroup` exposes a set of general collective primitive and group APIs for users to create a collective group and perform communication. It dispatches the execution of APIs to the specific collective implementations in backends.


### APIs

#### Rationale
The core problem to solve in designing APIs is to expose a minimal and least disrupted set of interfaces so that users can declare a set of Ray actors or tasks as a collective group, and assign each member therein with a few attributes required for collective communication. These attributes are:
- `collective_group`: the name of its participating group
- `world_size`: the number of collective participants 
- `rank`: the rank of this participant
- `backend`: the backend to use, NCCL, MPI, or Ray Object Store (gRPC)

Below I list a few proposals. **Note** the proposals are **very preliminary** and need extensive help and discussion among NumS, RayML and AnyScale developers to get into the right shape.

####  User APIs Proposal #1: mimicking torch.distributed
```python
# Example #1
import ray
import cupy

@ray.remote(num_gpus=1) 
def cupy_func(rank):
	send = cupy.ones((10,), dtype=cupy.float32)
	recv = cupy.ones((10,), dtype=cupy.float32)
	
	# This is a blocking call
	ray.collective.init_collective_group(collective_group='my_name',
					     backend='nccl',
					     world_size=4,
					     rank=rank)
	
	# This is a blocking call
	ray.collective.allreduce(send, recv)
	return recv

futures = [cupy_func.remote(i) for i in range(4)]
print(ray.get(futures))
```
```python
# Example #2
import ray
import torch
import cupy

@ray.remote
class CupyWorker:
	def __init__(self):
		self.send = cupy.ones((10,), dtype=cupy.float32)
		self.recv = cupy.zeros((10,), dtype=cupy.float32)
	
	def setup(self, rank):
		ray.init_collective_group(collective_group='my_name',
					  backend='nccl',
					  world_size=20,
					  rank=rank)
		return True
	
	def do_computation(self):
		ray.collective.allreduce(self.send, self.recv)
	
@ray.remote
class PytorchWorker:
	def __init__(self):
		self.send = torch.cuda.FloatTensor(10).fill_(1.0)
		self.recv = torch.cuda.FloatTensor(10).fill_(0.0)

	def setup_collective_group(self, name, backend, world_size, rank):
		ray.init_collective_group(collective_group='my_name',
					  backend='nccl',
					  world_size=20,
					  rank=rank)
		return True

	def do_computation(self):
		# Do some computation
		ray.collective.allreduce(self.send, self.recv)
		return self.secv
		

# Since the constructor of the workers have `init_collective_group`
actors = []
actors.extend([CupyWorker.remote() for i in range(10)])
actors.extend([PytorchWorker.remote() for i in range(10)])
ray.wait([a.setup.remote(rank) for rank, a in enumurate(actors)])

# Note: interoperatability between cupy and pytorch
# Similarly we can do that between numpy, cupy, pytorch, 
# or any tensor that expose a pointer
futures = [a.do_computation.remote(i) for a in actors]
print(ray.get(futures))
```

#### User APIs Proposal #2: Mimicking `ray.util.ActorPool`
The APIs above can be further simplified by defining some interfaces like `CollectivePool`, similar to the current `ray.util.ActorPool`.

```python
actors = [CupyWorkers.remote() for i in range(20)]

# This line will setup collective memberships among all actors 
# (i.e. implicitly generate a setup function as above, and call 
# it among all actor processes
pool = CollectivePool(actors, ranks, name='default')

# Do whatever that needs collective communication
futures = [a.do_computation.remote(i) for a in actors]
# ...
```
#### Collective Primitive Signatures
We will expose a set of collective APIs similar to `torch.distributed` and NCCL, which mostly follow MPI standards, under the namespace `ray.collective`. Some example signatures:
- `ray.collective.init_collective_group(group_name, world_size, rank, backend, ...)`
- `ray.collective.allreduce(tensor, reduce_op, ...)`
- `ray.collective.all_gather(tensor_list, tensor, ...)`
- `ray.collective.send(tensor, dst_rank, ...)`
- `ray.collective.recv(tensor, src_rank, ...)`

See [NCCL Collective](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#) and [torch.distributed](https://pytorch.org/docs/stable/distributed.html#) for more examples. 
A key difference worth mentioning here is that the `tensor` in Ray collective functions could be any of Numpy, PyTorch, or Cupy tensors, as long as their `dtype` matches.

#### (Optional) Lower-level APIs
`ray.collective` namespace dispatches different collective primitive calls to their backend-specific implementations. However, if a user wants to have fine-grained control of the `communicators` and communication backends, some lower-level APIs could be made available. Take the NCCL as an example:
- `comm = ray.collective.nccl.create_communication()`
- `ray.collective.nccl.allreduce(tensor, reduce_op, comm, ...)`
- `...`

Working with this layer of APIs needs to deal with the creation and destruction of `communicators`, and potentially manages the communication threads or CUDA streams when needed manually.

### Unsolved Problems

#### Deadlocks
In some of the prototypes, we occasionally observe deadlocks. An example is below:
```python
A = CupyWorker.remote()
B = CupyWorker.remote()
ray.wait([A.setup.remote(rank=0, ...), B.setup.remote(rank=1, ...)]
ray.get([A.send.remote(), B.recv.remote(), A.recv.remote(), B.send.remote()])
```
The above code tries to do two parallel send/recv between two workers A and B using NCCL APIs, and use `ray.get()` to trigger the round trip. It will hang.

#### GPU Stream Management
NCCL calls need to carefully synchronize GPU streams between workers to ensure correctness, and manage multiple asynchronous GPU streams on each worker to ensure performance. This could be realized using the python bindings of CUDA runtime APIs in Cupy. We haven't figured out the optimal way to allocate/manage streams, but a good reference is the `ProcessGroupNCCL.hpp` in PyTorch.

#### Multiple Concurrent CollectiveGroup 
In some cases (e.g. model parallelism + data parallelism) we might create multiple CollectiveGroups, in which some groups perform `allreduce` synchronization of gradients for data parallelism, while other groups perform `send/recv` P2P communications between two model partitions. This might require creating multiple `CollectiveGroups` within a  same pool of Ray actors or tasks, which in turn requires managing collective communicators, communication threads, and GPU streams. We need to figure out some details on this later during implementation.

### Alternative Design
#### Alternative #1: a C++ Architecture with Python Bindings

One alternative architecture is shown in the figure below. 

<p align="center"><img src="arch-alternative.png" width=600 /></p>

Several key differences between this design and the proposed one are:
- Import the NCCL (MPI, custom collective lib) into Ray in C++, compile together with Ray C++ core.
- Implement the communicator creation, management, and collective group creations, management all in C++. Implement a set of collective communication primitive APIs in C++. Then provide python APIs on top of them as Python bindings.

Its pros and cons are discussed below:
#### Pros
- Get rid of Cupy and MPI4py dependencies.
- Might observe slight performance improvement.
- In the future, it might be easier to generalize this set of CC APIs to work with Ray object store, and support collective communication between Ray objects or ObjectRefs (@Ion).
- It is easier to extrapolate to newer custom collective communication libraries, such as Intel oneCCL, because normally these libraries are implemented and released in C/C++, *without* Python bindings.

#### Cons
- For the NCCL part, we are essentially redoing a lot of engineering that Cupy/PyTorch have done, including: NCCL Python bindings, GPU memory management, some CUDA runtime API wrappers, definitions of CUDA tensor data structures, etc.
- For the MPI part, we are redoing a lot of engineering that Horovod has done.
- Apparently implementing things in C++ might significantly extend the development cycle of this project.


### Alternative #2: Using UCX/UCX-py as the communication backend
We have also considered using [UCX](https://github.com/openucx/ucx) and [UCX-py](https://github.com/rapidsai/ucx-py) for P2P communication (in particular, for GPU tensor), and then implementing collective communication APIs using its P2P API. We deny this option with the following discussion of pros and cons:

#### Cons
- According to some [benchmarking results](https://github.com/merrymercy/gpu-comm/tree/master/round-trip-benchmark) generated by Lianmin, UCX-py shows inferior performance compared to NCCL on GPU tensor collective communication. This gap is substantial.
- The UCX C++ backend is under development. Compared to NCCL, it is less mature -- we observe less cases in which UCX is adopted. In contrast, MPI/NCCL is the SoTA CPU/GPU collective communication library, esp. in distributed ML -- A major goal of this project is to build the communication backend infra for the two distributed ML projects: NumS and RayML.
- The UCX-py package is at a rather preliminary stage, and we find it not easy to use, e.g. building applications with its APIs heavily involves using Python3.8+ AsyncIO.

#### Pros
- UCX is supposed to offer a device-agnostic "one-size-fit-all" solution for cross-process communication. It aims to eventually provide some features to auto-detect the device topology and properties and auto-optimize communication algorithms. 
- In several scenarios where we want to communicate messages between CPU RAM and GPUs (cpu -> gpu send/recv), it might be advantageous.

### Other Considerations
We want to make it light-weight so we can develop most of the needed features in a fast pace, then refocus on the development of NumS and RayML core. Hence, we tend to favor design that can:
- Avoid redoing some of the engineering that existing projects have done
- If an implementation can be done either in C++ or Python, we lean toward Python, unless there is a substantial performance gap.


### Performance Implications
In summary, the performance of collective primitives of CPU tensors in Ray can improve 2x-10x (latency), matching MPI/GLOO performance. Collective communication of GPU tensors in Ray can improve 10x - 1000x, matching NCCL performance.
- See [micro benchmark#1](https://github.com/zhisbug/ray-scalable-ml-design/tree/main/pytorch/microbenchmark/primitives/results) and [microbenchmark#2](https://github.com/merrymercy/gpu-comm/tree/master/round-trip-benchmark) for the performance improvement for each collective and P2P communication primitives, on different cluster setups.
- See an [end-to-end benchmark](https://github.com/zhisbug/ray-scalable-ml-design/spacy) on training a custom Spacy NLP pipeline.

### Dependencies
If choosing NCCL as the CC backend, the proposed design will introduce [Cupy](https://github.com/cupy/cupy) as a new dependency. Users need to identify the right Cupy version to install based on their CUDA version. Cupy has a bundled version of NCCL, which will be used by default. Alternatively, users can install their desired NCCL version and tell Ray (Cupy) to use that version.
- Note: this Cupy dependency could be removed in longer run by building NCCL into Ray and expose NCCL APIs as python bindings. 
 
If choosing MPI as the CC backend, the proposed design will introduce [MPI4py](https://github.com/mpi4py/mpi4py) as a new dependency.


### Engineering Impact
-  Engineering impact: Minimal change in Ray binary. All the code will be implemented in Python so no impact on building time.
-   Maintenance: NumS and RayML team will develop and maintain it. This code relies on Ray to be tested in a distributed environment.


### Platforms and Environments


### Best Practices
Once this feature gets into Ray: 
- In general, we recommend this set of APIs to perform collective communication of tensors between distributed actors and tasks, regardless of the device that hosts the tensors (CPU RAM or GPU memory).
- We strongly recommend this set of APIs to perform collective communication and point-to-point communication when the tensors are hosted on GPUs.

### Tutorials and Examples

Some working prototypes can be found [here](https://github.com/zhisbug/ray-scalable-ml-design/tree/main/cupy).
Many of the design here draw some insights from `torch.distributed`.

### Compatibility

### User Impact

## Implementation Plan
We plan to prioritize the implementations of NCCL backends so to unblock some ongoing development in NumS and RayML.

## Detailed Design
### Collective Groups
### Communicators
### Implementations of Collective Primitives



## Questions and Discussion Topics
