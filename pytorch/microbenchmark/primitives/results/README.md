#  Benchmarking Results
## Summary
- [Ray tested collectives](https://github.com/zhisbug/ray-scalable-ml-design/tree/main/pytorch/microbenchmark/primitives/ray): `reduce`, `broadcast`, `allreduce`, `allgather`, `gather`, `send-recv`
- [PyTorch-NCCL tested collectives](https://github.com/zhisbug/ray-scalable-ml-design/tree/main/pytorch/microbenchmark/primitives/pytorch): `reduce`, `broadcast`, `allreduce`, `allgather`
- [PyTorch-GLOO tested collectives](https://github.com/zhisbug/ray-scalable-ml-design/tree/main/pytorch/microbenchmark/primitives/pytorch): `reduce`, `broadcast`, `allreduce`, `allgather`, `gather`, `send-recv`


##  Single-node Multi-GPU Setting

### Setup
- Cluster: **only 1** node
- Node hardware: each node is with **2 GeForce RTX 2080**, **1-Gigabit** Ethernet switch
- Node software: CUDA 10.1, torch==1.7.0+cu101
- ~~CPU tensor (not tested): `torch.FloatTensor(size).fill_(value)`~~
- GPU tensor: `torch.cuda.FloatTensor(size).fill_(value)`
- Collective world size:  `[2]`
- Collective tensor size: `[1KB, 32KB, 1MB, 32MB, 1GB]`
- **Title format:** `{setting}-{gpu/cpu}-{collective op}-{world_size}`

### GPU Tensor



#### Reduce
![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/multigpu-gpu-reduce-2.png?raw=true)
#### Broadcast
![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/multigpu-gpu-broadcast-2.png?raw=true)
#### AllReduce
![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/multigpu-gpu-allreduce-2.png?raw=true)

#### AllGather
![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/multigpu-gpu-allgather-2.png?raw=true)
  

## Distributed Settings
### Setup
- Cluster: **16 nodes**
- Node hardware: each node is with Supermicro 5018GR-T, 16-core Intel Xeon, 64 GiB RAM, **TitanX GPU**, Cisco model Nexus 3264-Q, 64-port QSFP+ **40-Gigabit** Ethernet switch
- Node software: CUDA 10.1, torch==1.7.0+cu101
- CPU tensor: `torch.FloatTensor(size).fill_(value)`
- GPU tensor: `torch.cuda.FloatTensor(size).fill_(value)`
- Collective world size:  `[2, 4, 8, 16]`
- Collective tensor size: `[1KB, 32KB, 1MB, 32MB, 1GB]`
- **Title format:** `{setting}-{gpu/cpu}-{collective op}-{world_size}`


### CPU Tensor
#### Reduce
![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-reduce-2.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-reduce-4.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-reduce-8.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-reduce-16.png?raw=true)


#### Broadcast
![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-broadcast-2.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-broadcast-4.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-broadcast-8.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-broadcast-16.png?raw=true)

#### AllReduce

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-allreduce-2.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-allreduce-4.png?raw=true)


![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-allreduce-8.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-allreduce-16.png?raw=true)

#### AllGather

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-allgather-2.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-allgather-4.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-allgather-8.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-allgather-16.png?raw=true)

#### Gather

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-gather-2.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-gather-4.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-gather-8.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-gather-16.png?raw=true)

#### Send/recv
![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-sendrecv-2.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-sendrecv-4.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-sendrecv-8.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-cpu-sendrecv-16.png?raw=true)


### GPU Tensor

#### Reduce
![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-reduce-2.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-reduce-4.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-reduce-8.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-reduce-16.png?raw=true)

#### Broadcast
![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-broadcast-2.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-broadcast-4.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-broadcast-8.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-broadcast-16.png?raw=true)


#### AllReduce

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-allreduce-2.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-allreduce-4.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-allreduce-8.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-allreduce-16.png?raw=true)

#### AllGather

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-allgather-2.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-allgather-4.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-allgather-8.png?raw=true)

![enter image description here](https://github.com/zhisbug/ray-scalable-ml-design/blob/main/pytorch/microbenchmark/primitives/results/plots/distributed-gpu-allgather-16.png?raw=true)