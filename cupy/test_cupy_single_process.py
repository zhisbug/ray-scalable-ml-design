import cupy as cp
from cupy.cuda import nccl

devs = [0, 1]
comms = nccl.NcclCommunicator.initAll(devs)
nccl.groupStart()
for comm in comms:
    dev_id = comm.device_id()
    rank = comm.rank_id()
    assert rank == dev_id

    if rank == 0:
        with cp.cuda.Device(dev_id):
            sendbuf_0 = cp.arange(10, dtype=cp.int64)
            # comm.send(sendbuf.data.ptr, 10, cp.cuda.nccl.NCCL_INT64,
            #           1, cp.cuda.Stream.null.ptr)
            recvbuf_0 = cp.zeros(10, dtype=cp.int64)
            comm.allReduce(sendbuf_0.data.ptr, recvbuf_0.data.ptr, 10, cp.cuda.nccl.NCCL_INT64,
                           1, cp.cuda.Stream.null.ptr)
    elif rank == 1:
        with cp.cuda.Device(dev_id):
            sendbuf_1 = cp.arange(10, dtype=cp.int64)
            recvbuf_1 = cp.zeros(10, dtype=cp.int64)
            comm.allReduce(sendbuf_1.data.ptr, recvbuf_1.data.ptr, 10, cp.cuda.nccl.NCCL_INT64,
                           1, cp.cuda.Stream.null.ptr)
nccl.groupEnd()

# check result
with cp.cuda.Device(1):
    expected = cp.arange(10, dtype=cp.int64) * 2
    assert (recvbuf_0 == expected).all()

