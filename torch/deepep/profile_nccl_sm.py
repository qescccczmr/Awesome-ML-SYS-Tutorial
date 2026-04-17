#!/usr/bin/env python3
"""
Profile NCCL collective operations to measure SM occupancy.
Usage: torchrun --nproc_per_node=2 profile_nccl_sm.py
"""
import torch
import torch.distributed as dist
import os
import time

def profile_collective(op_name, collective_fn, tensor_size_mb=100):
    """Profile a NCCL collective operation."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create tensor
    num_elements = (tensor_size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
    tensor = torch.randn(num_elements, device='cuda')

    # Warmup
    for _ in range(5):
        collective_fn(tensor)
    torch.cuda.synchronize()

    # Profile with nsys or measure time
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Profiling {op_name} with {tensor_size_mb}MB tensor")
        print(f"{'='*60}")

    start = time.time()
    for _ in range(10):
        collective_fn(tensor)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    if rank == 0:
        print(f"Average time: {elapsed/10*1000:.2f} ms")
        print(f"Bandwidth: {tensor_size_mb * world_size / (elapsed/10) / 1024:.2f} GB/s")

def main():
    # Initialize process group
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"NCCL version: {torch.cuda.nccl.version()}")
        print(f"NCCL_NCHANNELS_PER_NET_PEER: {os.environ.get('NCCL_NCHANNELS_PER_NET_PEER', 'default')}")

    # Test different collective operations
    tensor_size = 100  # MB

    # AllReduce
    profile_collective(
        "AllReduce",
        lambda t: dist.all_reduce(t, op=dist.ReduceOp.SUM),
        tensor_size
    )

    # AllGather
    output_tensors = [torch.empty_like(torch.randn(
        (tensor_size * 1024 * 1024) // 4, device='cuda'
    )) for _ in range(world_size)]
    profile_collective(
        "AllGather",
        lambda t: dist.all_gather(output_tensors, t),
        tensor_size
    )

    # AlltoAll
    input_list = [torch.randn(
        (tensor_size * 1024 * 1024) // 4 // world_size, device='cuda'
    ) for _ in range(world_size)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    profile_collective(
        "AlltoAll",
        lambda _: dist.all_to_all(output_list, input_list),
        tensor_size
    )

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
