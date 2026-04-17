#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void run_allreduce(ncclComm_t comm, int rank, int size, cudaStream_t stream) {
    int count = 1024 * 1024; // 1M elements
    float *d_sendbuff, *d_recvbuff;
    CUDACHECK(cudaMalloc(&d_sendbuff, count * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_recvbuff, count * sizeof(float)));

    // 预热
    NCCLCHECK(ncclAllReduce((const void*)d_sendbuff, (void*)d_recvbuff, count, ncclFloat, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    if (rank == 0) std::cout << "Running NCCL AllReduce..." << std::endl;
    NCCLCHECK(ncclAllReduce((const void*)d_sendbuff, (void*)d_recvbuff, count, ncclFloat, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    CUDACHECK(cudaFree(d_sendbuff));
    CUDACHECK(cudaFree(d_recvbuff));
}

void run_allgather(ncclComm_t comm, int rank, int size, cudaStream_t stream) {
    int count = 1024 * 1024; 
    float *d_sendbuff, *d_recvbuff;
    CUDACHECK(cudaMalloc(&d_sendbuff, count * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_recvbuff, size * count * sizeof(float)));

    if (rank == 0) std::cout << "Running NCCL AllGather..." << std::endl;
    NCCLCHECK(ncclAllGather((const void*)d_sendbuff, (void*)d_recvbuff, count, ncclFloat, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    CUDACHECK(cudaFree(d_sendbuff));
    CUDACHECK(cudaFree(d_recvbuff));
}

void run_alltoall(ncclComm_t comm, int rank, int size, cudaStream_t stream) {
    int count = 1024 * 1024; // 发送给每个rank的数据量
    float *d_sendbuff, *d_recvbuff;
    CUDACHECK(cudaMalloc(&d_sendbuff, size * count * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_recvbuff, size * count * sizeof(float)));

    if (rank == 0) std::cout << "Running NCCL AllToAll (via Send/Recv)..." << std::endl;
    
    NCCLCHECK(ncclGroupStart());
    for (int r = 0; r < size; r++) {
        // 发送数据给 rank r
        NCCLCHECK(ncclSend(d_sendbuff + r * count, count, ncclFloat, r, comm, stream));
        // 接收来自 rank r 的数据
        NCCLCHECK(ncclRecv(d_recvbuff + r * count, count, ncclFloat, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaStreamSynchronize(stream));

    CUDACHECK(cudaFree(d_sendbuff));
    CUDACHECK(cudaFree(d_recvbuff));
}

int main(int argc, char* argv[]) {
    int myRank, nRanks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    ncclUniqueId id;
    if (myRank == 0) ncclGetUniqueId(&id);
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // 绑定 GPU
    int localRank = myRank % 4; // 假设单节点4卡，根据实际情况修改
    CUDACHECK(cudaSetDevice(localRank));

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // 依次运行测试
    run_allreduce(comm, myRank, nRanks, stream);
    run_allgather(comm, myRank, nRanks, stream);
    run_alltoall(comm, myRank, nRanks, stream);

    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommDestroy(comm));
    MPI_Finalize();

    if (myRank == 0) std::cout << "Tests completed successfully." << std::endl;
    return 0;
}