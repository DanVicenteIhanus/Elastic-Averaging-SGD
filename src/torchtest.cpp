#include <torch/torch.h>
#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]) {
  torch::Tensor x = torch::tensor(1.0, torch::requires_grad());
  torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
  torch::Tensor b = torch::tensor(3.0, torch::requires_grad());
  std::cout << x;
  int size, rank, ierr;
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // testing mpi // looks like it works
  std::cout << "Hi, this is process " << rank << ", whats up? \n";
  ierr = MPI_Finalize();
  return 0;
}