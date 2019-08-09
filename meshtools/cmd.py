from dolfin import MPI


def mpi_comm():
    return MPI.comm_world


def mpi_barrier():
    MPI.barrier(mpi_comm())


def mpi_rank():
    return MPI.rank(mpi_comm())


def mpi_size():
    return MPI.size(mpi_comm())


def mpi_is_root():
    return mpi_rank() == 0
