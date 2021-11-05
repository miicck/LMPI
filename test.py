from LMPI import MPI, MPISession, mpi_function
import numpy as np


def i_dont_want_to_parallelize(data):
    # Complicated stuff happens here
    print(f"Serial: {MPI.COMM_WORLD.Get_rank()} {data}")
    return data


@mpi_function
def i_do_want_to_parallelize(data):
    # Expensive stuff happens here
    data = MPI.COMM_WORLD.bcast(data, 0)
    print(f"Parallel: {MPI.COMM_WORLD.Get_rank()} {data}")
    return data


def main():
    mode = 2

    data = np.random.random(4)

    if mode == 3:
        data = i_dont_want_to_parallelize(data)
        data = i_dont_want_to_parallelize(data)

    if mode == 2:
        data = i_dont_want_to_parallelize(data)
        data = i_do_want_to_parallelize(data)

    else:
        data = i_dont_want_to_parallelize(data)
        data = i_dont_want_to_parallelize(data)
        data = i_dont_want_to_parallelize(data)


if __name__ == "__main__":
    MPISession(main)
