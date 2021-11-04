from LMPI import mpi_session, mpi_function
import numpy as np


def i_dont_want_to_parallelize(data):
    # Complicated stuff happens here
    print(f"Serial: {mpi_session.rank}")
    return data


@mpi_function
def i_do_want_to_parallelize(data):
    # Expensive stuff happens here
    print(f"Parallel: {mpi_session.rank}")
    return data


def main():
    mode = 2

    data = np.random.random(10)

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
    mpi_session(main)
