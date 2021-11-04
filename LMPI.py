from mpi4py import MPI
import pkgutil


# Decorator to tag functions as mpi functions
def mpi_function(func):

    # Wrap the function up
    def wrapper(*args, **kwargs):

        if MPI.COMM_WORLD.Get_rank() == 0:
            # Tell the worker processes what to do
            MPI.COMM_WORLD.bcast(func.__name__, 0)
            MPI.COMM_WORLD.bcast(args, 0)

        return func(*args, **kwargs)

    # The wrapped function is now an mpi function
    wrapper.mpi_function_name = func.__name__
    return wrapper


# Function to start an mpi session
def mpi_session(main):

    # Initialize the collection of mpi functions
    mpi_session.mpi_functions = {}
    mpi_session.rank = MPI.COMM_WORLD.Get_rank()

    # Iterate over modules
    for p in pkgutil.iter_modules():

        # This will be replaced with a check
        # that we are in the QUEST codebase
        if p[1] == "test":

            # Import the module
            p = __import__(p.name)

            # Search the imported module for mpi functions
            for f in dir(p):
                f = getattr(p, f)
                if hasattr(f, "mpi_function_name"):
                    name = getattr(f, "mpi_function_name")
                    mpi_session.mpi_functions[name] = f

    # Call main on the root process
    if MPI.COMM_WORLD.Get_rank() == 0:
        main()
        MPI.COMM_WORLD.bcast("__session_finished__")

    # Wait for work on all other processes
    else:
        while True:

            # Wait for the name of the function that
            # the root process wants us to execute
            function_name = MPI.COMM_WORLD.bcast(None, 0)

            # Special case, mpi session has finished
            if function_name == "__session_finished__":
                break # We're done

            # Locate the function that needs calling
            if not function_name in mpi_session.mpi_functions:
                raise Exception(f"Unknown MPI function: {function_name}")
            work_function = mpi_session.mpi_functions[function_name]

            # Wait for the arguments that the root process
            # wants us to pass to the function
            arguments = MPI.COMM_WORLD.bcast(None, 0)

            # Call the work function
            work_function(*arguments)
