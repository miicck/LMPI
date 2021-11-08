from mpi4py import MPI
import pkgutil
from typing import Callable, Iterable
import inspect


def mpi_function(func: Callable):
    """
    Decorator to tag a function as parallel. Within
    such a function, MPI calls can be made. The arguments
    to the function will be automatically synchronised
    across processes.

    Parameters
    ----------
    func: Callable
        The function that will be tagged as parallel.

    Returns
    -------
        A wrapped version of the function, that is
        known to LMPI as a parallel function.
    """

    # Wrap the function up
    def wrapper(*args, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == 0:
            # Tell the worker processes what to do (i.e tell them the function id)
            MPI.COMM_WORLD.bcast(MPISession.mpi_function_ids[func.__name__], 0)

        return func(*args, **kwargs)

    # The wrapped function is now an mpi function
    wrapper.mpi_function_name = func.__name__
    wrapper.argument_count = len(inspect.signature(func).parameters)

    return wrapper


class MPISession:

    def __init__(self, main: Callable, modules: Iterable[str] = None):
        """
        Starts an MPI session.

        Parameters
        ----------
        main: Callable
            The main driver function of the program. Any
            functions that you want to parallelize should
            be called (either directly, or indirectly) by
            this function.
        modules: Iterable[str]
            The module names that LMPI will search for MPI
            functions (functions decorated with @mpi_function).
            Will also search submodules recursively.
        """

        if modules is None:
            caller = inspect.stack()[1]
            modules = [caller.filename.replace(".py", "")]

        # Build function maps
        MPISession.find_mpi_functions(modules)

        # Call main on the root process
        if MPI.COMM_WORLD.Get_rank() == 0:
            main()
            MPI.COMM_WORLD.bcast(-1)  # Tell workers we're done

        # Wait for work on all other processes
        else:
            MPISession.await_work()

    ################
    # STATIC STUFF #
    ################

    mpi_functions = []  # List of mpi functions
    mpi_function_ids = {}  # Dictionary of mpi function indices, keyed by function name

    @staticmethod
    def await_work():
        """
        Called on worker processes to listen for and
        carry out work requests from the root process.
        """
        while True:

            # Wait for the name of the function that
            # the root process wants us to execute
            function_id = MPI.COMM_WORLD.bcast(None, 0)

            # Special case, mpi session has finished
            if function_id < 0:
                break  # We're done

            # Locate the function that needs calling
            if 0 <= function_id < len(MPISession.mpi_functions):
                work_function = MPISession.mpi_functions[function_id]
            else:
                raise Exception(f"Unknown MPI function id: {function_id}")

            # Call the work function
            work_function([None] * getattr(work_function, "argument_count"))

    @staticmethod
    def recurse_modules(path: str):
        """
        Recursively return the path to all modules
        and submodules at the given path.

        Parameters
        ----------
        path: str
            The path where we start searching for modules
            (is immediately yielded by the function).
        Yields
        ------
        path: str
            The next path in a depth-first recursive
            search for submodules at path.
        """
        yield path
        for p in pkgutil.walk_packages([path]):
            next_path = f"{p.module_finder.path}/{p.name}"
            yield next_path
            yield from MPISession.recurse_modules(next_path)

    @staticmethod
    def find_mpi_functions(modules: Iterable[str]):
        """
        Call to enumerate all mpi functions and build
        the following function maps:
            MPISession.mpi_functions
            MPISession.mpi_function_ids

        Parameters
        ----------
        modules: Iterable[str]
            The module names that to will search for MPI
            functions (functions decorated with @mpi_function).
            Will also search submodules recursively.
        """
        module_set = set(modules)

        # Will contain all mpi functions
        mpi_functions = {}

        # Iterate over modules
        for p in pkgutil.iter_modules():

            # This will be replaced with a check
            # that we are in the QUEST codebase
            if p.name in module_set:

                path = f"{p.module_finder.path}/{p.name}"
                for mod in MPISession.recurse_modules(path):

                    # Check if the module contains mpi_function
                    # This is done so that we don't have to
                    # import every module that we scan.
                    with open(mod + ".py", "r") as mod_file:
                        if not ("@mpi_function" in mod_file.read()):
                            continue

                    try:
                        # Import the module
                        mod = __import__(mod.replace(path, p.name).replace("/", "."))
                    except:
                        continue

                    # Search the imported module for mpi functions
                    for f in dir(mod):
                        f = getattr(mod, f)
                        if hasattr(f, "mpi_function_name"):
                            name = getattr(f, "mpi_function_name")
                            mpi_functions[name] = f

        # Build function maps
        MPISession.mpi_function_ids = {}
        MPISession.mpi_functions = []
        for i, name in enumerate(sorted(mpi_functions)):
            MPISession.mpi_function_ids[name] = i  # Map from name to id
            MPISession.mpi_functions.append(mpi_functions[name])  # Map from id to function
