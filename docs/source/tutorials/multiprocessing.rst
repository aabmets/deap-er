Multiprocessing
===============

In this section of the tutorial we will learn how to speed up the computation of evolutionary algorithms
by using multiprocessing. The distribution of workloads over multiple cores or computing clusters requires
the serialization of data objects, which is usually done by pickling, therefore all objects that are to be
distributed *(e.g. functions and their arguments)* must be pickleable.

The correct way of using multiprocessing with **DEAP-ER** is to override the default ``map`` function in the
toolbox with one that supports parallel execution. The only requirement of this ``map`` function is that
its signature and return type must match with the regular ``map`` function. This enables the use of any
third-party distributed computing libraries, such as `Ray <https://docs.ray.io/en/latest/ray-more-libs/
multiprocessing.html>`_, that implement the same interface.

.. code-block::
   :caption: With context manager

    # Using the multiprocessing library
    with multiprocessing.Pool() as pool:
        toolbox.register('map', pool.map)
        # Execute the evolution


    # Using the concurrent.futures library
    with concurrent.futures.ProcessPoolExecutor() as executor:
        toolbox.register('map', executor.map)
        # Execute the evolution


.. code-block::
   :caption: Without context manager

    # Using the multiprocessing library
    pool = multiprocessing.Pool()
    toolbox.register('map', pool.map)
    # Execute the evolution
    pool.close()
    pool.join()


    # Using the concurrent.futures library
    executor = concurrent.futures.ProcessPoolExecutor()
    toolbox.register('map', executor.map)
    # Execute the evolution
    executor.shutdown()

.. raw:: html

   <br />

.. note::
    It is also suggested to take a look at the :ref:`full example <using_mp>` of using multiprocessing with **DEAP-ER**.

.. attention::
    | When using multiprocessing on Windows, the main function needs to be guarded
    | with the :code:`if __name__ == '__main__'` statement.

.. tip::
    | An excellent third-party tutorial about multiprocessing in Python is available
      `HERE <https://superfastpython.com/multiprocessing-in-python>`_.
    | The reference manuals of the `multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_
      and the `concurrent.futures <https://docs.python.org/3/library/concurrent.futures.html>`_ libraries
    | should also prove useful.

.. raw:: html

   <br />
