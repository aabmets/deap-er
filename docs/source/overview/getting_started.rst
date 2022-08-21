Getting Started
===============

Installation
++++++++++++

This library can be installed with:

.. code-block:: text

    pip install deap-er

or if you're using the `Poetry <https://python-poetry.org/docs/>`_ package manager:

.. code-block:: text

    poetry add deap-er


.. raw:: html

   <br/>
   <hr>


Namespaces
++++++++++

The functionality of this library is divided into the following namespaces:

    * **base** - The main *Toolbox* and *Fitness* classes.
    * **creator** - Type creator for use with toolboxes.
    * **tools** - Essential evolution components:

        - algorithms
        - operators
        - strategies
        - records
        - utilities
        - benchmarks

    * **gp** - Components for genetic programming.
    * **env** - The *Checkpoint* class for state persistence.
    * **dtypes** - Datatype aliases used by this library.

These namespaces can be imported with:

.. code-block::

   from deap_er import base, creator, tools, gp, env, dtypes

.. raw:: html

   <br />
