.. role:: raw-html(raw)
   :format: html

Important Differences
=====================

As this library is an *evolution* of the original `DEAP <https://github.com/DEAP/deap>`_ library
:raw-html:`<span style="font-size:70%;">(pun intended)</span>`, there are a few important
differences that the user should be aware of:

#. The whole documentation has been reworked for better comprehensibility.
#. The whole codebase has been completely refactored for better usability and maintainability.
#. Algorithms, strategies and benchmarks have been moved into the **tools** namespace.
#. All **camelCase** functions and methods have been renamed to **snake_case**.
#. Some functions and/or methods have been completely renamed.
#. All functions and methods have received proper type hints.
#. The parameters of many functions have been reordered and/or renamed.
#. Some properties of classes have been changed into method calls.
#. The hypervolume class is now pure-python only. The C-implementation has been
   removed to eliminate complexities regarding external module compilation.
#. The **least_contrib** hypervolume indicator is now capable of multiprocessing.
#. State persistence has been implemented with the Checkpoint class.
#. All deprecated and obsolete code is removed.
