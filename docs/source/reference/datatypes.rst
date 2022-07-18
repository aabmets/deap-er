Datatypes
=========

These custom datatypes are used by the various methods and functions of the DEAP-er library.

.. py:data:: Subscript
   :type: Union[dict, list, array.array, numpy.ndarray]

   | Any **mutable datatype** which can be used with the subscript operator **[ ]**.
   | This includes the `dict`_, the `list`_, the `array.array`_, the `numpy.ndarray`_ and other
   | datatypes as well as subclasses of these created by the :func:`~deap_er.creator.create` function.

.. py:data:: PairSIS
   :type: Tuple[Subscript, Subscript]

   A pair of :data:`Subscript` datatypes.

.. py:data:: NumOrSeq
   :type: Union[int, float, Sequence[int], Sequence[float]]

   Either an :data:`int` or a :data:`float` or a sequence of them.

.. py:data:: SeqOfNum
   :type: Union[Sequence[int], Sequence[float]]

   A sequence of :data:`int` or :data:`float` values.

.. py:data:: PairLAL
   :type: Tuple[list, Logbook]

   A pair of :data:`list` and :data:`Logbook` datatypes.


.. _dict: https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
.. _list: https://docs.python.org/3/library/stdtypes.html#lists
.. _array.array: https://docs.python.org/3/library/array.html
.. _numpy.ndarray: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
