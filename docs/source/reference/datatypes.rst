.. _datatypes:

Datatypes
=========

| These custom datatype aliases are available from their respective
| subpackages, as well as from the general :data:`deap_er.datatypes` module.


Base
----

.. py:data:: Individual
   :type: Union[list, array.array, numpy.ndarray]

.. py:data:: Mates
   :type: tuple[Individual, Individual]

.. py:data:: SeqOfNum
   :type: Union[Sequence[int], Sequence[float]]

.. py:data:: NumOrSeq
   :type: Union[int, float, SeqOfNum]


Records
-------

.. py:data:: Hof
   :type: Union[HallOfFame, ParetoFront]

.. py:data:: Stats
   :type: Union[Statistics, MultiStatistics]

.. py:data:: AlgoResult
   :type: tuple[list, Logbook]


Genetic Programming (GP)
------------------------

.. py:data:: TerminalTypes
   :type: Union[str, int, float, Callable]

.. py:data:: ExprTypes
   :type: Union[str, PrimitiveTree]

.. py:data:: ListOfSets
   :type: list[PrimitiveSetTyped]

.. py:data:: Graph
   :type: tuple[list, list, dict]


Utilities
---------

.. py:data:: Containers
   :type: type[Union[list, tuple, set, str]]
