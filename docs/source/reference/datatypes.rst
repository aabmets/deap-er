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

.. py:data:: Mutant
   :type: tuple[Individual]

.. py:data:: NumOrSeq
   :type: Union[int, float, Sequence[int], Sequence[float]]


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

.. py:data:: GPIndividual
   :type: Union[list, PrimitiveTree]

.. py:data:: GPMates
   :type: Tuple[GPIndividual, GPIndividual]

.. py:data:: GPMutant
   :type: Tuple[GPIndividual]

.. py:data:: GPExprTypes
   :type: Union[str, GPIndividual]

.. py:data:: GPTypedSets
   :type: list[PrimitiveSetTyped]

.. py:data:: GPGraph
   :type: tuple[list, list, dict]
