.. _datatypes:

Datatypes
=========

.. py:data:: deap_er.datatypes.Individual
   :type: Union[list, array.array, numpy.ndarray]

.. py:data:: deap_er.datatypes.Mates
   :type: Tuple[Individual, Individual]

.. py:data:: deap_er.datatypes.SeqOfNum
   :type: Union[Sequence[int], Sequence[float]]

.. py:data:: deap_er.datatypes.NumOrSeq
   :type: Union[int, float, SeqOfNum]

.. py:data:: deap_er.datatypes.Hof
   :type: Union[HallOfFame, ParetoFront]

.. py:data:: deap_er.datatypes.Stats
   :type: Union[Statistics, MultiStatistics]

.. py:data:: deap_er.datatypes.AlgoResult
   :type: Tuple[list, Logbook]

.. py:data:: deap_er.datatypes.Container
   :type: Type[Union[list, tuple, set, str]]