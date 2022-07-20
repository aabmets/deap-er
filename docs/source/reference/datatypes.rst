Datatypes
=========

.. py:data:: deap_er.datatypes.Individual
   :type: Union[list, array.array, numpy.ndarray]

   Any *MutableSequence* which has the *__getitem__* and *__setitem__* methods
   and can be used with the subscript *[ ]* operator. This includes the *list*,
   the *array.array*, the *numpy.ndarray* and other datatypes as well as
   their subclasses created with the :func:`~deap_er.creator.create` function.


.. py:data:: deap_er.datatypes.Mates
   :type: Tuple[Individual, Individual]

   A pair of :data:`Individual` datatypes.


.. py:data:: deap_er.datatypes.SeqOfNum
   :type: Union[Sequence[int], Sequence[float]]

   A sequence of *int* or *float* values.


.. py:data:: deap_er.datatypes.NumOrSeq
   :type: Union[int, float, SeqOfNum]

   Either an *int*, a *float* or a :data:`SeqOfNum`.


.. py:data:: deap_er.datatypes.Hof
   :type: Union[HallOfFame, ParetoFront]

   Either a :data:`~deap_er.records.HallOfFame` or a :data:`~deap_er.records.ParetoFront` object.


.. py:data:: deap_er.datatypes.Stats
   :type: Union[Statistics, MultiStatistics]

   Either a :data:`~deap_er.records.Statistics` or a :data:`~deap_er.records.MultiStatistics` object.


.. py:data:: deap_er.datatypes.AlgoResult
   :type: Tuple[list, Logbook]

   A pair of *list* and :data:`~deap_er.records.Logbook` objects.
