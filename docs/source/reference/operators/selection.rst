.. _sel_ops:

Selection Operators
===================

Genetic Selection
-----------------

.. autofunction:: deap_er.operators.sel_spea_2
.. autofunction:: deap_er.operators.sel_nsga_2
.. autofunction:: deap_er.operators.sel_nsga_3
.. autoclass:: deap_er.operators.SelNSGA3WithMemory
   :members:

.. raw:: html

   <br />
   <hr>


Tournament Selection
--------------------
.. autofunction:: deap_er.operators.sel_tournament
.. autofunction:: deap_er.operators.sel_double_tournament
.. autofunction:: deap_er.operators.sel_tournament_dcd

.. raw:: html

   <br />
   <hr>


Lexicase Selection
------------------
.. autofunction:: deap_er.operators.sel_lexicase
.. autofunction:: deap_er.operators.sel_epsilon_lexicase

.. raw:: html

   <br />
   <hr>


Various Selection
-----------------
.. autofunction:: deap_er.operators.sel_random
.. autofunction:: deap_er.operators.sel_best
.. autofunction:: deap_er.operators.sel_worst
.. autofunction:: deap_er.operators.sel_roulette
.. autofunction:: deap_er.operators.sel_stochastic_universal_sampling

.. raw:: html

   <br />
   <hr>


Helpers
-------
.. autofunction:: deap_er.operators.assign_crowding_dist
.. autofunction:: deap_er.operators.uniform_reference_points
