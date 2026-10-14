.. _kw_ensemble_ti_nep_ghost:

:attr:`ensemble` (TI_NEP_GHOST)
===============================

This keyword is used to set up a thermodynamic integration that removes one or more atoms (the ghost atoms) from a :term:`NEP` model, giving the free energy of forming a vacancy.
It can be run as a nonequilibrium switching (:attr:`neq`, see [Freitas2016]_) or as an equilibrium ladder of fixed-lambda windows (:attr:`eq`).

The system is driven by the Hamiltonian

.. math::

   H(\lambda) = (1-\lambda) U_1 + \lambda U_2,

where :math:`U_1` is the NEP potential energy of the full system and :math:`U_2` is the NEP potential energy of the system without the ghost atoms, plus the energy of Einstein springs that tether each ghost atom to its initial position.
At :math:`\lambda=1` the ghost atoms no longer interact with the other atoms and only feel their springs.
This is an NVT ensemble with a Langevin thermostat, so the free energy difference is evaluated at the volume of the simulation box.

The ghost atoms are the atoms in group 1 of grouping method 0 in :ref:`model.xyz <model_xyz>`; all the other atoms should be in group 0.
The NEP model is the first :ref:`potential <kw_potential>` in the :attr:`run.in` file, and must be a regular NEP model.
The ensemble evaluates this model on the system without the ghost atoms itself, so a single :attr:`potential` line is enough.

Syntax
------

The nonequilibrium mode is specified as follows::

    ensemble ti_nep_ghost neq temp <temperature> tperiod <tau_temperature> tequil <equilibrium_time> tswitch <switch_time> ghost_spring <k> path_order <n>

The equilibrium mode is specified as follows::

    ensemble ti_nep_ghost eq <N_lambda> <equil_steps> <prod_steps> <direction> temp <temperature> tperiod <tau_temperature> ghost_spring <k> path_order <n> pre_equil <pre_equil_steps>

Parameters of both modes:

- :attr:`<temperature>`: The temperature of the simulation.
- :attr:`<tau_temperature>`: This parameter is optional, and defaults to ``100``. It determines the period of the thermostat in units of the timestep.
- :attr:`<k>`: This parameter is optional, and defaults to ``10``. The spring constant of the ghost atoms in units of eV/Å\ :sup:`2`.
- :attr:`<n>`: This parameter is optional, and defaults to ``4``. The order of the Beta smoothstep :math:`S_n(x) = I_x(n+1, n+1)`, an integer from 4 to 8. :math:`S_4` is the smootherstep :math:`126x^5 - 420x^6 + 540x^7 - 315x^8 + 70x^9`. Higher orders put more of the path close to :math:`\lambda=0` and :math:`\lambda=1`.

Parameters of the :attr:`neq` mode:

- :attr:`<equilibrium_time>`: The number of timesteps to equilibrate the system at :math:`\lambda=0` and at :math:`\lambda=1`.
- :attr:`<switch_time>`: The number of timesteps to vary lambda from 0 to 1 (and from 1 to 0), following :math:`\lambda(t) = S_n(t / t_{\rm switch})`.

If you do not specify :attr:`<equilibrium_time>` and :attr:`<switch_time>`, they will be automatically set in a 1:4 ratio.
If you specify them, you must specify both, and the number of steps in the :ref:`run keyword <kw_run>` should be 2 × (:attr:`<equilibrium_time>` + :attr:`<switch_time>`).

Parameters of the :attr:`eq` mode:

- :attr:`<N_lambda>`: The number of lambda windows, at least 2. The windows are placed at :math:`\lambda_k = S_n(k/(N_\lambda - 1))`, :math:`k = 0, \ldots, N_\lambda - 1`.
- :attr:`<equil_steps>`: The number of timesteps to equilibrate in each window before sampling.
- :attr:`<prod_steps>`: The number of timesteps to sample :math:`\langle U_2 - U_1 \rangle` in each window.
- :attr:`<direction>`: ``forward`` runs the windows from :math:`\lambda=0` to :math:`\lambda=1`, ``backward`` from :math:`\lambda=1` to :math:`\lambda=0`.
- :attr:`<pre_equil_steps>`: This parameter is optional, and defaults to ``0``. The number of timesteps before the first window, at :math:`\lambda=0` for ``forward``, or switching lambda from 0 to 1 with the smootherstep for ``backward``. No samples are taken.

The number of steps in the :ref:`run keyword <kw_run>` must be :attr:`<pre_equil_steps>` + :attr:`<N_lambda>` × (:attr:`<equil_steps>` + :attr:`<prod_steps>`).
The window means are integrated over :math:`\lambda` with the trapezoidal rule.

For compatibility with older inputs, :attr:`ensemble ti_nep_ghost` without a mode is the :attr:`neq` mode, and :attr:`ensemble ti_eq_nep_ghost <N_lambda> ...` is the :attr:`eq` mode.

Only the forces are mixed; the potential energy and pressure written by :ref:`dump_thermo <kw_dump_thermo>` are those of the full system (:math:`U_1`) during the whole run.

Free energy
-----------

The integral :math:`E_{\rm diff} = \int_0^1 \langle U_2 - U_1 \rangle_\lambda \, d\lambda` is the free energy difference between the system without the ghost atoms plus the Einstein oscillators and the full system.
Subtracting the free energy of the classical Einstein oscillators,

.. math::

   E_{\rm Ein} = 3 k_B T \sum_{i \in \rm ghost} \ln \frac{\hbar \omega_i}{k_B T}, \quad \omega_i = \sqrt{k / m_i},

gives :math:`F_{\rm diff} = E_{\rm diff} - E_{\rm Ein}`, the Helmholtz free energy of the system without the ghost atoms minus that of the full system, at the same volume.
Together with the chemical potential of the removed atoms, this gives the vacancy formation free energy.

At :math:`\lambda` close to 1 the neighbors of the ghost atoms can move into the vacancy, so :math:`U_2 - U_1` becomes large and negative there.
Increase :attr:`<switch_time>` or :attr:`<N_lambda>` until :attr:`F_diff` is converged, and compare the forward and backward directions.

Examples
--------

.. code-block:: rst

    potential nep.txt
    velocity 600
    ensemble ti_nep_ghost neq temp 300 tequil 10000 tswitch 200000 ghost_spring 60
    run 420000

This command equilibrates for 10000 timesteps at :math:`\lambda=0`, switches lambda from 0 to 1 in 200000 timesteps, equilibrates for 10000 timesteps at :math:`\lambda=1`, and switches lambda back from 1 to 0 in 200000 timesteps.

.. code-block:: rst

    potential nep.txt
    velocity 600
    ensemble ti_nep_ghost eq 101 1000 2000 forward temp 300 ghost_spring 60 path_order 5 pre_equil 20000
    run 323000

This command equilibrates for 20000 timesteps at :math:`\lambda=0`, then runs 101 windows from :math:`\lambda=0` to :math:`\lambda=1`, each with 1000 equilibration and 2000 sampling timesteps.

Output files
------------

The :attr:`neq` mode produces a csv file :attr:`ti_nep_ghost.csv` with the columns lambda, dlambda, the potential energy of the full system, the potential energy of the system without the ghost atoms, and the spring energy (eV/atom).
The file is only written during the switching.
It also produces a yaml file :attr:`ti_nep_ghost.yaml`.

The :attr:`eq` mode produces

- :attr:`ti_eq_nep_ghost_dudl.csv`, with the step, window index, :math:`u = k/(N_\lambda - 1)`, :math:`\lambda` and :math:`U_2 - U_1` (eV) of every sampled step,
- :attr:`ti_eq_nep_ghost_summary.csv`, with the window index, :math:`u`, :math:`\lambda`, the mean and standard error of :math:`U_2 - U_1` (eV), the number of samples, and the integral over the windows completed so far,
- a yaml file :attr:`ti_eq_nep_ghost.yaml`.

The yaml files contain the mode, the path order, :attr:`E_diff`, :attr:`E_Ein`, :attr:`F_diff` (eV) and the temperature; the :attr:`eq` file also contains the direction, :attr:`N_lambda` and the number of completed windows.

All files are overwritten by a subsequent run in the same directory, so use separate directories for separate calculations.
