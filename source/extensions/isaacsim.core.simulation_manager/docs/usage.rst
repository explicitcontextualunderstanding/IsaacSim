Usage
=====

Simulation Lifecycle
--------------------

The following diagram illustrates the simulation lifecycle and the events taking place within it.
Refer to the :py:class:`~isaacsim.core.simulation_manager.SimulationEvent` enum for more details.

These events can be triggered/operated from:

* The application window, via the :guilabel:`Play`, :guilabel:`Pause`, and :guilabel:`Stop` buttons.
* The Core Experimental API :py:func:`~isaacsim.core.experimental.utils.impl.app.play`,
  :py:func:`~isaacsim.core.experimental.utils.impl.app.pause`, and
  :py:func:`~isaacsim.core.experimental.utils.impl.app.stop` utils functions,
  which are a thin and convenient wrapper around the ``omni.timeline`` API.

|

.. mermaid::

    sequenceDiagram
        participant Timeline as Omniverse<br/>Timeline
        participant SimulationManager
        participant Consumers as Consumers<br/>(Extensions / Users)
        %% play
        Timeline->>+SimulationManager: PLAY
        Note over SimulationManager: Initialize Physics
        SimulationManager->>+Consumers: SIMULATION_SETUP #185;
        Consumers-->-SimulationManager: #160
        SimulationManager->>+Consumers: SIMULATION_STARTED #178;
        Consumers-->-SimulationManager: #160
        SimulationManager-->-Timeline: #160
        %% pause
        opt
        Timeline->>+SimulationManager: PAUSE
        SimulationManager->>+Consumers: SIMULATION_PAUSED
        Consumers-->-SimulationManager: #160
        SimulationManager-->-Timeline: #160
        end
        %% resume
        opt
        Timeline->>+SimulationManager: PLAY
        SimulationManager->>+Consumers: SIMULATION_RESUMED
        Consumers-->-SimulationManager: #160
        SimulationManager-->-Timeline: #160
        end
        %% stop
        opt
        Timeline->>+SimulationManager: STOP
        Note over SimulationManager: Invalidate Physics
        SimulationManager->>+Consumers: SIMULATION_STOPPED
        Consumers-->-SimulationManager: #160
        SimulationManager-->-Timeline: #160
        end

|

**Notes:**

* [1,2] When the application is played for the first time (or after being stopped),
  the :py:attr:`~isaacsim.core.simulation_manager.SimulationEvent.SIMULATION_SETUP` and
  the :py:attr:`~isaacsim.core.simulation_manager.SimulationEvent.SIMULATION_STARTED` events
  are triggered one after the other. The :py:attr:`~isaacsim.core.simulation_manager.SimulationEvent.SIMULATION_SETUP`
  event is used by the Isaac Sim Core extensions, among other extensions, to prepare the physics tensor entities, for example.

  Therefore, from the user perspective and to avoid conflicts, it is recommended that the
  :py:attr:`~isaacsim.core.simulation_manager.SimulationEvent.SIMULATION_STARTED` event be used
  to carry out custom preparatory procedures just before the simulation progresses rather than the
  :py:attr:`~isaacsim.core.simulation_manager.SimulationEvent.SIMULATION_SETUP`.
