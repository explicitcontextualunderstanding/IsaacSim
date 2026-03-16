API
===

.. warning::

    **The API featured in this extension is experimental and subject to change without deprecation cycles.**
    Although we will try to maintain backward compatibility in the event of a change, it may not always be possible.

Python API
----------

.. Summary

The following table summarizes the available materials.

.. currentmodule:: isaacsim.sensors.experimental.camera

.. rubric:: sensors
.. autosummary::
    :nosignatures:

    CameraSensor
    SingleViewDepthCameraSensor
    TiledCameraSensor

.. rubric:: utils
.. autosummary::
    :nosignatures:

    draw_annotator_data_to_image

Sensors
"""""""

.. autoclass:: isaacsim.sensors.experimental.camera.CameraSensor
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: isaacsim.sensors.experimental.camera.SingleViewDepthCameraSensor
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: isaacsim.sensors.experimental.camera.TiledCameraSensor
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Utils
"""""

.. autofunction:: isaacsim.sensors.experimental.camera.draw_annotator_data_to_image
