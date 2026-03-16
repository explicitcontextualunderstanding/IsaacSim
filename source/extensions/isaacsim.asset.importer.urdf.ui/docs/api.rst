URDF Importer UI Extension [isaacsim.asset.importer.urdf.ui]
############################################################

URDF Import UI Workflow
========================
Enable the URDF UI extension to import URDF assets through the asset importer UI.
The UI uses the same configuration object as the core importer.


Deprecated Commands
===================
The following commands are deprecated and provided for backward compatibility only.
New code should use :class:`URDFImporter` and :class:`URDFImporterConfig` directly.

.. autoclass:: isaacsim.asset.importer.urdf.ui.impl.command.URDFCreateImportConfig
    :members:
    :undoc-members:
    :no-show-inheritance:

.. autoclass:: isaacsim.asset.importer.urdf.ui.impl.command.URDFParseText
    :members:
    :undoc-members:
    :no-show-inheritance:

.. autoclass:: isaacsim.asset.importer.urdf.ui.impl.command.URDFParseFile
    :members:
    :undoc-members:
    :no-show-inheritance:

.. autoclass:: isaacsim.asset.importer.urdf.ui.impl.command.URDFImportRobot
    :members:
    :undoc-members:
    :no-show-inheritance:
