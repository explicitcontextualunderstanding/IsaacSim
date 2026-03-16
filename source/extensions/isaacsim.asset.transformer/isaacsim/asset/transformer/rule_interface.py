"""Abstract base interface for asset transformer rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pxr import Usd

from .models import RuleConfigurationParam


class RuleInterface(ABC):
    """Abstract base class for asset transformation rules.

    Implementations operate on a source :class:`pxr.Usd.Stage` and may write
    opinions to a destination :class:`pxr.Usd.Stage`. Subclasses should record
    human-readable log messages and any identifiers for stages or layers they
    affect so the manager can produce comprehensive reports.

    Rules may request a stage replacement by returning a stage identifier (file
    path) from :meth:`process_rule`. The manager will open the new stage and
    use it for subsequent rules in the pipeline.

    Args:
        source_stage: Input stage providing opinions to read from.
        package_root: Root directory for output files.
        destination_path: Relative path for rule outputs.
        args: Mapping of parameters including keys such as ``destination`` and
            ``params``.
    """

    def __init__(self, source_stage: Usd.Stage, package_root: str, destination_path: str, args: dict[str, Any]) -> None:
        self.source_stage: Usd.Stage = source_stage
        self.package_root: str = package_root
        self.destination_path: str = destination_path
        self.args: dict[str, Any] = args or {}
        self._log: list[str] = []
        self._affected_stages: list[str] = []

    @abstractmethod
    def process_rule(self) -> str | None:
        """Execute the rule logic.

        This method must be implemented by subclasses. Implementations should
        emit log messages via :meth:`log_operation` and record any affected
        stage or layer identifiers via :meth:`add_affected_stage`.

        Returns:
            The file path of the stage to be used by subsequent rules. Return
            ``None`` if the current working stage should continue to be used.
            If a path is returned and differs from the current working stage,
            the manager will open the new stage for subsequent rules.

        Example:

        .. code-block:: python

            from pxr import Usd
            from isaacsim.asset.transformer import RuleInterface

            class NoOpRule(RuleInterface):
                def process_rule(self) -> None:
                    self.log_operation("noop")

            stage = Usd.Stage.CreateInMemory()
            NoOpRule(stage, "/tmp", "", {"destination": "out.usda"}).process_rule()
        """
        raise NotImplementedError

    def log_operation(self, message: str) -> None:
        """Append a human-readable message to the operation log.

        Args:
            message: Message to record in the rule execution log.

        Example:

        .. code-block:: python

            rule.log_operation("Copied prim /World")
        """
        self._log.append(message)

    def get_operation_log(self) -> list[str]:
        """Return the accumulated operation log messages.

        Returns:
            List of log message strings in chronological order.

        Example:

        .. code-block:: python

            messages = rule.get_operation_log()
        """
        return list(self._log)

    def add_affected_stage(self, stage_identifier: str) -> None:
        """Record an identifier for a stage or layer affected by this rule.

        Args:
            stage_identifier: Logical label, file path, or layer id that was
                created, modified, or otherwise affected by the rule.

        Example:

        .. code-block:: python

            rule.add_affected_stage("/tmp/output.usda")
        """
        if stage_identifier and stage_identifier not in self._affected_stages:
            self._affected_stages.append(stage_identifier)

    def get_affected_stages(self) -> list[str]:
        """Return identifiers for stages or layers affected by this rule.

        Returns:
            List of unique identifiers provided via :meth:`add_affected_stage`.

        Example:

        .. code-block:: python

            affected = rule.get_affected_stages()
        """
        return list(self._affected_stages)

    @abstractmethod
    def get_configuration_parameters(self) -> list[RuleConfigurationParam]:
        """Return the configuration parameters for this rule.

        Returns:
            List of configuration parameters.

        Example:

        .. code-block:: python

            params = rule.get_configuration_parameters()
        """
        raise NotImplementedError
