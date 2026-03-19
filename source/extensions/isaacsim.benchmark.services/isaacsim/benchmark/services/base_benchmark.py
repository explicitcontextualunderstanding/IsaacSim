# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark base classes and helpers for performance measurement."""

import os
import tempfile
import time

import carb
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
import omni.usd
from isaacsim.benchmark.services import utils

# Import all recorders to trigger decorator registration (intentionally unused imports)
from isaacsim.benchmark.services.datarecorders import app_frametime  # noqa: F401
from isaacsim.benchmark.services.datarecorders import cpu_continuous  # noqa: F401
from isaacsim.benchmark.services.datarecorders import gpu_frametime  # noqa: F401
from isaacsim.benchmark.services.datarecorders import hardware  # noqa: F401
from isaacsim.benchmark.services.datarecorders import memory  # noqa: F401
from isaacsim.benchmark.services.datarecorders import physics_frametime  # noqa: F401
from isaacsim.benchmark.services.datarecorders import render_frametime  # noqa: F401
from isaacsim.benchmark.services.datarecorders import runtime  # noqa: F401
from isaacsim.benchmark.services.datarecorders import (
    InputContext,
    MeasurementDataRecorderRegistry,
)
from isaacsim.benchmark.services.datarecorders.interface import MeasurementDataRecorder
from isaacsim.benchmark.services.metrics import backend, measurements, report
from isaacsim.storage.native import get_assets_root_path, get_assets_root_path_async

from .utils import wait_until_stage_is_fully_loaded, wait_until_stage_is_fully_loaded_async

logger = utils.set_up_logging(__name__)


# Default recorders used if none specified
DEFAULT_RECORDERS = [
    "app_frametime",
    "physics_frametime",
    "runtime",
    "memory",
    "cpu_continuous",
    "hardware",
]


def set_sync_mode():
    """Enable synchronous USD and material loading.

    This adjusts Kit settings to block until assets and materials are fully
    loaded, which helps produce deterministic benchmark measurements.

    Example:

    .. code-block:: python

        set_sync_mode()
    """
    carb_settings = carb.settings.get_settings()
    if carb_settings.get("/app/asyncRendering"):
        logger.warning("Async rendering is enabled, setting sync mode might not work")
    carb_settings.set("/omni.kit.plugin/syncUsdLoads", True)
    carb_settings.set("/rtx-defaults/materialDb/syncLoads", True)
    carb_settings.set("/rtx-defaults/hydra/materialSyncLoads", True)


class _BaseIsaacBenchmarkCore:
    """Core benchmark functionality shared between sync and async implementations."""

    def _initialize_benchmark(
        self,
        benchmark_name: str,
        backend_type: str,
        report_generation: bool,
        workflow_metadata: dict | None,
        recorders: list[str] | None,
    ):
        """Initialize common benchmark state and recorders.

        Args:
            benchmark_name: Name of benchmark to use in outputs.
            backend_type: Type of backend used to collect and print metrics.
            report_generation: Whether to generate a formatted report.
            workflow_metadata: Metadata describing the benchmark workflow.
            recorders: List of recorder names to use, or None for defaults.
        """
        self.benchmark_name = benchmark_name
        self.report = report_generation
        self._test_phases: list[measurements.TestPhase] = []

        self.settings = carb.settings.get_settings()
        version, _, _ = utils.get_kit_version_branch()
        prefix = f"{benchmark_name}_{version}"

        self.context: InputContext | None = InputContext(
            artifact_prefix=prefix,
            kit_version=version,
            phase="benchmark",
        )

        # Use provided recorders or defaults
        if recorders is None:
            recorders = DEFAULT_RECORDERS.copy()

        # Instantiate recorders from registry
        self.recorders: list[MeasurementDataRecorder] | None = []
        self._active_recorders: set[MeasurementDataRecorder] = set()  # Track which recorders are currently collecting
        for recorder_name in recorders:
            recorder_class = MeasurementDataRecorderRegistry.get(recorder_name)
            if recorder_class:
                self.recorders.append(recorder_class(self.context))
                logger.info("Loaded recorder: %s", recorder_name)
            else:
                logger.warning("Unknown recorder: %s (skipped)", recorder_name)

        self._metrics_output_folder = self.settings.get(
            "/exts/isaacsim.benchmark.services/metrics/metrics_output_folder"
        )

        if not self._metrics_output_folder:
            self._metrics_output_folder = tempfile.gettempdir()

        # Create report
        if self.report:
            logger.info("Generating formatted report")
            self.final_report = report.Report()

        # Get metrics backend
        logger.info("Using metrics backend = %s", backend_type)
        self._metrics = backend.MetricsBackend.get_instance(instance_type=backend_type)

        # Generate workflow-level metadata
        self._metadata: list[measurements.MetadataBase] = [
            measurements.StringMetadata(name="workflow_name", data=self.benchmark_name)
        ]
        if workflow_metadata:
            if "metadata" in workflow_metadata:
                self._metadata.extend(measurements.TestPhase.metadata_from_dict(workflow_metadata))
            else:
                logger.warning(
                    "workflow_metadata provided, but missing expected 'metadata' entry. Metadata will not be read."
                )

        logger.info("Local folder location = %s", self._metrics_output_folder)
        logger.info("Starting")
        self.benchmark_start_time = time.time()
        self.test_mode = os.getenv("ISAAC_TEST_MODE") == "1"
        logger.info("Test mode = %s", self.test_mode)

    def set_phase(self, phase: str, start_recording_frametime: bool = True, start_recording_runtime: bool = True):
        """Set the active benchmarking phase and start recorders.

        Args:
            phase: Name of the phase, used in output.
            start_recording_frametime: False to skip frametime recorders.
            start_recording_runtime: False to skip runtime recorder.

        Raises:
            RuntimeError: If the benchmark context or recorders are not initialized.

        Example:

        .. code-block:: python

            benchmark.set_phase("loading", start_recording_frametime=False)
        """
        context = self.context
        if context is None:
            raise RuntimeError("Benchmark context is not initialized")
        recorders = self.recorders
        if recorders is None:
            raise RuntimeError("Recorders are not initialized")

        logger.info("Starting phase: %s", phase)
        context.phase = phase

        # Frametime recorders - only start if requested
        frametime_recorders = {
            "AppFrametimeRecorder",
            "PhysicsFrametimeRecorder",
            "GPUFrametimeRecorder",
            "RenderFrametimeRecorder",
        }

        # Always-on recorders - collect in every phase
        always_on_recorders = {
            "CPUContinuousRecorder",
            "MemoryRecorder",
            "HardwareSpecRecorder",
        }

        # Active recorders should already be cleared by store_measurements()
        # If not (e.g., set_phase called without store_measurements), clear them now
        if self._active_recorders:
            logger.warning(
                "Active recorders not cleared before new phase. "
                "Ensure store_measurements() is called before set_phase()."
            )
            self._active_recorders.clear()

        # Start appropriate recorders
        for recorder in recorders:
            recorder_name = recorder.__class__.__name__
            should_activate = False

            # Frametime recorders - conditional
            if recorder_name in frametime_recorders:
                if start_recording_frametime:
                    should_activate = True
                else:
                    logger.debug("Skipped %s for phase '%s' (frametime disabled)", recorder_name, phase)

            # Runtime recorder - conditional
            elif recorder_name == "RuntimeRecorder":
                if start_recording_runtime:
                    should_activate = True
                else:
                    logger.debug("Skipped %s for phase '%s' (runtime disabled)", recorder_name, phase)

            # Always-on recorders (including stateless ones like Memory and Hardware)
            elif recorder_name in always_on_recorders:
                should_activate = True

            # Custom recorders - always activate them
            else:
                should_activate = True

            # Activate recorder: start collecting if it has the method, otherwise just track it
            if should_activate:
                if hasattr(recorder, "start_collecting"):
                    recorder.start_collecting()
                    logger.debug("Started %s for phase '%s'", recorder_name, phase)
                else:
                    logger.debug("Activated stateless recorder %s for phase '%s'", recorder_name, phase)
                self._active_recorders.add(recorder)

    def _store_measurements_impl(self):
        """Stop active recorders and collect their data."""
        context = self.context
        if context is None:
            raise RuntimeError("Benchmark context is not initialized")
        # Stop only the recorders that were started in this phase
        for recorder in self._active_recorders:
            if hasattr(recorder, "stop_collecting"):
                recorder.stop_collecting()

        # Retrieve metrics, metadata, and artifacts only from active recorders
        run_measurements = []
        run_metadata = []
        run_artifacts = []
        for m in self._active_recorders:
            data = m.get_data()
            run_measurements.extend(data.measurements)
            run_metadata.extend(data.metadata)
            run_artifacts.extend(data.artefacts)

        # Create a new test phase to store these measurements
        test_phase = measurements.TestPhase(
            phase_name=context.phase, measurements=run_measurements, metadata=run_metadata
        )
        # Update test phase metadata with phase name and benchmark metadata
        test_phase.metadata.extend(self._metadata)
        test_phase.metadata.append(measurements.StringMetadata(name="phase", data=context.phase))
        self._test_phases.append(test_phase)

        # Clear active recorders after storing measurements
        self._active_recorders.clear()

    def _finalize_impl(self):
        """Finalize metrics collection and write output files."""
        if not os.path.exists(self._metrics_output_folder):
            os.mkdir(path=self._metrics_output_folder)

        randomize_filename_prefix = self.settings.get(
            "/exts/isaacsim.benchmark.services/metrics/randomize_filename_prefix"
        )

        logger.info("Stopping")

        if not self._test_phases:
            logger.warning(
                "No test phases collected. After set_phase(), store_measurements() should be called. "
                "No metrics will be written."
            )
            return

        logger.info("Writing metrics data.")
        logger.info("Metrics type = %s", type(self._metrics).__name__)

        # Finalize by adding all test phases to the backend metrics
        for test_phase in self._test_phases:
            self._metrics.add_metrics(test_phase)
            if self.report:
                self.final_report.add_metric_phase(test_phase)

        self._metrics.finalize(self._metrics_output_folder, randomize_filename_prefix)
        if self.report:
            self.final_report.create_report()

        self.test_run = None
        self.recorders = None
        self.context = None

    def _store_custom_measurement_impl(self, phase_name: str, custom_measurement: measurements):
        """Store a custom measurement for a specific phase.

        Args:
            phase_name: The phase name to which the measurement belongs.
            custom_measurement: The measurement object to store.
        """
        # Check if the phase already exists
        existing_phase = next((phase for phase in self._test_phases if phase.phase_name == phase_name), None)

        if existing_phase:
            # Add the custom measurement to the existing phase
            existing_phase.measurements.append(custom_measurement)
            logger.info("Stored %s for phase '%s'", custom_measurement, phase_name)
        else:
            # If the phase does not exist, create a new test phase
            new_test_phase = measurements.TestPhase(
                phase_name=phase_name, measurements=[custom_measurement], metadata=[]
            )
            # Update test phase metadata with phase name and benchmark metadata
            new_test_phase.metadata.extend(self._metadata)
            new_test_phase.metadata.append(measurements.StringMetadata(name="phase", data=phase_name))

            # Add the new test phase to the list of test phases
            self._test_phases.append(new_test_phase)

            logger.info("Created new phase '%s' and stored %s", phase_name, custom_measurement)


class BaseIsaacBenchmark(_BaseIsaacBenchmarkCore):
    """Benchmark class for standalone (synchronous) scripts.

    Args:
        benchmark_name: Name of benchmark to use in outputs.
        backend_type: Type of backend used to collect and print metrics.
        report_generation: Whether to generate a formatted report.
        workflow_metadata: Metadata describing benchmark.
        recorders: List of recorder names to use, or None for defaults.

    Example:

    .. code-block:: python

        benchmark = BaseIsaacBenchmark(benchmark_name="MyBenchmark", workflow_metadata={"metadata": []})
        benchmark.set_phase("loading")
        # load stage, configure sim, etc.
        benchmark.store_measurements()
        benchmark.set_phase("benchmark")
        # run benchmark
        benchmark.store_measurements()
        benchmark.stop()
    """

    def __init__(
        self,
        benchmark_name: str = "BaseIsaacBenchmark",
        backend_type: str = "OmniPerfKPIFile",
        report_generation: bool = True,
        workflow_metadata: dict | None = None,
        recorders: list[str] | None = None,
    ):
        set_sync_mode()

        self.assets_root_path = get_assets_root_path()
        if self.assets_root_path is None:
            logger.error("Could not find Isaac Sim assets folder")
            return

        self._initialize_benchmark(
            benchmark_name=benchmark_name,
            backend_type=backend_type,
            report_generation=report_generation,
            workflow_metadata=workflow_metadata or {},
            recorders=recorders,
        )

    def stop(self):
        """Stop benchmarking and write accumulated metrics to file.

        Example:

        .. code-block:: python

            benchmark.stop()
        """
        self._finalize_impl()

    def store_measurements(self):
        """Store measurements and metadata collected during the previous phase.

        Example:

        .. code-block:: python

            benchmark.store_measurements()
        """
        self._store_measurements_impl()

    def fully_load_stage(self, usd_path: str):
        """Load a USD stage and block until it is fully loaded.

        Args:
            usd_path: Path to USD stage.

        Example:

        .. code-block:: python

            benchmark.fully_load_stage("/path/to/scene.usd")
        """
        stage_utils.open_stage(usd_path)
        wait_until_stage_is_fully_loaded()

    def store_custom_measurement(self, phase_name: str, custom_measurement: measurements):
        """Store a custom measurement for the current benchmark.

        Args:
            phase_name: The phase name to which the measurement belongs.
            custom_measurement: The measurement object to store.

        Example:

        .. code-block:: python

            benchmark.store_custom_measurement("warmup", custom_measurement)
        """
        self._store_custom_measurement_impl(phase_name, custom_measurement)


class BaseIsaacBenchmarkAsync(_BaseIsaacBenchmarkCore, omni.kit.test.AsyncTestCase):
    """Benchmark class for async test cases.

    Example:

    .. code-block:: python

        class MyBenchmark(BaseIsaacBenchmarkAsync):
            async def setUp(self):
                await super().setUp()

            async def test_my_benchmark(self):
                self.set_phase("loading")
                await self.fully_load_stage("path/to/stage.usd")
                await self.store_measurements()

                self.set_phase("benchmark")
                # ... run benchmark ...
                await self.store_measurements()

            async def tearDown(self):
                await super().tearDown()
    """

    async def setUp(
        self,
        backend_type: str = "JSONFileMetrics",
        report_generation: bool = False,
        workflow_metadata: dict | None = None,
        recorders: list[str] | None = None,
    ):
        """Must be awaited by derived benchmarks to properly set up the benchmark.

        Args:
            backend_type: Type of backend used to collect and print metrics.
            report_generation: Whether to generate a formatted report.
            workflow_metadata: Metadata describing the benchmark workflow.
            recorders: List of recorder names to use, or None for defaults.
        """
        set_sync_mode()

        self.assets_root_path = await get_assets_root_path_async()
        if self.assets_root_path is None:
            logger.error("Could not find Isaac Sim assets folder")
            return

        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()

        # Use test method name as benchmark name for async tests
        benchmark_name = getattr(self, "_testMethodName", "AsyncBenchmark")

        self._initialize_benchmark(
            benchmark_name=benchmark_name,
            backend_type=backend_type,
            report_generation=report_generation,
            workflow_metadata=workflow_metadata or {},
            recorders=recorders,
        )

    async def tearDown(self):
        """Tear down the benchmark and finalize metrics."""
        # Wait for stage to finish loading
        while stage_utils.is_stage_loading():
            print("asset still loading, waiting to finish")
            await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        self._finalize_impl()

        await omni.kit.app.get_app().next_update_async()

    async def store_measurements(self):
        """Store measurements and metadata collected during the previous phase.

        Example:

        .. code-block:: python

            await benchmark.store_measurements()
        """
        self._store_measurements_impl()

    async def fully_load_stage(self, usd_path: str):
        """Open a stage and wait for it to fully load.

        Args:
            usd_path: Path to USD stage.

        Example:

        .. code-block:: python

            await benchmark.fully_load_stage("/path/to/scene.usd")
        """
        stage_utils.open_stage(usd_path)
        await wait_until_stage_is_fully_loaded_async()

    async def store_custom_measurement(self, phase_name: str, custom_measurement: measurements):
        """Store a custom measurement for the current benchmark.

        Args:
            phase_name: The phase name to which the measurement belongs.
            custom_measurement: The measurement object to store.

        Example:

        .. code-block:: python

            await benchmark.store_custom_measurement("warmup", custom_measurement)
        """
        self._store_custom_measurement_impl(phase_name, custom_measurement)
