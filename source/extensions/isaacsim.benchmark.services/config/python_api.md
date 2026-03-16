# Public API for module isaacsim.benchmark.services:

## Classes

- class BaseIsaacBenchmarkAsync(_BaseIsaacBenchmarkCore, omni.kit.test.AsyncTestCase)
  - async def setUp(self, backend_type: str = 'JSONFileMetrics', report_generation: bool = False, workflow_metadata: dict | None = None, recorders: list[str] | None = None)
  - async def tearDown(self)
  - async def store_measurements(self)
  - async def fully_load_stage(self, usd_path: str)
  - async def store_custom_measurement(self, phase_name: str, custom_measurement: measurements)

- class BaseIsaacBenchmark(_BaseIsaacBenchmarkCore)
  - def __init__(self, benchmark_name: str = 'BaseIsaacBenchmark', backend_type: str = 'OmniPerfKPIFile', report_generation: bool = True, workflow_metadata: dict | None = None, recorders: list[str] | None = None)
  - def stop(self)
  - def store_measurements(self)
  - def fully_load_stage(self, usd_path: str)
  - def store_custom_measurement(self, phase_name: str, custom_measurement: measurements)
