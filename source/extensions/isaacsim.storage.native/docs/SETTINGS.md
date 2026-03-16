```{csv-table}
**Extension**: {{ extension_version }},**Documentation Generated**: {sub-ref}`today`
```

# Settings

## Settings Provided by the Extension

### persistent.isaac.asset_root.default
   - **Default Value**: "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"
   - **Description**: Default asset root path for Isaac Sim.

### persistent.isaac.asset_root.timeout
   - **Default Value**: 5.0
   - **Description**: Timeout in seconds for asset root path to be resolved.

### persistent.isaac.asset_root.retry_attempts
   - **Default Value**: 3
   - **Description**: Number of retries for transient asset root connectivity checks.

### persistent.isaac.asset_root.retry_base_delay
   - **Default Value**: 0.5
   - **Description**: Initial retry delay in seconds using exponential backoff.
