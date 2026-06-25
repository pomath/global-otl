"""Cloud pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CloudConfig:
    """Settings for running the gotl pipeline on Coiled (Dask on AWS).

    Attributes:
        s3_bucket: S3 bucket for staging data and results.
        s3_prefix: Key prefix within the bucket for all cloud runs.
        aws_region: AWS region for S3 and Coiled workers.
        use_spot: Use spot instances to reduce cost (~70% savings).
        worker_vm_type: EC2 instance type for workers.
        n_workers: Number of parallel workers.
        idle_timeout: Shut down workers after this idle period.
        software_env_name: Name of the Coiled software environment.
        tide_models: Tide models to process (passed to Config on workers).
        force: Re-run all pipeline steps even if cached.
    """

    s3_bucket: str
    s3_prefix: str = "cloud-runs"
    aws_region: str = "us-east-1"
    use_spot: bool = True
    worker_vm_type: str = "m6i.large"
    n_workers: int = 3
    idle_timeout: str = "5 minutes"
    software_env_name: str = "gotl-env"
    tide_models: list[str] = field(default_factory=lambda: ["TPXO9"])
    force: bool = False
