# submit_to_azureml.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Environment,
    CommandJob,
    BuildContext,
)
from azure.identity import DefaultAzureCredential

# 1 Connect to workspace
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# 2 Define environment (from environment.yml)
env = Environment(
    name="fraud-detection-env",
    description="Environment for fraud detection training",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)

# 3 Create the training job
job = CommandJob(
    code="./",  # your project root directory
    command="python train_and_register.py",  # training entrypoint
    environment=env,
    compute="cpu-cluster",  # your AzureML compute name
    display_name="fraud-detection-train-job",
    experiment_name="fraud-detection-training",
)

# 4 Submit job
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted! View progress in Azure ML Studio:\n{returned_job.studio_url}")
