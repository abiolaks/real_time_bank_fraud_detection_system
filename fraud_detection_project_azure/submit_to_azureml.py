# submit_to_azureml.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command

# 1 Connect to workspace


ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# 2 Define environment (from environment.yml)
env = Environment(
    name="fraud-detection-v5-env",
    description="Environment for fraud detection training",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

# 3 Create the training job
# Define the job
job = command(
    code="./",  # folder containing your scripts
    command="python train.py",  # your training script
    environment=env,
    compute="alawani2",  # existing compute cluster
    experiment_name="fraud_detection_train_v5",
    display_name="bank-fraud-train-job_v5",
    description="Train fraud detection model on AzureML compute",
)

# Submit job
returned_job = ml_client.jobs.create_or_update(job)
print("Job submitted successfully!")
print(f"View progress in Azure ML Studio:\n{returned_job.studio_url}")
