import os
import logging
import wandb
from typing import Any

logger = logging.getLogger(__name__)


def log_artifact(name: str, type: str, description: str, path:str, run: Any):
    # Ensure that the file exists before logging
    if not os.path.isfile(path):
        raise ValueError(f"Path is not a valid file: {path}")
    # Create a new artifact
    artifact = wandb.Artifact(name, type, description=description)
    
    # Add the file to the artifact
    artifact.add_file(path)
    
    # Log the artifact to W&B
    run.log_artifact(artifact)
    logger.info(f"Logged artifact: {name}")

def _fixed_path():
    path = os.path.abspath(os.getcwd())
    keyword = "rental-prices/"
    result = path.split(keyword)[0] + keyword
    return result

def go(args):
    run = wandb.init(project="nyc_airbnb", job_type="upload_file")
    run.config.update(vars(args))  # If `args` is a Namespace, convert it to a dictionary.
    # Absolute file path
    
    # sample_file_path = os.path.join(os.cwd() ,"components", "get_data", "data", args.sample)

    sample_file_path = os.path.join(_fixed_path(), "components", "get_data", "data", args.sample)
    # sample_file_path = "components/get_data/data/sample1.csv"
    print(f"DEBUG: File path being logged: {sample_file_path}")  # Debug statement
    print(f"DEBUG: sample: {args.sample}")
    log_artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
        sample_file_path,
        run,
    )
    run.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload data to W&B")
    parser.add_argument("--sample", type=str, required=True, help="Sample file to upload")
    parser.add_argument("--artifact_name", type=str, required=True, help="Artifact name to upload")
    parser.add_argument("--artifact_type", type=str, required=True, help="Artifact type")
    parser.add_argument("--artifact_description", type=str, required=True, help="Description of the artifact")
    args = parser.parse_args()
    go(args)