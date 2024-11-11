#!/usr/bin/env python
"""
    Download dataset from W&B and clean it, and than upload as a new artifact
"""
import argparse
import logging
import os

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def clean(args):
    run_wandb = wandb.init(job_type="basic_cleaning")
    run_wandb.config.update(args)

    logger.info("Downloading Artifact")
    artifact_path = run_wandb.use_artifact(args.input_artifact).file()
    df_artic = pd.read_csv(artifact_path)

    # Drop outliers
    logger.info("Dropping outliers")
    idx = df_artic['price'].between(args.min_price, args.max_price)
    df_artic = df_artic[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting last_review to datetime")
    df_artic['last_review'] = pd.to_datetime(df_artic['last_review'])

    # Drop rows in the dataset that are not in the proper geolocation
    idx = df_artic['longitude'].between(-74.25, -73.50) & df_artic['latitude'].between(40.5, 41.2)
    df_artic = df_artic[idx].copy()

    # Save the cleaned dataset
    logger.info("Saving the output artifact")
    file_name = "clean_sample.csv"
    df_artic.to_csv(file_name, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(file_name)

    logger.info("Logging artifact")
    run_wandb.log_artifact(artifact)

    os.remove(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove rows that are not in the proper geolocation, ans also remove outliers")

    parser.add_argument(
        "--artifact_input",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_output",
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_output_type",
        type=str,
        help="Type of the artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_output_description",
        type=str,
        help="Description for the artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price for cleaning outliers",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price for cleaning outliers",
        required=True
    )

    args = parser.parse_args()

    clean(args)