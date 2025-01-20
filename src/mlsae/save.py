import json
import torch
import petname
from pathlib import Path

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from mlsae.data import DTYPES

S3_BUCKET = "deep-sae"
S3_PREFIX = "models"

# If your original model_dir is in model.py, either re-import or replicate it here:
model_dir = Path(__file__).parent / "checkpoints"

def get_pet_name() -> str:
    """
    Generates a random "pet" name, e.g. 'lively-spaniel', using the petname library.
    """
    # petname.generate() produces a random name like "fragrant-emu" by default
    return petname.generate(words=3)

def save_model(
    model,
    architecture_name: str,
    model_id: str | None = None,
    save_to_s3: bool = False,
):
    """
    Save model state_dict() and config to local disk, optionally push to S3.
    """

    # 1) If the user didn't supply a model_id, generate a pet name
    if model_id is None:
        model_id = get_pet_name()

    save_path = model_dir / architecture_name
    save_path.mkdir(exist_ok=True, parents=True)

    # 2) Save to local disk
    torch.save(model.state_dict(), save_path / f"{model_id}.pt")
    config_dict = {
        "encoder_dims": model.encoder_dims,
        "decoder_dims": model.decoder_dims,
        "sparse_dim": model.sparse_dim,
        "act_size": model.act_size,
        "enc_dtype": model.enc_dtype,
        "device": model.device,
        "l1_lambda": model.l1_lambda,
        "name": model.name,
    }
    with open(save_path / f"{model_id}_cfg.json", "w") as f:
        json.dump(config_dict, f)

    print(f"Saved model '{model_id}' for architecture '{architecture_name}' to disk.")

    # 3) Optionally save to S3
    if save_to_s3 and S3_BUCKET is not None:
        pt_local_path = str(save_path / f"{model_id}.pt")
        cfg_local_path = str(save_path / f"{model_id}_cfg.json")

        key_pt = f"{S3_PREFIX}/{architecture_name}/{model_id}.pt" if S3_PREFIX else f"{architecture_name}/{model_id}.pt"
        key_cfg = f"{S3_PREFIX}/{architecture_name}/{model_id}_cfg.json" if S3_PREFIX else f"{architecture_name}/{model_id}_cfg.json"

        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file(pt_local_path, S3_BUCKET, key_pt)
            s3_client.upload_file(cfg_local_path, S3_BUCKET, key_cfg)
            print(f"Uploaded '{model_id}' of {architecture_name} to S3 bucket '{S3_BUCKET}'.")
        except NoCredentialsError as e:
            print("No AWS credentials found. Could not upload to S3.")
            raise e
        except ClientError as e:
            print(f"Client error uploading to S3: {e}")
            raise e
        except Exception as e:
            print(f"Unexpected error uploading to S3: {e}")
            raise e

def load_model(
    cls,
    architecture_name: str,
    model_id: str | None = None,
    load_from_s3: bool = False,
):
    """
    Loads model from local disk or optionally downloads from S3 first.
    """
    load_path = model_dir / architecture_name
    load_path.mkdir(exist_ok=True, parents=True)

    # 1) Optionally download from S3 first
    if load_from_s3 and S3_BUCKET is not None:
        pt_local_path = str(load_path / f"{model_id}.pt")
        cfg_local_path = str(load_path / f"{model_id}_cfg.json")

        key_pt = f"{S3_PREFIX}/{architecture_name}/{model_id}.pt" if S3_PREFIX else f"{architecture_name}/{model_id}.pt"
        key_cfg = f"{S3_PREFIX}/{architecture_name}/{model_id}_cfg.json" if S3_PREFIX else f"{architecture_name}/{model_id}_cfg.json"

        s3_client = boto3.client('s3')
        try:
            s3_client.download_file(S3_BUCKET, key_pt, pt_local_path)
            s3_client.download_file(S3_BUCKET, key_cfg, cfg_local_path)
            print(f"Downloaded '{model_id}' of {architecture_name} from S3 bucket '{S3_BUCKET}'.")
        except NoCredentialsError:
            print("No AWS credentials found. Could not download from S3.")
        except ClientError as e:
            print(f"Client error downloading from S3: {e}")
        except Exception as e:
            print(f"Unexpected error downloading from S3: {e}")

    # If no id supplied, assert only one model
    if model_id is None:
        assert len(list(load_path.glob("*.pt"))) == 1, "More than one model found in directory"
        model_id = next(load_path.glob("*.pt")).stem

    # 2) Now load from local disk
    with open(load_path / f"{model_id}_cfg.json", "r") as f:
        config_dict = json.load(f)

    new_model = cls(
        encoder_dim_mults=[
            dim // config_dict["act_size"] for dim in config_dict["encoder_dims"]
        ],
        sparse_dim_mult=config_dict["sparse_dim"] // config_dict["act_size"],
        decoder_dim_mults=[
            dim // config_dict["act_size"] for dim in config_dict["decoder_dims"]
        ],
        act_size=config_dict["act_size"],
        enc_dtype=config_dict["enc_dtype"],
        device=config_dict["device"],
        l1_lambda=config_dict["l1_lambda"],
        name=config_dict["name"],
    )

    state_dict = torch.load(load_path / f"{model_id}.pt", map_location="cpu")
    new_model.load_state_dict(state_dict)
    new_model.to(new_model.dtype)

    print(f"Loaded model '{model_id}' for architecture '{architecture_name}' from disk.")
    return new_model
