import json

import boto3
import torch

client = boto3.client("s3")


def download_model(layer_key, model_path):
    client.download_file("kawara-swallow-lambda", layer_key, model_path)


def download_hidden_state(hidden_path, local_path):
    client.download_file("kawara-swallow-lambda", hidden_path, local_path)
    hidden_state = json.load(open(local_path))
    return hidden_state


def upload_hidden_state(embedding):
    upload_path = "logits/12345.json"

    json.dump(embedding, open("/tmp/12345.json", "w"))

    client = boto3.client("s3")
    client.upload_file("/tmp/12345.json", "kawara-swallow-lambda", upload_path)

    return upload_path


def lambda_handler(event, context):
    hidden_state_s3_path = event["hidden_state_s3_path"]

    norm_key = "norm.pt"
    norm_path = "/tmp/norm.pt"

    lm_head_key = "lm_head.pt"
    lm_head_path = "/tmp/lm_head.pt"

    download_model(norm_key, norm_path)
    download_model(lm_head_key, lm_head_path)
    hidden_states = download_hidden_state(
        hidden_state_s3_path, "/tmp/hidden_state.json"
    )
    hidden_states = torch.tensor(hidden_states)

    norm = torch.load(norm_path)
    lm_head = torch.load(lm_head_path)

    with torch.no_grad():
        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states[:, -1, :])

    s3_path = upload_hidden_state(logits.cpu().tolist())

    return {
        "logits_s3_path": s3_path,
        "input_ids": event["input_ids"],
        "new_token_num": event["new_token_num"],
    }
