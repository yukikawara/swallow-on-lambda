import json

import boto3
import torch

client = boto3.client("s3")


def download_model(model_key, model_path):
    client.download_file("kawara-swallow-lambda", model_key, model_path)


def upload_embedding(embedding):
    upload_path = "hidden_state/12345.json"

    json.dump(embedding, open("/tmp/12345.json", "w"))

    client.upload_file("/tmp/12345.json", "kawara-swallow-lambda", upload_path)

    return upload_path


def lambda_handler(event, context):
    model_path = "/tmp/embedder.pt"
    model_key = "embedder.pt"

    download_model(model_key, model_path)

    raw_input_ids = event["input_ids"]
    input_ids = torch.tensor(raw_input_ids)
    embedder = torch.load(model_path)
    with torch.no_grad():
        embed_inputs = embedder(input_ids)

    embed_inputs = embed_inputs.cpu().tolist()
    s3_path = upload_embedding(embedding=embed_inputs)

    return {
        "input_ids": raw_input_ids,
        "hidden_state_s3_path": s3_path,
        "layer_num": 0,
        "new_token_num": event["new_token_num"],
    }
