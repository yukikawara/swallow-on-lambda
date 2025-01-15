import glob
import json
import os

import boto3
import torch
from transformers.cache_utils import DynamicCache

client = boto3.client("s3")


def remove_model():
    for p in glob.glob("/tmp/*.pt"):
        if os.path.isfile(p):
            os.remove(p)


def download_model(layer_key, model_path):
    client.download_file("kawara-swallow-lambda", layer_key, model_path)


def download_hidden_state(hidden_path, local_path):
    client.download_file("kawara-swallow-lambda", hidden_path, local_path)
    hidden_state = json.load(open(local_path))
    return hidden_state


def upload_hidden_state(embedding):
    upload_path = "hidden_state/12345.json"

    json.dump(embedding, open("/tmp/12345.json", "w"))

    client = boto3.client("s3")
    client.upload_file("/tmp/12345.json", "kawara-swallow-lambda", upload_path)

    return upload_path


def lambda_handler(event, context):
    layer_num = event["layer_num"]
    hidden_state_s3_path = event["hidden_state_s3_path"]
    input_ids = event["input_ids"]

    layer_key = f"decoder_layer_{layer_num}.pt"
    model_path = f"/tmp/decoder_layer_{layer_num}.pt"

    rotary_key = "rotary_emb.pt"
    rotary_path = "/tmp/rotary_emb.pt"

    remove_model()

    download_model(layer_key, model_path)
    download_model(rotary_key, rotary_path)
    hidden_states = download_hidden_state(
        hidden_state_s3_path, "/tmp/hidden_state.json"
    )
    hidden_states = torch.tensor(hidden_states)
    print(f"{hidden_states.shape=}")

    decoder_layer = torch.load(model_path)
    rotary_embedder = torch.load(rotary_path)

    past_key_values = DynamicCache()
    past_seen_tokens = past_key_values.get_seq_length()
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + hidden_states.shape[1]
    )
    position_ids = cache_position.unsqueeze(0)
    causal_mask = None
    position_embeddings = rotary_embedder(hidden_states, position_ids)
    with torch.no_grad():
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attenions=False,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    hidden_states = layer_outputs[0]

    s3_path = upload_hidden_state(hidden_states.cpu().tolist())

    return {
        "input_ids": input_ids,
        "hidden_state_s3_path": s3_path,
        "layer_num": layer_num + 1,
        "new_token_num": event["new_token_num"],
    }
