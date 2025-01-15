import json

import boto3
import torch
from transformers import AutoTokenizer


client = boto3.client("s3")


def download_logits(logits_path, local_path):
    client.download_file("kawara-swallow-lambda", logits_path, local_path)
    hidden_state = json.load(open(local_path))
    return hidden_state


def lambda_handler(event, context):
    logits_s3_path = event["logits_s3_path"]
    input_ids = event["input_ids"]
    input_ids = torch.tensor(input_ids)

    logtis_path = "/tmp/logits.json"
    tokenizer = AutoTokenizer.from_pretrained(
        "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
    )
    logits = download_logits(logits_s3_path, logtis_path)

    next_tokens = torch.argmax(torch.tensor(logits), dim=-1)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

    next_tokens = next_tokens.cpu().tolist()
    is_end = next_tokens[0] == tokenizer.eos_token_id or event["new_token_num"] == 4

    return {
        "input_ids": input_ids.cpu().tolist(),
        "stop_generation": is_end,
        "new_token_num": event["new_token_num"] + 1,
    }
