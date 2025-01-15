from transformers import AutoTokenizer


def lambda_handler(event, context):
    input_ids = event["input_ids"]
    tokenizer = AutoTokenizer.from_pretrained(
        "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
    )
    text = tokenizer.batch_decode(input_ids)

    return {"generated_text": text}
