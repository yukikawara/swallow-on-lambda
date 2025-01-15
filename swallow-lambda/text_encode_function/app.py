from transformers import AutoTokenizer


def text_to_prompt(text, tokenizer):
    message = [
        {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。"},
        {
            "role": "user",
            "content": text,
        },
    ]
    return tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )


def lambda_handler(event, context):
    text = event["text"]
    tokenizer = AutoTokenizer.from_pretrained(
        "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
    )
    prompt = text_to_prompt(text, tokenizer)

    input_ids = tokenizer.encode(prompt)

    return {"input_ids": [input_ids], "new_token_num": 1}
