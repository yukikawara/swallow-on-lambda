FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ./

RUN python3.12 -m pip install -r requirements.txt -t .

COPY app.py ./

ENV HF_HOME=/tmp/.cache/huggingface
ENV HF_HUB_CACHE=/tmp/.cache/huggingface/hub

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
