# NLP_GROUP_CW

create model archive file: torch-model-archiver --model-name DistilBERTModel --version 1.0 --serialized-file traced_distilbert.pt --handler handler

host the server: torchserve --start --model-store model_store --models DistilBERTModel=DistilBERTModel.mar