# NLP_GROUP_CW
Before Hosting the Server:
1. Ensure that OpenJDK 11 is installed on your machine.
2. Install all of the requirements in the 'requirements.txt' file.
3. Make sure you have the traced_distilbert.pt file as well.


COMMAND FOR MAKING MODEL ARCHIVE FILE (.mar): 
```console
torch-model-archiver --model-name DistilBERTModel --version 1.0 --serialized-file traced_distilbert.pt --handler handler
```

Next 'mkdir model_store' inside the folder, and move the new generated .mar file into that folder. 'mv DistilBERTModel.mar /model_store 

COMMAND FOR HOSTING MODEL: 
```console
torchserve --start --model-store model_store --models DistilBERTModel=DistilBERTModel.mar
```
