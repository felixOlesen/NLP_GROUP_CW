# NLP_GROUP_CW
Before Hosting the Server:
1. Ensure that OpenJDK 11 is installed on your machine.
2. Install all of the requirements in the 'requirements.txt' file.
3. For conda installations, the "environment.yml" file can be used to install the necessary packages.

### Running TorchServe:
#### With CI/CD Pipeline Script
```console
./ci_pipeline.sh
```

#### Manually:
##### Making Model Archive File (.mar): 
```console
torch-model-archiver --model-name DistilBERTModel --version 1.0 --serialized-file traced_distilbert.pt --handler handler
```
##### Move the produced Model Archive File to Model Store:
```console
mkdir model_store
mv DistilBERTModel.mar /model_store
```

##### Hosting the Model: 
```console
torchserve --start --model-store model_store --models DistilBERTModel=DistilBERTModel.mar --ts-config config.properties
```
