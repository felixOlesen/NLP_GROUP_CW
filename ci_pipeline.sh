file="DistilBERTModel.mar"
dir="model_store/"

echo "Stopping the TorchServe"
torchserve --stop

echo "Training Model on Debug"
python train_distilBERT.py --debug

echo "Removing mar file from Model Store"
rm -f "$dir$file"

echo "Creating new mar file and relocating to Model Store"
torch-model-archiver --model-name DistilBERTModel --version 1.0 --serialized-file traced_distilbert.pt --handler handler --extra-files labelMap.json 

mv  "$file" "$dir"

echo "Hosting server with new update model"
torchserve --start --model-store model_store --models DistilBERTModel="$file"