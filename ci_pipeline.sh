file="DistilBERTModel.mar"
dir="model_store/"
debug=true


echo "Stopping the TorchServe"
torchserve --stop

if  $debug
then
    echo "Training Model on Debug"
    python train_distilBERT.py --debug
else
    echo "Training Model on Train"
    python train_distilBERT.py 
fi

echo "Removing mar file from Model Store"
rm -f "$dir$file"

echo "Creating new mar file and relocating to Model Store"
torch-model-archiver --model-name DistilBERTModel --version 1.0 --serialized-file traced_distilbert.pt --handler handler --extra-files handlerConfig.json 

mv  "$file" "$dir"

export LOG_FILE_PATH = /Users/felixolesen/Documents/University/Semester_two/NaturalLanguageProcessing/groupCW/NLP_GROUP_CW/logs

echo "Hosting server with new update model"
torchserve --start --model-store model_store --models DistilBERTModel="$file" 