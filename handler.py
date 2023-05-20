import ast
import json
import logging
import os
from abc import ABC
#from torchserve import TorchServeLoggingHandler

import torch
import json
import transformers
from captum.attr import LayerIntegratedGradients
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    GPT2TokenizerFast,
)

from ts.torch_handler.base_handler import BaseHandler

with open('handlerConfig.json', 'r') as json_file:
    handlerConfig = json.load(json_file)
    handlerConfig = json.loads(handlerConfig)
            
workingDir = handlerConfig['extraConfig']['workingDir']

logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join(workingDir, 'logs/predictions.log'))
datefmt = "%Y-%m-%d || %H:%M:%S"
formatter = logging.Formatter('%(asctime)s.%(msecs)03d || %(message)s', datefmt)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# root_logger = logging.getLogger()

class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithm for Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # read configs for the mode, model_name, etc. from setup_config.json
        # setup_config_path = os.path.join(model_dir, "setup_config.json")
        # if os.path.isfile(setup_config_path):
        #     with open(setup_config_path) as setup_config_file:
        #         self.setup_config = json.load(setup_config_file)
        # else:
        #     logger.warning("Missing the setup_config.json file.")


        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.

        self.model = torch.jit.load(model_pt_path, map_location=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            'distilbert-base-cased',
            do_lower_case=True,
        )

        self.model.eval()

        # Read the mapping file, index to object name
        # with open('handlerConfig.json') as json_file:
        #     handlerConfig = json.load(json_file)
        #     handlerConfig = json.loads(self.handlerConfig)
        
        self.label_mappings = handlerConfig['labelMap']
        
        
        # Question answering does not need the index_to_name.json file.
        self.initialized = True

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's choice of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            
            max_length = 256
            self.input_text = input_text
            # preprocessing text for sequence_classification, token_classification or text_generation

            inputs = self.tokenizer.encode_plus(
                input_text,
                max_length=int(max_length),
                pad_to_max_length=True,
                #padding?
                add_special_tokens=True,
                return_tensors="pt",
                truncation=True,
            )

            # preprocessing text for question_answering.
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            # making a batch out of the recieved requests
            # attention masks are passed for cases where input tokens are padded.
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat(
                        (attention_mask_batch, attention_mask), 0
                    )
        return (input_ids_batch, attention_mask_batch)

    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """

        input_ids_batch, attention_mask_batch = input_batch
        inferences = []
        # Handling inference for sequence_classification.
        
        predictions = self.model(input_ids_batch, attention_mask_batch)
        logger.debug(
            "This the output size from the Seq classification model",
            predictions[0].size(),
        )
        logger.debug("This the output from the Seq classification model", predictions)

        #num_rows, num_cols = predictions[0].shape
        num_rows = predictions.shape[0]
    
        
        for i in range(num_rows):
            out = predictions[i].unsqueeze(0)
            y_hat = out.argmax(1).item()
            predicted_idx = str(y_hat)
            inferences.append(predicted_idx)
        # Handling inference for question_answering.
        
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """

        
        input_text = self.input_text.replace("\n", " ")
        
        labeledOutputs = [self.label_mappings[output] for output in inference_output]
        logger.info("%s || %s", input_text, labeledOutputs[0])
        
        return labeledOutputs

    