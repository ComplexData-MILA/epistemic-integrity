from transformers import AutoTokenizer, AutoModelForSequenceClassification
from certainty_estimator.edited_bert import BertModel, BertForSequenceClassification
import torch
from torch import Tensor
import numpy as np
import math
from accelerate import Accelerator


class CertaintyEstimator(object):
    def __init__(self, task='sentence-level', use_auth_token=False):

        self.task = task

        if task == 'sentence-level':
            model_path = 'pedropei/sentence-level-certainty'
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, num_labels=1, output_attentions=False,
                                                         output_hidden_states=False, cache_dir='./model_cache', use_auth_token=use_auth_token)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1,
                                                                           output_attentions=False, output_hidden_states=False,
                                                                           cache_dir='./model_cache', use_auth_token=use_auth_token)
        elif task == 'aspect-level':
            model_path = 'pedropei/aspect-level-certainty'
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, num_labels=3, output_attentions=False,
                                                         output_hidden_states=False, cache_dir='./model_cache', use_auth_token=use_auth_token)
            self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3, output_attentions=False,
                                                                       output_hidden_states=False, cache_dir='./model_cache', use_auth_token=use_auth_token)

        # Use Accelerate to manage multiple GPUs
        self.accelerator = Accelerator()
        self.model = self.accelerator.prepare_model(self.model)

    def data_iterator(self, train_x, batch_size):
        n_batches = math.ceil(len(train_x) / batch_size)
        for idx in range(n_batches):
            x = train_x[idx * batch_size:(idx + 1) * batch_size]
            yield x

    def padding(self, text, pad, max_len=512):
        return text if len(text) >= max_len else (text + [pad] * (max_len - len(text)))

    def encode_batch(self, text):

        tokenizer = self.tokenizer
        t1 = []
        for line in text:
            t1.append(self.padding(tokenizer.encode(line, add_special_tokens=True, max_length=512, truncation=True),
                                  tokenizer.pad_token_id))

        return t1

    def predict_sentence_level(self, text, batch_size=512, tqdm=None):
        if type(text) == str:
            text = [text]

        test_iterator = self.data_iterator(text, batch_size)
        all_preds = []

        if tqdm:
            # tqdm is not yet supported with Accelerate, you can implement your own progress tracking
            # for x in tqdm(test_iterator, total=int(len(text) / batch_size)):
            for x in tqdm(test_iterator, total=int(len(text) / batch_size)):
                ids = self.encode_batch(x)

                with torch.no_grad():
                    self.model.to(self.accelerator.device)  # Move model to the device
                    inputs = {"input_ids": Tensor(ids).long().to(self.accelerator.device)}  # Move input to device
                    outputs = self.model(**inputs)

                predicted = outputs[0].cpu().data.numpy()
                all_preds.extend(predicted)

            all_res = np.array(all_preds).flatten()
            return list(all_res)
        else:
            for x in test_iterator:

                ids = self.encode_batch(x)

                with torch.no_grad():
                    inputs = self.accelerator.prepare_batch({"input_ids": Tensor(ids).long()})
                    outputs = self.model(**inputs)

                predicted = outputs[0].cpu().data.numpy()
                all_preds.extend(predicted)

            all_res = np.array(all_preds).flatten()
            return list(all_res)

    def predict_aspect_level(self, text, get_processed_output, batch_size=1024, tqdm=None):
        return None

    def predict(self, text, get_processed_output=True, batch_size=512, tqdm=None):
        if self.task == 'sentence-level':
            return self.predict_sentence_level(text, batch_size, tqdm)
        else:
            return self.predict_aspect_level(text, get_processed_output, batch_size, tqdm)
