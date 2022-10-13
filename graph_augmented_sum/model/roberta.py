import torch
from transformers import RobertaTokenizer, RobertaModel


class RobertaEmbedding(object):
    def __init__(self, model='roberta-base'):
        self._model = RobertaModel.from_pretrained(model)
        self._tokenizer = RobertaTokenizer.from_pretrained(model)
        #self._model = BertModel.from_pretrained(model)
        #self._tokenizer = BertTokenizer.from_pretrained(model)
        if torch.cuda.is_available():
            self._model.cuda()
        self._model.eval()

        #print('Roberta initialized')
        print('Bert initialized')
        self._pad_id = self._tokenizer.pad_token_id
        self._eos_id = self._tokenizer.eos_token_id
        self._bos_id = self._tokenizer.bos_token_id
        self._unk_id = self._tokenizer.unk_token_id
        self._cls_token = self._tokenizer.cls_token
        self._sep_token = self._tokenizer.sep_token
        self._embedding = self._model.embeddings.word_embeddings
        self._embedding.weight.requires_grad = False

    def __call__(self, input_ids):
        attention_mask = (input_ids != self._pad_id).float()
        return self._model(input_ids, attention_mask=attention_mask)
