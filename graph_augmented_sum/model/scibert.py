import torch
from transformers import AutoModel, AutoTokenizer


class ScibertEmbedding:
    def __init__(self):
        self._model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')
        self._tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
        self._tokenizer.bos_token = '[unused1]'
        self._tokenizer.eos_token = '[unused2]'
        if torch.cuda.is_available():
            self._model.cuda()
        self._model.eval()

        # print('Roberta initialized')
        print('Bert initialized')
        self._pad_id = self._tokenizer.pad_token_id
        self._eos_id = self._tokenizer.eos_token_id
        self._bos_id = self._tokenizer.bos_token_id
        self._unk_id = self._tokenizer.unk_token_id
        self._cls_token = self._tokenizer.cls_token_id
        self._sep_token = self._tokenizer.sep_token_id
        self._embedding = self._model.embeddings.word_embeddings
        self._embedding.weight.requires_grad = False

    def __call__(self, input_ids):
        attention_mask = (input_ids != self._pad_id).float()
        return self._model(input_ids, attention_mask=attention_mask)
