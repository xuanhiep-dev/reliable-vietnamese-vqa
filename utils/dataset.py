from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
from underthesea import word_tokenize
import re
import json
import torch
from omegaconf import OmegaConf
from lavis.common.registry import registry
from typing import List, Optional, Union
import transformers
from transformers.utils import TensorType
from lavis.processors.base_processor import BaseProcessor
from PIL import Image


class Process:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_processor = self.load_preprocess()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "vinai/bartpho-word", use_fast=False)

    def load_preprocess(self):
        config = OmegaConf.load(registry.get_model_class(
            name="blip2_feature_extractor").default_config_path(model_type="pretrain"))
        preprocess_cfg = config.preprocess

        def _build_proc_from_cfg(cfg):
            return (
                registry.get_processor_class(cfg.name).from_config(cfg)
                if cfg is not None
                else BaseProcessor()
            )
        vis_proc_cfg = preprocess_cfg.get("vis_processor")
        vis_eval_cfg = vis_proc_cfg.get("eval")
        vis_processors = _build_proc_from_cfg(vis_eval_cfg)
        return vis_processors

    def process_image(self, image):
        return self.image_processor(image.convert("RGB"))

    def process_text(self, text: Union[str, List[str], List[int]] = None, **kwargs):
        text = word_tokenize(text.lower(), format='text')
        text = self.tokenizer.encode_plus(text=text, **kwargs)
        return text

    def __call__(self, image, text=None, **kwargs):
        image = self.process_image(image)
        text = self.process_text(text, **kwargs)
        text['padding_mask'] = 1 - text['attention_mask']

        return {
            'image': image,
            'question': text['input_ids'],
            'padding_mask': text['padding_mask']
        }

    def process_punctuation(self, s):
        period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
        comma_strip = re.compile(r'(\d)(,)(\d)')
        punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!.')
        punctuation = re.compile(
            r'([{}])'.format(re.escape(punctuation_chars)))
        punctuation_with_a_space = re.compile(
            r'(?<= )([{0}])|([{0}])(?= )'.format(punctuation_chars))

        if punctuation.search(s) is None:
            return s
        s = punctuation_with_a_space.sub('', s)
        if re.search(comma_strip, s) is not None:
            s = s.replace(',', '')
        s = punctuation.sub(' ', s)
        s = period_strip.sub('', s)
        return s.strip()

    def preprocess_questions(self, df):
        questions = [word_tokenize(question.lower(), format='text')
                     for question in list(df['question'])]
        return questions

    def preprocess_answers(self, df):
        answers = [self.process_punctuation(answer.lower())
                   for answer in list(df['answer'])]
        return answers


class ViVQADataset(Dataset):
    def __init__(self, dataframe, processor, image_path, answers_path):
        self.dataframe = dataframe
        self.image_path = image_path
        self.processor = processor

        with open(answers_path, 'r') as f:
            self.vocab_a = json.load(f)['answer']
        self.labels = self.answers2idx(
            self.processor.preprocess_answers(dataframe), self.vocab_a)

    def answers2idx(self, answers, vocab_a):
        return [vocab_a[answer] for answer in answers]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        question = self.dataframe['question'].iloc[idx]
        img_id = self.dataframe['img_id'].iloc[idx]
        answer = self.dataframe['answer'].iloc[idx]
        label = self.labels[idx]

        image = Image.open(f'{self.image_path}/{img_id}.jpg')
        inputs = self.processor(image, question,
                                return_tensors='pt',
                                return_token_type_ids=False,
                                return_attention_mask=True,
                                truncation=True,
                                padding='max_length',
                                max_length=40)

        inputs.update({'labels': label, 'answers': answer})

        return inputs

    def get_sample_metadata(self, idx):
        return {
            "question": self.dataframe['question'].iloc[idx],
            "img_id": self.dataframe['img_id'].iloc[idx]
        }

    def get_vocab(self):
        return self.vocab_a


class ViVQAProcessor:
    def __init__(self, cfg):
        self.processor = Process()
        with open(cfg["ans_path"], 'r') as f:
            self.vocab = {v:  self.processor.process_punctuation(
                k.lower()) for k, v in json.load(f)["answer"].items()}

    def process_sample(self, image_path, question):
        image = Image.open(image_path)
        output = self.processor(image, question,
                                return_tensors='pt',
                                return_token_type_ids=False,
                                return_attention_mask=True,
                                truncation=True,
                                padding='max_length',
                                max_length=40)

        output["image"] = torch.tensor(output["image"]).unsqueeze(0)
        output["question"] = torch.tensor(output["question"])
        output["padding_mask"] = torch.tensor(output["padding_mask"])
        output['vocab'] = self.vocab_a

        return output
