from locale import normalize
import re 
from typing import List, Tuple, Dict, Callable
import os
import unidecode
#from sklearn.naive_bayes import BernoulliNB
import json
#import numpy as np
#import logging
from datasets import load_dataset, load_metric
#import spacy
#from sympy import Q
import torch
from questeval import DIR, __version__
from tqdm import tqdm
#from questeval.bertscore import BERTScore
from utils import (
    API_T2T,
    LinearizeWTQInput,
)
import argparse
import sys
import collections
import string
from nltk.translate.bleu_score import sentence_bleu

DIR = os.path.dirname("__file__")
__version__ = "0.2.4"

def parse_args():
    parser = argparse.ArgumentParser('Evaluation script for each model in questeval.')
    parser.add_argument('--task', choices=['QA_D2T', 'QG_D2T', 'QA_T2T', 'QG_T2T'], help='prediction of QA or QG')
    parser.add_argument('--author', choices=['hf', 'trained'], default='hf')
    parser.add_argument('--dataset', choices=['squad', 'sqa', 'wtq'], help='datasets', default = 'squad')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--training_set', choices=['squad', 'sqa', 'wtq', 'webnlg'], help='dataset used for training the model', default='squad')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
    

HF_ORGANIZATION = 'ThomasNLG'

class QuestEval:
    def __init__(
        self,
        task: str,
        author: str,
        dataset: str,
        training_set: str,
        qg_batch_size: int = 36,
        clf_batch_size: int = 48,
        limit_sent: int = 5,
        no_cuda: bool = False,
    ) -> None:

        
        self.task = task
        self.author = author
        self.training_set = training_set
        self.limit_sent = limit_sent
        self.sep = '</s>'
        self.qg_prefix = None
        self.qg_batch_size = qg_batch_size
        self.clf_batch_size = clf_batch_size
        self.device = 'cuda' if (torch.cuda.is_available() and not no_cuda) else 'cpu'

        self.T2T_DATASETS = ['squad']
        self.D2T_DATASETS = ['sqa', 'wtq']
        self.TRAIN_DATASETS = ['sqa', 'wtq', 'webnlg']
        
        if 'T2T' in self.task and dataset not in self.T2T_DATASETS:
            raise (
                f'Dataset {dataset} is not available for model {self.task}. The list of available question-answering datasets are: {self.T2T_DATASETS}.'
            )

        if 'D2T' in self.task and dataset not in self.D2T_DATASETS:
            raise (
                f'Dataset {dataset} is not available for model {self.task}. The list of available question-answering datasets are: {self.D2T_DATASETS}.'
            )

        if 'D2T' in self.task and training_set not in self.TRAIN_DATASETS:
            raise (
                f'Model Data2Text trained by {dataset} is not available. The list of available models are: {self.TRAIN_DATASETS}.'
            )
        
        self._load_all_models(author)
        self._load_dataset(dataset)
        preds = self._prediction()
        self._save_json(preds, author)
        #with open(f'{self.task}_{author}.json', 'r') as f:
        #    data = json.load(f)

        #self.answer = [d['answer'] for d in data]
        #self.question = [d['question'] for d in data]
        #preds = [d['prediction'] for d in data]
        
        print(self._evaluation(preds))
        #self._save_json(preds, author)



    def _load_all_models(self, 
        author: str):
        if author == 'hf':

            if self.task == 'QA_T2T':
                self.model = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-qa_squad2neg-en')
            if self.task == 'QA_D2T':
                self.model = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-qa_webnlg_synth-en')
            if self.task == 'QG_T2T':
                self.model = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-qg_squad1-en')
            if self.task == 'QG_D2T':
                self.model = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-qg_webnlg_synth-en')

        else :

            if self.task == 'QA_D2T':
                self.model = self.get_model(model_name=f'/home/tonglin.yan/identification_context/t5_qa_{self.training_set}_en')
            if self.task == 'QA_T2T':
                self.model = self.get_model(model_name='/home/tonglin.yan/identification_context/t5_qa_newsqa_en')
            if self.task == 'QG_D2T':
                self.model = self.get_model(model_name=f'/home/tonglin.yan/identification_context/t5_qg_{self.training_set}_en')
            if self.task == 'QG_T2T':
                self.model = self.get_model(model_name='/home/tonglin.yan/identification_context/t5_qg_squad1_en')


    def _load_dataset(self, 
        dataset:str,
    ):
        if 'T2T' in self.task:
            self.text, self.question, self.answer = self._load_squad()
        if dataset == 'wtq':
            self.text, self.question, self.answer = self._load_wtq()
        if dataset == 'sqa':
            self.text, self.question, self.answer = self._load_sqa()


    def _load_squad(
        self,
    ):
        raw_dataset = load_dataset('squad')['validation']
        context = raw_dataset['context']
        question = raw_dataset['question']
        answer = [a['text'][0] for a in raw_dataset['answers']]
        return context, question, answer


    def _load_sqa(
        self, 
    ):
        dataset = load_dataset('msr_sqa')['test']
        questions = dataset['question']
        answers = [', '.join(a) for a in dataset['answer_text']]
        table = dataset['table_data']
        header = dataset['table_header']

        positions = dataset["position"]
        domain = [answers[i-1] if positions[i] != 0 else " " for i in range(1, len(answers))]
        domain = [" "] + domain

        def linearization(header, data):

            def clean_obj(
                s,
                lc: bool = False
            ):
                s = unidecode.unidecode(s)
                if lc: s = s.lower()
                s = re.sub('^"|"$', "", s)  # remove useless quotesigns
                s = re.sub('_', ' ', s)  # turn underscores to spaces
                return s

            entites = {h: [] for h in header}
            context = []
            for r in data:
                for h, v in zip(header, r):
                    entites[h].append(clean_obj(v))
            for h in header:
                e = ", ".join(entites[h])
                context.append(f'{clean_obj(h)} [ {e} ]')

            return '; '.join(context)

        texts = [linearization(h, d) for h, d in zip(header, table)]

        context = [f'{t} Domain [ {d} ]' for t, d in zip(texts, domain)] 
        res = context if "QA" in self.task else texts    
        return res, questions, answers

    
    def _load_wtq(
        self, 
    ):
        dataset = load_dataset('wikitablequestions')['test']
        questions = dataset['question']
        answers = [', '.join(a) for a in dataset['answers']]
        table = dataset['table']

        text = []
        for t in tqdm(table):
            text.append(LinearizeWTQInput(t['header'], t['rows']))
        return text, questions, answers


    def _save_json(
        self, 
        preds: List,
        author: str
    ):
        data = []
        for c, q, a, p in zip(self.text, self.question, self.answer, preds):
            log = {"context":c, "question": q, "answer": a, "prediction": p}
            data.append(log)

        with open(f'{self.task}_{self.training_set}_{author}.json', 'w') as f:
            json.dump(data, f, indent=4)


    def _prediction(
        self,
        batch_size: int=128
    ) -> List[str]:

        if 'QA' in self.task:
            preds = []
            for idx in tqdm(range(0, len(self.text), batch_size)):
                questions = self.question[idx:idx+batch_size]
                texts = self.text[idx:idx+batch_size]
                to_do_exs = [(q, c) for q, c in zip(questions, texts)]
                score, pred = self._predict_answers(to_do_exs)
                assert len(pred) == len(to_do_exs)
                preds += pred
            
        if 'QG' in self.task:
            preds = []
            for idx in tqdm(range(0, len(self.text), batch_size)):
                answers = self.answer[idx:idx+batch_size]
                texts = self.text[idx:idx+batch_size]
                to_do_exs = [(a, c) for a, c in zip(answers, texts)]
                pred = self._predict_questions(to_do_exs)
                assert len(pred) == len(to_do_exs)
                preds += pred
        
        return preds

    def _evaluation(
        self, 
        preds: List,
    ):
        if 'QA' in self.task:
            gold = self.answer
        
            em, f1 = [], []
            for g, a in zip(gold, preds):
                em.append(compute_exact(g, a))
                f1.append(compute_f1(g, a))

            total = len(em)
            return collections.OrderedDict([
                ('exact', 100.0 * sum(em) / total),
                ('f1', 100.0 * sum(f1) / total),
                ('total', total),
            ])

        else:
            gold = self.question

            em, f1, score_bleu, rouge, score_meteor = [], [], [], [], []
            for g, a in zip(gold, preds):
                em.append(compute_exact(g, a))
                f1.append(compute_f1(g, a))

            bleu = load_metric("bleu")
            meteor = load_metric("meteor")
            rouge = load_metric("rouge")
                
            score_bleu = bleu.compute(predictions=[p.split() for p in preds], references=[[g.split()] for g in gold])["bleu"]
            score_meteor = meteor.compute(predictions=preds, references=gold)["meteor"]
            score_rouge = rouge.compute(predictions=preds, references=gold)["rougeL"]

            total = len(em)
            return collections.OrderedDict([
                ('exact', 100.0 * sum(em) / total),
                ('f1', 100.0 * sum(f1) / total),
                ('bleu', score_bleu * 100), 
                ('meteor', score_meteor * 100), 
                ('rouge', score_rouge), 
                ('total', total),
            ])


    def _predict_questions(
        self,
        to_do_exs: List[tuple]
    ) -> List[str]:
        str_prefix = f'{self.qg_prefix} {self.sep} ' if self.qg_prefix is not None else ''
        formated_inputs = [f'{str_prefix}{asw} {self.sep} {context}' for asw, context in to_do_exs]
        _, question_texts = self.model.predict(formated_inputs)

        return question_texts

    
    def _predict_answers(
        self,
        to_do_exs: List[tuple]
    ) -> Tuple[List[float], List[str]]:
        if self.author == 'hf':
            formated_inputs = [f'{question} {self.sep} {context}' for question, context in to_do_exs]
        else:
            formated_inputs = [f'question:{question}context:{context}' for question, context in to_do_exs]
        qa_scores, qa_texts = self.model.predict(formated_inputs)

        return qa_scores, qa_texts


    def get_model(self, model_name: str):
        keep_score_idx = None

        if 't5' in model_name.lower():

            if 't5_qg_squad1_en' in model_name:
                # the default models were trained with this prefix 'sv1' and 'nqa' prefix on the two datasets
                self.qg_prefix = 'sv1'

            # batch size
            model_batch_size = self.qg_batch_size if 'qg' in model_name.lower() else self.clf_batch_size

            model = API_T2T(
                pretrained_model_name_or_path=model_name,
                keep_score_idx=keep_score_idx,
                max_source_length=512,
                model_batch_size=model_batch_size,
                device=self.device
            )

        else:
            raise NotImplementedError(f'Model Name Not Handled: the model name should contain t5 ({model_name}).')

        return model


if __name__ == '__main__':
    args = parse_args()
    questeval = QuestEval(args.task, args.author, args.dataset, args.training_set)

