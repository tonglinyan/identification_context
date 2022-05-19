from locale import normalize
import re 
from typing import List, Tuple, Dict, Callable
import os

#from sklearn.naive_bayes import BernoulliNB
import json
import numpy as np
import logging
from datasets import load_metric, load_dataset
import spacy
#from sympy import Q
import torch
from questeval import DIR, __version__
from tqdm import tqdm
#from questeval.bertscore import BERTScore
from questeval.utils import (
    API_T2T,
    LinearizeWTQInput
)
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser('Official evaluation script for SQuAD version 2.0.')
    parser.add_argument('task', choices=["QA_D2T", "QG_D2T", "QA_T2T", "QG_T2T"], help='prediction of QA or QG')
    parser.add_argument('author', choices=["hf", "trained"], default="hf")
    parser.add_argument('dataset', choices=["squad", "WTQ"], help='datasets', default = "squad")
    parser.add_argument('--out-file', '-o', metavar='eval.json',
                        help='Write accuracy metrics to file (default is stdout).')
    parser.add_argument('--verbose', '-v', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

HF_ORGANIZATION = "ThomasNLG"

class QuestEval:
    def __init__(
        self,
        task: str,
        author: str,
        dataset: str,
        qg_batch_size: int = 36,
        clf_batch_size: int = 48,
        limit_sent: int = 5,
        no_cuda: bool = False,
    ) -> None:

        self.task = task
        self.limit_sent = limit_sent
        self.sep = "</s>"
        self.qg_prefix = None
        self.qg_batch_size = qg_batch_size
        self.clf_batch_size = clf_batch_size
        self.device = 'cuda' if (torch.cuda.is_available() and not no_cuda) else 'cpu'

        if 'bertscore' in self.list_scores:
            #modifié
            #self.metric_BERTScore = BERTScore()
            self.metric_BERTScore = load_metric("bertscore")

        self._load_all_models(task, author)


        self.T2T_DATASETS = ["squad"]
        self.D2T_DATASETS = ["WTQ"]
        if 'T2T' in self.task and dataset not in self.T2T_DATASETS:
            raise (
                f"Dataset {dataset} is not available for model {self.task}. The list of available question-answering datasets are: {self.T2T_DATASETS}."
            )

        if 'D2T' in self.task and dataset not in self.D2T_DATASETS:
            raise (
                f"Dataset {dataset} is not available for model {self.task}. The list of available question-answering datasets are: {self.D2T_DATASETS}."
            )
        self._load_dataset(dataset)
        preds = self._prediction()
        self._save_json(preds)



    def _load_all_models(self, 
        author: str):
        if author == 'hf':

            if self.task == 'QA_T2T':
                self.model = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-qa_squad1-en')
            if self.task == 'QA_D2T':
                self.model = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-qa_webnlg_synth-en')
            if self.task == 'QG_T2T':
                self.model = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-qa_squad2neg-en')
            if self.task == 'QG_D2T':
                self.model = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-qg_webnlg_synth-en')

        else :

            if self.task == 'QA_D2T':
                self.model = self.get_model(model_name='/home/tonglin.yan/questeval/questeval/t5_qa_webnlg_en')
            if self.task == 'QA_T2T':
                self.model = self.get_model(model_name='/home/tonglin.yan/questeval/questeval/t5_qa_squad2neg_en')
            if self.task == 'QG_D2T':
                self.model = self.get_model(model_name='/home/tonglin.yan/questeval/questeval/t5_qg_webnlg_en')
            if self.task == 'QG_T2T':
                self.model = self.get_model(model_name='/home/tonglin.yan/questeval/questeval/t5_qa_squad1_en')


    def _load_dataset(self, 
        dataset:str,
    ):
        if 'T2T' in self.task:
            self.text, self.question, self.answer = self._load_squad()
        if 'D2T' in self.task:
             self.text, self.question, self.answer = self._load_wtq()


    def _load_squad(
        self,
    ):
        raw_dataset = load_dataset("squad")['train']
        context = raw_dataset["context"]
        question = raw_dataset["question"]
        answer = raw_dataset['answer']['text'][0]
        return context, question, answer

    def _load_wtq(
        self, 
    ):
        dataset = load_dataset("wikitablequestions")['train']
        question = dataset["question"]
        answer = dataset["answer"]
        table = dataset["table"]
        text = []
        for t in table:
            text.append(LinearizeWTQInput(t))
        return text, question, answer


    def _save_json(self, preds):
        data = []
        for i in range(len(preds)):
            d = {"context": self.text[i], 'question': self.question[i], 'answer': self.answer[i], 'prediction': preds[i]}
            data.append(d)

        with open("./prediction.json", "w") as f:
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
                pred = self._predict_answers(to_do_exs)
                preds += pred
            
        if 'QG' in self.task:
            preds = []
            for idx in tqdm(range(0, len(self.text), batch_size)):
                answers = self.answer[idx:idx+batch_size]
                texts = self.text[idx:idx+batch_size]
                to_do_exs = [(a, c) for a, c in zip(answers, texts)]
                pred = self._predict_questions(to_do_exs)
                preds += pred
        assert len(preds) == len(self.text)
        return preds


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

        formated_inputs = [f'{question} {self.sep} {context}' for question, context in to_do_exs]
        qa_scores, qa_texts = self.model_QA.predict(formated_inputs)

        return qa_scores, qa_texts


    def get_model(self, model_name: str):
        keep_score_idx = None

        if 't5' in model_name.lower():

            if "t5_qg_squad1_en" in model_name:
                # the default models were trained with this prefix 'sv1' and 'nqa' prefix on the two datasets
                self.qg_prefix = 'sv1'

            # batch size
            model_batch_size = self.qg_batch_size if "qg" in model_name.lower() else self.clf_batch_size

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


questeval = QuestEval()
questeval.corpus_questeval()
