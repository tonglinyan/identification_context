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
import torch
#from questeval.bertscore import BERTScore
from utils import (
    API_T2T,
    LinearizeDataInput,
    clean_table, 
    clean_obj,
    calculate_f1_squad, 
    calculate_BERTScore, 
    calculate_exact, 
)
import argparse
import sys
import collections
from nltk.translate.bleu_score import sentence_bleu

DIR = os.path.dirname("__file__")
__version__ = "0.2.4"

def parse_args():
    parser = argparse.ArgumentParser('Evaluation script for each model in questeval.')
    parser.add_argument('--task', choices=['QA_D2T', 'QG_D2T', 'QA_T2T', 'QG_T2T', 'IC'], help='prediction of question answering, question generation or identification of context')
    parser.add_argument('--author', choices=['hf', 'trained'], default='hf')
    parser.add_argument('--dataset', choices=['squad', 'sqa', 'wtq', 'webnlg'], help='datasets', default = 'squad')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--save_table', choices=[True, False], default=False)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

HF_ORGANIZATION = 'ThomasNLG'

class Evaluation:
    def __init__(
        self,
        args, 
        qg_batch_size: int = 36,
        clf_batch_size: int = 48,
        limit_sent: int = 5,
        no_cuda: bool = False,
    ) -> None:

        
        self.task = args.task
        self.author = args.author
        self.dataset = args.dataset
        self.limit_sent = limit_sent
        self.sep = '</s>'
        self.qg_prefix = None
        self.qg_batch_size = qg_batch_size
        self.clf_batch_size = clf_batch_size
        self.device = 'cuda' if (torch.cuda.is_available() and not no_cuda) else 'cpu'

        self.T2T_DATASETS = ['squad']
        self.D2T_DATASETS = ['sqa', 'wtq', 'webnlg']
        self.IC_DATASETS = ['squad']
        
        if 'T2T' in self.task and self.dataset not in self.T2T_DATASETS:
            raise (
                f'Dataset {self.dataset} is not available for model {self.task}. The list of available question-answering datasets are: {self.T2T_DATASETS}.'
            )

        if 'D2T' in self.task and self.dataset not in self.D2T_DATASETS:
            raise (
                f'Dataset {self.dataset} is not available for model {self.task}. The list of available question-answering datasets are: {self.D2T_DATASETS}.'
            )

        if 'IC' == self.task and self.dataset not in self.IC_DATASETS:
            raise (
                f'Dataset {self.dataset} is not available for model {self.task}. The list of available question-answering datasets are: {self.IC_DATASETS}.'
            )
        
        self._load_all_models()
        self._load_dataset()
        preds = self._prediction()
        self._save_json(preds)
        #with open(f'{self.task}_{author}.json', 'r') as f:
        #    data = json.load(f)

        #self.answer = [d['answer'] for d in data]
        #self.question = [d['question'] for d in data]
        #preds = [d['prediction'] for d in data]
        
        print(self._evaluation(preds))
        #self._save_json(preds, author)



    def _load_all_models(self):
        if self.author == 'hf':

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
                self.model = self.get_model(model_name=f'/home/tonglin.yan/identification_context/questeval/t5_qa_{self.dataset}_en')
            if self.task == 'QA_T2T':
                self.model = self.get_model(model_name='/home/tonglin.yan/identification_context/questeval/t5_qa_squad2neg_en')
            if self.task == 'QG_D2T':
                self.model = self.get_model(model_name=f'/home/tonglin.yan/identification_context/questeval/t5_qg_{self.dataset}_en')
            if self.task == 'QG_T2T':
                self.model = self.get_model(model_name='/home/tonglin.yan/identification_context/questeval/t5_qg_squad1_en')
            if self.task == 'IC':
                self.model = self.get_model(model_name='/home/tonglin.yan/identification_context/questeval/qa_inverse_squad')

    def _load_dataset(self):
        if 'T2T' in self.task:
            self.text, self.question, self.answer = self._load_squad()
        if self.dataset == 'wtq':
            self.text, self.question, self.answer = self._load_wtq()
        if self.dataset == 'sqa':
            self.text, self.question, self.answer = self._load_sqa()
        if self.dataset == 'webnlg':
            raw_datasets = load_dataset(
                'data/webnlg.py',
                data_files={"test":"webnlg_qgqa_test.json"},
            )
            test_set = raw_datasets['test']
            self.text, self.question, self.answer = test_set['context'], test_set['question'], test_set['answer']
        if self.task == 'IC':
            raw_datasets = load_dataset(
                'data/squad_cls.py',
                data_files={"validation":"dev-v1.1.json"},
            )
            test_set = raw_datasets['validation']
            self.sentence1, self.sentence2, self.label = test_set['sentence1'], test_set['sentence2'], test_set['label'],


    def _load_squad(self):
        raw_dataset = load_dataset('squad')['validation']
        context = raw_dataset['context']
        question = raw_dataset['question']
        answer = [a['text'][0] for a in raw_dataset['answers']]
        return context, question, answer
        

    def _load_sqa(self):
        dataset = load_dataset('msr_sqa')['test']

        position = dataset["position"]
        all_questions = dataset['question']
        all_answers = [', '.join(a) for a in dataset['answer_text']]
        all_headers = dataset['table_header']
        all_tables = dataset["table_data"]
        all_ids = dataset['id']
        
        #questions, headers, answers, tables = all_questions, all_headers, all_answers, all_tables
        questions, headers, answers, tables, ids = [], [], [], [], []
        for i in range(len(all_questions)):
            if position[i] == 0:
                #if all_answers[i] not in answers:
                questions.append(all_questions[i])
                answers.append(all_answers[i])
                headers.append(all_headers[i])
                tables.append(all_tables[i])
                ids.append(all_ids[i])

        def save_tables(header, rows, id):
            header = [clean_obj(h) for h in header]
            logs = []
            for r in rows:
                logs.append({h: clean_obj(v) for (h, v) in zip(header, r)})

            import csv   
            with open(f'data/{id}.csv', "w") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writerow({h:h for h in header})
                for log in logs:
                    w.writerow(log)

        tables = [clean_table(t, h) for t, h in zip(tables, headers)]
        texts = [LinearizeDataInput(t[0], t[1:]) for t in tables]

        #tables = [clean_table(t, h) for t, h in zip(all_tables, all_headers)]
        #texts = [LinearizeDataInput(t[0], t[1:]) for t in tables]

        if args.save_table:
            for t, id in zip(tables, all_ids):
                save_tables(t[0], t[1:], id)
        
  
        return texts, questions, answers

    
    def _load_wtq(
        self, 
    ):
        dataset = load_dataset('wikitablequestions')['test']
        questions = dataset['question']
        answers = [', '.join(a) for a in dataset['answers']]
        table = dataset['table']

        text = []
        for t in tqdm(table):
            text.append(LinearizeDataInput(t['header'], t['rows']))
        return text, questions, answers


    def _save_json(
        self, 
        preds: List
    ):
        data = []
        for c, q, a, p in zip(self.text, self.question, self.answer, preds):
            log = {"table":c, "question": q, "answer": a, "prediction": p}        
            data.append(log)

        with open(f'{self.task}_{self.dataset}_{self.author}.json', 'w') as f:
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
        
        if self.task == 'IC':
            preds =[]
            for idx in tqdm(range(0, len(self.label), batch_size)):
                pred = self._predict_label(self.sentence1, self.sentence2)
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
                em.append(calculate_exact(g, a))
                f1.append(calculate_f1_squad(g, a))

            total = len(em)
            return collections.OrderedDict([
                ('exact', 100.0 * sum(em) / total),
                ('f1', 100.0 * sum(f1) / total),
                ('total', total),
            ])

        elif 'QG' in self.task:
            gold = self.question

            em, f1, score_bleu, rouge, score_meteor, bert = [], [], [], [], [], []
            for g, a in zip(gold, preds):
                em.append(calculate_exact(g, a))
                f1.append(calculate_f1_squad(g, a))

            bleu = load_metric("bleu")
            meteor = load_metric("meteor")
            rouge = load_metric("rouge")
            metric_BERTScore = load_metric("bertscore")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            bert = calculate_BERTScore(preds, gold, metric_BERTScore, device)
                
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
                ('bert', 100 * sum(bert)/total), 
                ('total', total),
            ])

        else:
            return sum([1 if p==l else 0 for p, l in zip(preds, self.label)])/len(preds)


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


    def _predict_label(
        self, 
        st1: List[str],
        st2: List[str],
    ):
        preds = []    
        for i in range(0, len(st1), self.model.model_batch_size):
            input = [(s1, s2) for s1, s2 in zip(st1[i: i+self.model.model_batch_size], st2[i: i+self.model.model_batch_size])]
            preds += self.model.predict(input)
        return preds
        

    def get_model(self, model_name: str):
        keep_score_idx = None

        if 'qg' in model_name:
            self.qg_prefix = 'sv1'

        # batch size
        model_batch_size = self.qg_batch_size if 'qg' in model_name.lower() else self.clf_batch_size

        model = API_T2T(
            pretrained_model_name_or_path=model_name,
            keep_score_idx=keep_score_idx,
            max_source_length=512,
            model_batch_size=model_batch_size,
            device=self.device, 
            task=self.task, 
        )

        return model


if __name__ == '__main__':
    args = parse_args()
    questeval = Evaluation(args)

