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
    sentencize,
    calculate_f1_squad,
    calculate_BERTScore,
    extract_table_answers,
    text2hash, 
    normalize_answer
    #_load_webnlg
)
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser('Official evaluation script for SQuAD version 2.0.')
    parser.add_argument('model', choices=["QA_D2T", "QG_D2T", "QA_T2T", "QG_T2T"], help='prediction of QA or QG')
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
        model: str,
        author: str,
        dataset: str,
        qg_batch_size: int = 36,
        clf_batch_size: int = 48,
        limit_sent: int = 5,
        no_cuda: bool = False,
    ) -> None:
        """
        Main class for the QuestEval metric

        Args:
            task (:str):
                the task to evaluate with QuestEval

        Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """
        """
        format for the json logs:
            hash(txt) #json file name
                {
                'triple': "triple"
                'references': [{'text': "reference text" 
                                'answers': [answers],
                                'questions': [questions],
                                }, 
                                {'text': "reference text" 
                                'answers': [answers],
                                'questions': [questions],
                                }, 
                                ...
                              ]
                }
        """
        
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

        self._load_all_models(model, author)


        self.T2T_DATASETS = ["squad"]
        self.D2T_DATASETS = ["WTQ"]
        if 'T2T' in model and dataset not in self.T2T_DATASETS:
            raise (
                f"Dataset {dataset} is not available for model {model}. The list of available question-answering datasets are: {self.T2T_DATASETS}."
            )

        if 'D2T' in model and dataset not in self.D2T_DATASETS:
            raise (
                f"Dataset {dataset} is not available for model {model}. The list of available question-answering datasets are: {self.D2T_DATASETS}."
            )
        self._load_datasets(model, dataset)



    def _load_all_models(self, 
        model: str, 
        author: str):
        if author == 'hf':

            if model == 'QA_T2T':
                self.model = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-qa_squad1-en')
            if model == 'QA_D2T':
                self.model = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-qa_webnlg_synth-en')
            if model == 'QG_T2T':
                self.model = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-qa_squad2neg-en')
            if model == 'QG_D2T':
                self.model = self.get_model(model_name=f'{HF_ORGANIZATION}/t5-qg_webnlg_synth-en')

        else :

            if model == 'QA_D2T':
                self.model = self.get_model(model_name='/home/tonglin.yan/questeval/questeval/t5_qa_webnlg_en')
            if model == 'QA_T2T':
                self.model = self.get_model(model_name='/home/tonglin.yan/questeval/questeval/t5_qa_squad2neg_en')
            if model == 'QG_D2T':
                self.model = self.get_model(model_name='/home/tonglin.yan/questeval/questeval/t5_qg_webnlg_en')
            if model == 'QG_T2T':
                self.model = self.get_model(model_name='/home/tonglin.yan/questeval/questeval/t5_qa_squad1_en')


    def _load_dataset(self, 
        model: str,
        dataset:str,
        ):
        if 'T2T' in model:
            self.text, self.question, self.answer = _load_squad()
        if 'D2T' in model:
             self.text, self.question, self.answer = _load_wtq()


    def corpus_questeval(
        self,
        batch_size: int = 512
    ) -> Dict:

        logs = []
        d_loaded_logs = dict()


        """logs, hyp_hashes, modified_logs = self._texts2logs(type_logs='hyp', d_loaded_logs=d_loaded_logs)

        if modified_logs: 
            self.is_question(logs)
            self.answer_filtering(logs)
            self._save_json(logs)
            print(modified_logs)"""


        with open("webnlg_qgqa.json") as f:
            logs = json.load(f)
        self._save_json(logs)

        return 


    def save_json(self, logs, file_name='webnlg_qgqa.json'):

        with open(file_name, "w") as f:
            json.dump(logs, f, indent=4)


    def _save_json(self, logs):
        data = []
        for log in logs:
            answers, questions = [], []
            for ref in log["references"]:
                for a, q in zip(ref['answers'], ref['questions']):
                    if not (a in answers and q in questions):                
                        answers.append(a)
                        questions.append(q)
                        d = {"triplet": log["triple"], "context": log["triple_linearized"], 'question': q, 'answer': a}
                        data.append(d)
        
        import random
        random.shuffle(data)

        with open("./webnlg_qgqa_train.json", "w") as f:
            json.dump(data[:int(0.6*len(data))], f, indent=4)

        with open("./webnlg_qgqa_dev.json", "w") as f:
            json.dump(data[int(0.6*len(data)):int(0.8*len(data))], f, indent=4)

        with open("./webnlg_qgqa_test.json", "w") as f:
            json.dump(data[int(0.8*len(data)):], f, indent=4)



    def _texts2logs(
        self,
        type_logs: str,
        d_loaded_logs: Dict,
        batch_size: int=128
    ):
        modified_logs = False

        logs, logs_hashes = self._load_logs(d_loaded_logs)

        for idx in tqdm(range(0, len(logs), batch_size)):
            # Selecting the answers (bool, max acts as or)
            modified_logs = max(self._compute_answer_selection(logs[idx:idx+batch_size], type_logs), modified_logs)
            # Generating the questions (bool, max acts as or)
            modified_logs = max(self._compute_question_generation(logs[idx:idx+batch_size], type_logs), modified_logs)
            self.save_json(logs)
        return logs, logs_hashes, modified_logs


    def _load_logs(
        self,
        d_loaded_logs: Dict
    ) -> Tuple[List[Dict], List[str]]:

        raw_datasets = load_dataset("web_nlg", "webnlg_challenge_2017")
        logs, log_hashs = [], []
        
        for type in ['train', 'dev', 'test']:
            for data in tqdm(raw_datasets[type]):
                texts = data['lex']['text']
                
                triples = data['modified_triple_sets']['mtriple_set'][0]

                from questeval.utils import LinearizeWebnlgInput
                self.src_preproc_pipe = LinearizeWebnlgInput(spacy_pipeline=self.spacy_pipeline)
                linearized = self.src_preproc_pipe(triples)

                log_hash = text2hash(linearized) # hashcode of a text
                if log_hash not in d_loaded_logs:
                    log = {'triple': triples, 'triple_linearized': linearized, 'references': list()}

                    for text in texts:
                        ref = {"text": text, 'questions': list(), 'answers': list()}
                        log['references'].append(ref)
                        
                    if not (self.use_cache and log_hash in self.hash_files and linearized != ""):
                        temp=1

                    if self.use_cache and log_hash in self.hash_files and linearized != "":
                        cached_path = os.path.join(self.log_dir, log_hash)
                        try:
                            with open(cached_path, 'r') as f_log:
                                tmp  = json.load(f_log)
                                assert all([k in log for k in ['triple', 'references']])
                                assert isinstance(log["triple"], list)
                                assert isinstance(log["triple_linearized"], str)
                                assert isinstance(log['references'], list)
                                assert len(log['references']) == len(texts)
                                log = tmp
                        except json.decoder.JSONDecodeError:
                            self.hash_files.remove(log_hash)
                            os.remove(cached_path)
                        except AssertionError:
                            self.hash_files.remove(log_hash)
                            os.remove(cached_path)

                    d_loaded_logs[log_hash] = log

                logs.append(d_loaded_logs[log_hash])
                log_hashs.append(log_hash)

        return logs, log_hashs


    def _serialize_logs(
        self,
        logs: List[Dict],
        hashes: List[str]
    ) -> None:
        for log, hash in zip(logs, hashes):
            with open(os.path.join(self.log_dir, hash), 'w') as outfile:
                json.dump(log, outfile, indent=2)


    def open_log_from_text(self, text: str) -> Dict:
        """
        Function to open a serialised log and analyse it.
        """
        log_hash = text2hash(text)
        with open(os.path.join(self.log_dir, log_hash), 'r') as f_log:
            log = json.load(f_log)
        return log


    def _compute_answer_selection(
        self,
        logs: List[Dict],
        type_logs: str
    ) -> None:
        #for answer_type in self._get_answer_types(type_logs):
        to_do_exs, to_do_exs_idxs, to_do_ref_idxs = [], [], []
        # log: dictionnary of hashcode of each sentence
        for idx, log in enumerate(logs):
            for ref_idx, ref in enumerate(log['references']):
                to_do_exs.append(ref['text'])
                to_do_exs_idxs.append(idx)
                to_do_ref_idxs.append(ref_idx)

        assert len(to_do_exs) == len(to_do_exs_idxs) == len(to_do_ref_idxs)

        if len(to_do_exs) != 0:
            for answer_type in self._get_answer_types(type_logs):
                list_answers = self._predict_self_answers(to_do_exs, answer_type)
                for i in range(len(list_answers)):
                    logs[to_do_exs_idxs[i]]['references'][to_do_ref_idxs[i]]['answers'] += list_answers[i]

        for log in logs:
            for ref in log['references']:
                ref['answers'] = list(set(ref['answers']))
        
        return len(to_do_exs) != 0


    def _compute_question_generation(
        self,
        logs: List[Dict],
        type_logs: str
    ) -> None:

        to_do_exs, to_do_exs_idxs, to_do_ref_idxs = [], [], []
        for idx, log in enumerate(logs):

            for ref_idx, ref in enumerate(log['references']):
                if ref['text'] == '':
                    continue
                to_do_exs += [(a, ref['text']) for a in ref['answers']]
                to_do_exs_idxs += [idx] * len(ref['answers'])
                to_do_ref_idxs += [ref_idx] * len(ref['answers'])
        
        assert len(to_do_exs) == len(to_do_exs_idxs) == len(to_do_ref_idxs)

        if len(to_do_exs) != 0:
            question_texts = self._predict_questions(to_do_exs, type_logs)
            for i in range(len(question_texts)):
                if len(logs[to_do_exs_idxs[i]]['references'][to_do_ref_idxs[i]]['questions'])<len(logs[to_do_exs_idxs[i]]['references'][to_do_ref_idxs[i]]['answers']):
                    logs[to_do_exs_idxs[i]]['references'][to_do_ref_idxs[i]]['questions'].append(question_texts[i])

        return len(to_do_exs) != 0


    def _predict_questions(
        self,
        to_do_exs: List[tuple],
        type_logs: str
    ) -> List[str]:
        model_QG = self.models[type_logs]['QG']

        str_prefix = f'{self.qg_prefix} {self.sep} ' if self.qg_prefix is not None else ''
        formated_inputs = [f'{str_prefix}{asw} {self.sep} {context}' for asw, context in to_do_exs]
        
        _, question_texts = model_QG.predict(formated_inputs)

        return question_texts

    
    def _predict_answers(
        self,
        to_do_exs: List[tuple],
        type_logs: str
    ) -> Tuple[List[float], List[str]]:
        model_QA = self.models[type_logs]['QA']
        formated_inputs = [f'{question} {self.sep} {context}' for question, context in to_do_exs]
        qa_scores, qa_texts = model_QA.predict(formated_inputs)

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

    def set_model(
        self,
        key: str,
        task: str,
        model_name: str,
    ) -> None:

        assert key in [None, 'hyp', 'src', 'ref']
        assert task in ['weighter', 'QG', 'QG']

        model = self.get_model(model_name=model_name)

        if key is None:
            self.models[task] = model
        else:
            self.models[key][task] = model

    def _get_answer_hash(self) -> str:
        # TODO: self.spacy_pipeline
        msg = f"LimitSent={self.limit_sent}" \
              f"_models={'_'.join(self.answer_types)}"

        return msg


    def __hash__(self) -> str:
        msg = f"QuestEval_version={__version__}" \
              f"_task={self.task}_lang={self.language}_preproc={self.src_preproc_pipe}" \
              f"_consist={self.do_consistency}_scores={self.list_scores}" \
              f"{self._get_weighter_hash()}" \
              f"_hyp_{self._get_qa_hash('hyp')}_ref_{self._get_qa_hash('ref')}_src_{self._get_qa_hash('src')}" \
              f"_hyp_{self._get_qg_hash('hyp')}_ref_{self._get_qg_hash('ref')}_src_{self._get_qg_hash('src')}"

        return msg


questeval = QuestEval()
questeval.corpus_questeval()
