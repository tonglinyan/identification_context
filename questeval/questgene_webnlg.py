from locale import normalize
import re 
from typing import List, Tuple, Dict, Callable
import os

import json
import numpy as np
import logging
from datasets import load_metric, load_dataset
import spacy
import torch
from tqdm import tqdm
from utils import (
    API_T2T,
    sentencize,
    calculate_f1_squad,
    calculate_BERTScore,
    extract_table_answers,
    text2hash, 
    normalize_answer
    #_load_webnlg
    )

DIR = os.path.dirname(__file__)
__version__ = "0.2.4"

HF_ORGANIZATION = "ThomasNLG"

class QuestEval:
    def __init__(
        self,
        language: str = "en",
        answer_types: Tuple = ('NER', 'NOUN'),
        list_scores: Tuple = ('answerability', 'bertscore', 'f1'),
        src_preproc_pipe=None,
        do_weighter: bool = False,
        do_consistency: bool = False,
        qg_batch_size: int = 36,
        clf_batch_size: int = 48,
        limit_sent: int = 5,
        reduction_multi_refs: Callable = max,
        no_cuda: bool = False,
        use_cache: bool = True
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
        self.AVAILABLE_LANGUAGES = ("en",)  # todo: "multi"
        self.AVAILABLE_TASKS = ("text2text", "summarization", "text_simplification", "data2text")

        if language not in self.AVAILABLE_LANGUAGES:
            raise (
                f"Language {language} is not implemented. The list of available languages are: {self.AVAILABLE_LANGUAGES}."
            )

        self.log_dir = os.path.join(DIR, 'logs')
        self.hash_files = set(os.listdir(self.log_dir))
        self.use_cache = use_cache

        self.language = language

        self.answer_types = answer_types
        self.src_preproc_pipe = src_preproc_pipe
        self.limit_sent = limit_sent
        self.sep = "</s>"
        self.qg_prefix = None
        self.qg_batch_size = qg_batch_size
        self.clf_batch_size = clf_batch_size
        self.device = 'cuda' if (torch.cuda.is_available() and not no_cuda) else 'cpu'

        self.reduction_multi_refs = reduction_multi_refs
        self.do_consistency = do_consistency
        self.do_weighter = do_weighter
        self.list_scores = list_scores
        if 'bertscore' in self.list_scores:
            #modifié
            #self.metric_BERTScore = BERTScore()
            self.metric_BERTScore = load_metric("bertscore")

        if language == 'en':
            try:
                # spacy.pipeline includes tokenizer->tagger->parser->ner->doc, and formes the final doc.
                self.spacy_pipeline = spacy.load('en_core_web_sm')
            except OSError:
                logging.warning("Downloading language model for the spaCy model.")
                from spacy.cli import download
                download('en_core_web_sm')
                self.spacy_pipeline = spacy.load('en_core_web_sm')

        logging.info("Loading the models, it can take time to download at first time.")
        self.models = self._load_all_models()


    def _load_all_models(self) -> Dict:
        # Textual hypothesis
        models = {"hyp": {}}
        if self.language == 'en':
            #models['hyp']['QG'] = f'{HF_ORGANIZATION}/t5-qg_squad1-en'
            models['hyp']['QG'] = 't5_qg_squad1_en'
        else:
            raise("Multilingual evaluation not handled yet.")

        # Loading all the different models
        for modality in models.keys():
            for task in models[modality].keys():
                if not type(models[modality][task]) == str:
                    continue
                models[modality][task]= self.get_model(model_name=models[modality][task])

        return models


    def corpus_questeval(
        self,
        batch_size: int = 512
    ) -> Dict:

        logs = []
        d_loaded_logs = dict()


        logs, hyp_hashes, modified_logs = self._texts2logs(type_logs='hyp', d_loaded_logs=d_loaded_logs)

        if modified_logs: 
            self.is_question(logs)
            self.answer_filtering(logs)
            self._save_json(logs)
            print(modified_logs)

        """
        with open("webnlg_qgqa.json") as f:
            logs = json.load(f)
        self._save_json(logs)
        """
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
                                #log = tmp
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

    
    def is_question(
        self, 
        logs: List[Dict],
    ):

        QUESTION_WORDS = ['What', 'When', 'Where', 'Who', 'Which', 'Who', 'How', 'Whose', 'Whom', 'Why']
        QUESTION_WORDS += [w.lower() for w in QUESTION_WORDS]

        for log in logs:
            for ref in log['references']:
                assert len(ref['answers']) == len(ref['questions'])
                ref["isQuestion"] = []

                for q in ref['questions']:
                    if max([True if w in q else False for w in QUESTION_WORDS]):
                        ref["isQuestion"].append("True")
                        if not "?" in q:
                            q = q + "?"
                    else:
                        ref["isQuestion"].append("True")
                
                assert len(ref['questions']) == len(ref['isQuestion'])


    def answer_filtering(
        self, 
        logs: List[Dict],
    ):
        for log in logs:

            triples = log["triple_linearized"]

            for ref in log['references']:
                keep_idx = []

                for idx, a in enumerate(ref['answers']):
                    
                    if self.word_exist(a, triples):
                        keep_idx.append(idx)

                if len(keep_idx) > 0:           
                    ref['questions'] = [ref['questions'][i] for i in keep_idx]
                    ref['answers'] = [ref['answers'][i] for i in keep_idx]
                    ref['isQuestion'] = [ref['isQuestion'][i] for i in keep_idx]

                                        

    def word_exist(
        self, 
        str1:str,
        str2:str,
    ):
        #print(str1)
        """if bool(re.search(r'\d', str1)):
            str1 = re.findall(r'\d+\.?\d*', str1)[0]
        else:"""
        str1 = normalize_answer(str1)
        #print(str1)
        
        entites = [normalize_answer(a) for a in extract_table_answers(str2)]
        #print(entites)
        #print(max([str1==e for e in entites]))
        return max([str1==e for e in entites])


    # for source of data2text, return Table, else retuen self.answer_type 
    def _get_answer_types(self, type_logs: str) -> str:
        return ('TABLE', ) if type_logs == 'src' and self.task == 'data2text' else self.answer_types


    def _predict_self_answers(
        self,
        texts: List,
        answer_type: str
    ) -> List[str]:
        if self.limit_sent is not None:
            list_sentences = [sentencize(text, self.spacy_pipeline) for text in texts]
            texts = [' '.join(sentences[:self.limit_sent]) for sentences in list_sentences]

        list_answers = []
        if answer_type == 'NER':
            list_answers = [list(set([a.text for a in self.spacy_pipeline(text).ents])) for text in texts]
        elif answer_type == 'NOUN':
            list_answers = [list(set([a.text for a in self.spacy_pipeline(text).noun_chunks])) for text in texts]
        elif answer_type == 'TABLE':
            list_answers = [extract_table_answers(text) for text in texts]

        return list_answers


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
