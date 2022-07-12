import re
from typing import List, Dict
import os
import json
from datasets import load_dataset
import spacy
import torch
from questeval import DIR, __version__
from tqdm import tqdm
import torch
from utils import (
    API_BERT, 
    API_T2T,
    sentencize, 
    extract_table_answers,
    LinearizeDataInput, 
    normalize_answer, 
)
import numpy as np
import argparse
import sys
import collections
from nltk.tokenize import sent_tokenize
import requests
from bs4 import BeautifulSoup


DIR = os.path.dirname("__file__")
__version__ = "0.2.4"

def parse_args():
    parser = argparse.ArgumentParser('Evaluation script for each model in questeval.')
    parser.add_argument('--dataset', help='datasets', default = 'rotowire')
    parser.add_argument('--verbose', '-v', action='store_true')
    #parser.add_argument('--save_table', choices=[True, False], default=False)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

class data2text:
    def __init__(
        self,
        args, 
        qg_batch_size: int = 36,
        clf_batch_size: int = 48,
        limit_sent: int = 5,
        no_cuda: bool = False,
    ) -> None:

        self.dataset = args.dataset
        self.limit_sent = None
        self.sep = '</s>'
        self.qg_prefix = None
        self.qg_batch_size = qg_batch_size
        self.clf_batch_size = clf_batch_size
        self.spacy_pipeline = spacy.load('en_core_web_sm')
        self.device = 'cuda' if (torch.cuda.is_available() and not no_cuda) else 'cpu'

        self._load_all_models() 
        """
        self._load_dataset()
       
        with open(f'prediction_score_{self.dataset}.json', 'w') as f:
            json.dump(self.logs, f, indent=4)  
        
        self._compute_answer_selection(self.logs)
        with open(f'prediction_score_{self.dataset}.json', 'w') as f:
            json.dump(self.logs, f, indent=4)  
        
        self._compute_question_generation(self.logs)
        with open(f'prediction_score_{self.dataset}.json', 'w') as f:
            json.dump(self.logs, f, indent=4)  
        """
        with open(f'prediction_score_{self.dataset}.json', 'r') as f:
            self.logs = json.load(f)  

        self._compute_context_selection(self.logs)
        with open(f'prediction_score_{self.dataset}.json', 'w') as f:
            json.dump(self.logs, f, indent=4)   
            
        #self._save_json(preds)
        print(self._evaluation(self.logs))

        with open(f'prediction_score_{self.dataset}.json', 'w') as f:
            json.dump(self.logs, f, indent=4)  
        
        for log in self.logs:
            for sent in log['phrases']:
                for qa in sent['qa_pairs']:
                    del qa['scores']

        with open(f'prediction_{self.dataset}.json', 'w') as f:
            json.dump(self.logs, f, indent=4)  


    def _load_all_models(self):
    
        self.qg_model = self.get_model(model_name=f'/home/tonglin.yan/identification_context/questeval/t5_qg_sqa_en')
        #self.cls_model = self.get_model(model_name=f'/home/tonglin.yan/identification_context/questeval/bert_classification')
        self.ranking_model = self.get_model(model_name=f'/home/tonglin.yan/identification_context/questeval/bert_cls2')


    def _load_dataset(self):
        if self.dataset == 'rotowire':
            table, text = self._load_rotowire()
        if self.dataset == 'wikibio':
            table, text = self._load_wikibio()
        if self.dataset == 'totto':
            table, text, self.answers = self._load_totto()
            with open(f'prediction_score_{self.dataset}.json', 'w') as f:
                logs = {'table': table, 'text':text, 'answer':self.answers}
                json.dump(logs, f, indent=4)   
        self.logs = self._load_logs(table, text)
        #print(self.logs)

    
    def _load_wikibio(self):
        dataset = load_dataset("wiki_bio")
        train = dataset['test']# train, test, validation
        
        def table_text(data):
            header = [d['input_text']['table']['column_header'] for d in tqdm(data)]
            content = [d['input_text']['table']['content'] for d in tqdm(data)]
            
            table = []
            for h, c in zip(header, content):
                table.append(LinearizeDataInput(h, [c]))
            text = [d['target_text'] for d in data]

            assert len(table) == len(text)
            return table, text
        
        return table_text(train)
        
    def _load_totto(self):
        dataset = load_dataset("totto")
        test = dataset['validation']
        # keys: 'id', 'table_page_title', 'table_webpage_url', 'table_section_title', 'table_section_text', 'table', 
        # 'highlighted_cells', 'example_id', 'sentence_annotations', 'overlap_subset'
        
        def table_text_anwser(data):

            tables = []
            answers = []
            text = []

            for d in tqdm(data):
                sentence = d['sentence_annotations']
                table = d['table']
                answer_idx = d['highlighted_cells']
                site = d['table_webpage_url']

                def scraping(site, max=10):
                    try:
                        response = requests.get(site)
                        response.encoding = "utf-8"
                        soup = BeautifulSoup(response.content, features='html.parser')
                        div = soup.find('div', id='mw-content-text').text.strip()
                        
                        sentences = re.split('\n', div)
                        def not_empty(s):
                            return s and s.strip() and s.strip()[-1]=='.' and len(s)>30
                        sentences = list(filter(not_empty, sentences))
                        if len(sentences) > max:
                            sentences = sentences[:max]
                        sentences = [s.replace('^', '').strip() for s in sentences]
                    except:
                        sentences = []

                    #import time
                    #time.sleep(1)

                    return sentences
                
                sentences = scraping(site)
                try:
                    sentences.append(' '.join(sentence['final_sentence']))
                except:
                    sentences.append(' '.join(sentences['original_sentence']))

                rows = []
                header = -1
                for line in table:
                    row = []
                    for ent in line:
                        row += [ent['value']] * ent['column_span']
                    rows.append(row)
                
                asws = []

                for asw_idx in answer_idx:
                    if table[asw_idx[0]][asw_idx[1]]['is_header']:
                        asws.append(rows[asw_idx[0]][asw_idx[1]])
                    else:
                        asws.append(f'[{rows[asw_idx[0]][asw_idx[1]]}]')
                        
                if header >= 0:
                    table_linearized = LinearizeDataInput(rows[0], rows[1:])
                else:
                    table_linearized = LinearizeDataInput(None, rows)
                
                tables.append(table_linearized)
                answers.append(', '.join(asws))
                text.append(sentences)
            return tables, text, answers

        return table_text_anwser(test)


    def _load_rotowire(self):
        dataset = load_dataset("GEM/RotoWire_English-German")
        train = dataset['train']# train, test, validation

        # 'id', 'gem_id', 'home_name', 'box_score', 'vis_name', 'summary', 'home_line', 'home_city', 'vis_line', 
        # 'vis_city', 'day', 'detok_summary_org', 'detok_summary', 'summary_en', 'sentence_end_index_en', 'summary_de', 
        # 'target', 'references', 'sentence_end_index_de', 'linearized_input'
        
        def table_text(data):
            table = [d['linearized_input'].replace('<', '] ').replace('>', ' [').lstrip(']')+' ]' for d in data]
            text = [d['detok_summary'] for d in data]
            return table, text
        
        return table_text(train)
        
        
    def _load_logs(self, table, text):
        if self.dataset == 'totto':
            logs = []
            for tab, t, a in zip(table, text, self.answers):
                log = {'table': tab, 'text': t, 'phrases':[]}
                sent = {'text': t[-1], 'qa_pairs':[{'answer':a}]}
                log['phrases'].append(sent)
                logs.append(log)
        else:
            logs = []
            for tab, t in zip(table, text):
                log = {'table': tab, 'text': t, 'phrases':[]}
                phrases = sent_tokenize(t)
                for p in phrases:
                    sent = {'text': p, 'qa_pairs':[]}
                    log['phrases'].append(sent)
                logs.append(log)
        return logs
        
    """
    def _save_json(
        self, 
        preds: List
    ):
        data = []
        if self.task == 'BERT_Classification':
                qa_pair, log = None, None
                for s1, s2, l, p in zip(self.sentence1, self.sentence2, self.label, preds):
                    if s1 != qa_pair:
                        data.append(log)
                        log = {'question-answer:': s1, 'context': [], 'prediction': []}
                        qa_pair = s1
                    if l == 1:
                        log['context'].append(s2)
                    if p == 1:
                        log['prediction'].append(s2)
                self.data = list(filter(None, data))
            
        elif self.task == 'BERT_Ranking':
                qa_pair, log = None, None
                for s1, s2, p in zip(self.sentence1, self.sentence2, preds):
                    if s1 == qa_pair:
                        dict_text = {'text':s2, 'score': p}
                        log['context'].append(dict_text)
                    else:
                        data.append(log)
                        dict_text = {'text':s2, 'score': p}
                        log = {'question-answer:': s1, 'context': [dict_text]}
                        qa_pair = s1
                self.data = list(filter(None, data))
        else:
            for c, q, a, p in zip(self.text, self.question, self.answer, preds):
                log = {"table": c, "question": q, "answer": a, "prediction": p}        
                data.append(log)
            self.data = data

        with open(f'{self.task}_{self.dataset}_{self.author}.json', 'w') as f:
            json.dump(self.data, f, indent=4)        
    """

    def _compute_answer_selection(
        self,
        logs: List[Dict],
    ) -> None:


        if self.dataset != 'totto':
            to_do_exs_tab, to_do_exs, to_do_idx, to_do_sent_idx = [], [], [], []
            # log: dictionnary of hashcode of each sentence
            for idx, log in enumerate(logs):
                to_do_exs_tab.append(log['table'])
                for sent_idx, sent in enumerate(log["phrases"]):
                    to_do_exs.append(sent['text'])
                    to_do_idx.append(idx)
                    to_do_sent_idx.append(sent_idx)

            if len(to_do_exs) != 0 or len(to_do_exs_tab) != 0 :
                list_answers_tab = self._predict_self_answers(to_do_exs_tab, 'TABLE')
                list_answers = {t:[] for t in ['NER', 'NOUN']}
                for answer_type in ['NER', 'NOUN']:
                    list_answers[answer_type] = self._predict_self_answers(to_do_exs, answer_type)

                for i in range(len(to_do_exs)):

                    def filtering_entity(answer_ner, answer_noun, answer_tab):
                        answers_all = list(set(answer_ner+answer_noun))
                        answers = []

                        answers_from_table = []
                        """
                        for a_t in answer_tab:
                            words_tab = a_t.split(' ')
                            
                            #words_tab = [normalize_answer(w) for w in words_tab]
                            answers_from_table.append(words_tab)

                        for a in answers_all:
                            if a in answer_tab:
                                answers.append(a)
                            else:
                                words = re.split(' |-', a)
                                #print(words)
                                #words = [normalize_answer(w) for w in words]
                                
                                for w in words:
                                    for ans_tab in answers_from_table:
                                        if w in ans_tab:
                                            answers.append(' '.join(ans_tab))                   
                        """
                        from fuzzywuzzy import fuzz
                        for asw in answers_all:
                            for a_t in answer_tab:
                                if fuzz.token_sort_ratio(a_t, asw) > 70:
                                    answers.append(a_t)
                                
                        if '' in answers:
                            answers.remove('')

                        try:
                            #print('all answers: ', list(set(answers)))
                            return list(set(answers))
                        except:
                            #print("Extracted answers don't correspond to any cell in table")
                            return None

                    answers = filtering_entity(list_answers['NER'][i], list_answers['NOUN'][i], list_answers_tab[to_do_idx[i]])
                    #print(ref)
                    if answers != None:
                        logs[to_do_idx[i]]['phrases'][to_do_sent_idx[i]]['qa_pairs'] = []
                        for a in answers:
                            qa_pair = {'answer': a}
                            logs[to_do_idx[i]]['phrases'][to_do_sent_idx[i]]['qa_pairs'].append(qa_pair)
                    else: 
                        logs[to_do_idx[i]]['phrases'][to_do_sent_idx[i]]['qa_pairs'] = answers
                    


    def _compute_question_generation(
        self,
        logs: List[Dict],
    ) -> None:

        to_do_exs, to_do_idx, to_do_sent_idx, to_do_qa_idx = [], [], [], []
        # log: dictionnary of hashcode of each sentence
        for idx, log in enumerate(logs):
            for sent_idx, sent in enumerate(log["phrases"]):
                for qa_idx, ans in enumerate(sent['qa_pairs']):
                    if ans != None:
                        to_do_exs.append((ans['answer'], log['table']))
                        to_do_idx.append(idx)
                        to_do_sent_idx.append(sent_idx)
                        to_do_qa_idx.append(qa_idx)

        if len(to_do_exs) != 0:
            question_texts = self._predict_questions(to_do_exs)
            for i in range(len(question_texts)):
                logs[to_do_idx[i]]['phrases'][to_do_sent_idx[i]]['qa_pairs'][to_do_qa_idx[i]]['question'] = question_texts[i]


    def _compute_context_selection(
        self, 
        logs: List[Dict],
    ) -> None: 
        
        to_do_exs, to_do_sent, to_do_idx, to_do_sent_idx, to_do_qa_idx = [], [], [], [], []
        # log: dictionnary of hashcode of each sentence
        for idx, log in enumerate(logs):
            try:
                phrases = sent_tokenize(log['text'])
            except:
                phrases = log['text']
            for sent_idx, sent in enumerate(log["phrases"]):
                for qa_idx, ans in enumerate(sent['qa_pairs']):
                    if ans != None:
                        ans['scores'] = []
                        to_do_exs += [(ans['answer'], ans['question'])]*len(phrases)
                        to_do_sent += phrases
                        to_do_idx += [idx] * len(phrases)
                        to_do_sent_idx += [sent_idx] * len(phrases)
                        to_do_qa_idx += [qa_idx] * len(phrases)
        
        if len(to_do_exs) != 0:
            scores = self._predict_context(to_do_exs, to_do_sent)
            for i, s in enumerate(scores):
                logs[to_do_idx[i]]['phrases'][to_do_sent_idx[i]]['qa_pairs'][to_do_qa_idx[i]]['scores'].append(s[0])


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
            list_answers = [[a.text for a in self.spacy_pipeline(text).ents] for text in tqdm(texts)]
        elif answer_type == 'NOUN':
            list_answers = [[a.text for a in self.spacy_pipeline(text).noun_chunks] for text in tqdm(texts)]
        elif answer_type == 'TABLE':
            list_answers = [extract_table_answers(text) for text in texts]

        return list_answers
    
    """
    def _predict_self_answers(
        self,
        texts: List,
        answer_type: str
    ) -> List[str]:
        import nltk
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')


        for t in texts:
            sents = nltk.sent_tokenize(t)
            words = []
            for sent in sents:
                words += nltk.word_tokenize(sent)
            
            tags = nltk.pos_tag(words)
            ners = nltk.ne_chunk(tags)
            print(str(ners))
            print(str(ners.node))
    """

    def _predict_questions(
        self,
        to_do_exs: List[tuple], 
        batch_size: int=256, 
    ) -> List[str]:
        str_prefix = f'{self.qg_prefix} {self.sep} ' if self.qg_prefix is not None else ''
        formated_inputs = [f'{str_prefix}{asw} {self.sep} {context}' for asw, context in to_do_exs]
        question_texts = []

        for i in tqdm(range(0, len(formated_inputs), batch_size)):
            _, question_text = self.qg_model.predict(formated_inputs[i:i+batch_size])
            question_texts += question_text

        return question_texts

    def _predict_context(
        self,
        to_do_exs: List[tuple], 
        sent: List[str], 
        batch_size: int=256, 
    ) -> List[str]:
        sent1 = [f'question: {q} answer: {a}' for q, a in to_do_exs]
        sent2 = sent
        preds = []
        
        for i in tqdm(range(0, len(sent1), batch_size)):
            preds += self.ranking_model.predict(sent1[i:i+batch_size], sent2[i:i+batch_size], binary=False)
        return preds


    def _evaluation(
        self, 
        logs: List[Dict],
    ):
        
        list_pred = []
        for log in logs:
            try:
                phrases = sent_tokenize(log['text'])
            except:
                phrases = log['text']
            for sent in log['phrases']:
                context = sent['text']

                if sent['qa_pairs'] != None:
                    for qa in sent['qa_pairs']:
                        context_predicted = phrases[np.argmax(qa['scores'])]
                        qa['context'] = context_predicted
                        
                        list_pred.append(1 if context_predicted == context else 0)
        
        precision = sum(list_pred)/len(list_pred)
        return collections.OrderedDict([
            ('precision', 100 * precision)])
        

    def get_model(self, model_name: str):
        keep_score_idx = None

        if 'qg' in model_name:
            self.qg_prefix = 'sv1'

        # batch size
        model_batch_size = self.qg_batch_size if 'qg' in model_name.lower() else self.clf_batch_size

        if 'bert' in model_name:
            model = API_BERT(
                pretrained_model_name_or_path=model_name,
                keep_score_idx=keep_score_idx,
                max_source_length=512,
                model_batch_size=model_batch_size,
                device=self.device, 
            )

        else:
            model = API_T2T(
                pretrained_model_name_or_path=model_name,
                keep_score_idx=keep_score_idx,
                max_source_length=512,
                model_batch_size=model_batch_size,
                device=self.device, 
            )

        return model


if __name__ == '__main__':
    args = parse_args()
    questeval = data2text(args)

