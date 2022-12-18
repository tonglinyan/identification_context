import re
import os
from tokenize import String
from typing import List, Dict
from questeval import DIR
import json
from datasets import load_dataset
import spacy
import torch
from tqdm import tqdm
import torch
from utils import (
    API_BERT, 
    API_T2T,
    sentencize, 
    extract_table_answers,
    extract_table_answers_with_keys,
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
from fuzzywuzzy import fuzz

DIR = f'{os.path.dirname(os.path.abspath(__file__))}/questeval'

def parse_args():
    parser = argparse.ArgumentParser('Pipeline')
    parser.add_argument('--dataset', '-d', choices=['wikibio', 'totto', 'rotowire'], help='datasets', default = 'rotowire')
    parser.add_argument('--ranking', '-r', choices=['ranking_sent', 'ranking', 'classification_sent', 'classification'], default='ranking', help="context that we want to identify, choices: ranking, ranking_sent, classification, classification_sent")
    parser.add_argument('--filtrage', '-f', action='store_true', default=False, help='using the fuzzywuzzy to filter the values extracted from text or not')
    parser.add_argument('--withkeys', '-wk', action='store_true', default=False, help='question-genenration system generates question with keys or not')
    parser.add_argument('--withquestion', '-wq', action='store_true', default=False, help='identification system with question or not')

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
        no_cuda: bool = False,
    ) -> None:

        self.filtrage = args.filtrage
        self.ranking = args.ranking
        self.withkeys = args.withkeys
        self.withquestion = args.withquestion

        self.dataset = args.dataset
        self.limit_sent = None
        self.sep = '</s>'
        self.qg_prefix = None
        self.qg_batch_size = qg_batch_size
        self.clf_batch_size = clf_batch_size
        self.spacy_pipeline = spacy.load('en_core_web_sm')
        self.device = 'cuda' if (torch.cuda.is_available() and not no_cuda) else 'cpu'

        self._load_all_models() 

        self._load_dataset()

        self._compute_answer_selection(self.logs)
        
        if self.withquestion:
            self._compute_question_generation(self.logs)

        self._compute_context_selection(self.logs)
        with open(f'prediction_score_{self.dataset}_cs_{self.filtrage}_{self.withkeys}_{self.withquestion}_{self.ranking}.json', 'w') as f:
            json.dump(self.logs, f, indent=4)   

        print(self._evaluation(self.logs))

        with open(f'prediction_score_{self.dataset}_{self.filtrage}_{self.withkeys}_{self.withquestion}_{self.ranking}.json', 'w') as f:
            json.dump(self.logs, f, indent=4)  


    def _load_all_models(
        self
    ) -> None :
        if self.withkeys:
            self.qg_model = self.get_model(model_name=f'{DIR}/t5_qg_sqa_key_en')
        else:
            self.qg_model = self.get_model(model_name=f'{DIR}/t5_qg_sqa_en')
        self.ranking_model = self.get_model(model_name=f'{DIR}/bert_{self.ranking}')


    def _load_dataset(
        self
    ) -> None :
        """
        Load logs of dataset

        logs: [{'table': (linearised table)
                'text': (passage)
                'phrases': [{'text': (one paragraph/phrase)
                            'qa_pairs': [{
                                    'question':
                                    'ambiguity':
                                    'key':
                                    'answer':
                            }, ...]
            }, ...]
        """

        if self.dataset == 'rotowire':
            table, text = self._load_rotowire()
        if self.dataset == 'wikibio':
            table, text = self._load_wikibio()
        if self.dataset == 'totto':
            table, text, self.answers, self.keys = self._load_totto()
            with open(f'prediction_score_{self.dataset}.json', 'w') as f:
                logs = {'table': table, 'text':text, 'answer':self.answers, }
                json.dump(logs, f, indent=4)   
        self.logs = self._load_logs(table, text)

    
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

            tables, answers, text, keys = [], [], [], []

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
                header = False
                for line in table:
                    row = []
                    for ent in line:
                        header = header or ent['is_header']
                        row += [ent['value']] * ent['column_span']
                    rows.append(row)
                
                asws = []
                key = []

                for asw_idx in answer_idx:
                    if table[asw_idx[0]][asw_idx[1]]['is_header']:
                        key.append(rows[asw_idx[0]][asw_idx[1]])
                    else:
                        asws.append(f'[{rows[asw_idx[0]][asw_idx[1]]}]')
                        
                if header:
                    table_linearized = LinearizeDataInput(rows[0], rows[1:])
                else:
                    table_linearized = LinearizeDataInput(None, rows)
                
                tables.append(table_linearized)
                answers.append(', '.join(asws))
                text.append(sentences)
                keys.append(key)
            return tables, text, answers, keys

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
        
        
    def _load_logs(
        self, 
        table: List,
        text: List
    ) -> Dict: 
        if self.dataset == 'totto':
            logs = []
            for tab, t, a, k in zip(table, text, self.answers, self.keys):
                log = {'table': tab, 'text': t, 'phrases':[]}
                sent = {'text': t[-1], 'qa_pairs':[{'answer':a, 'ambiguity': 1, 'key':k}]}
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

    def _compute_answer_selection(
        self,
        logs: List[Dict],
    ) -> None:
        """
        Select the value(s) in the table that we want to describ with given corpus
        If the answer(value) are already highted or given, jump to the next step
        """

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
                list_answers_tab = self._predict_self_answers(to_do_exs_tab, 'KEYS')
                list_answers = {t:[] for t in ['NER', 'NOUN']}
                for answer_type in ['NER', 'NOUN']:
                    list_answers[answer_type] = self._predict_self_answers(to_do_exs, answer_type)

                for i in range(len(to_do_exs)):

                    def filtering_entity(answer_ner, answer_noun, answer_tab):
                        answers_all = list(set(answer_ner+answer_noun))
                        answers, keys = [], []
                        pair_ka = []
                        def extract_key(text):
                            key_token = []
                            for tok in text.split():
                        
                                if tok != '[':   
                                    key_token.append(tok)
                                else: 
                                    return ' '.join(key_token)

                        def extract_answer(text):
                            asw_token = []
                            is_asw = False
                            for tok in text.split():

                                if tok == ']':
                                    ans = ' '.join(asw_token)
                                    return ans

                                if is_asw:
                                    asw_token.append(tok)

                                if tok == '[':
                                    is_asw = True

                        # fuzzywuzzy filtrage
                        if self.filtrage:
                            for asw in answers_all:
                                for a_t in answer_tab:
                                    value = extract_answer(a_t)
                                    key = extract_key(a_t)
                                    if fuzz.token_sort_ratio(normalize_answer(value), normalize_answer(asw)) > 70 and [key, value] not in pair_ka:
                                        pair_ka.append([key, value])

                        # filtrage
                        else:
                            answers_from_table = []
                            for a_t in answer_tab:
                                value = extract_answer(a_t)
                                key = extract_key(a_t)
                                words_tab = value.split(' ')
                                
                                #words_tab = [normalize_answer(w) for w in words_tab]
                                answers_from_table.append(words_tab)

                            for a in answers_all:
                                if a in answer_tab:
                                    answers.append(a)
                                else:
                                    words = re.split(' |-', a)
                                    
                                    for w in words:
                                        for ans_tab in answers_from_table:
                                            if w in ans_tab:
                                                answers.append(' '.join(ans_tab))
                        
                        #if '' in answers:
                        #    answers.remove('')
                        answers = [p[1] for p in pair_ka]
                        keys = [p[0] for p in pair_ka]
                        

                        try:
                            #print('all answers: ', list(set(answers)))
                            return answers, keys
                        except:
                            #print("Extracted answers don't correspond to any cell in table")
                            return None


                    answers, keys = filtering_entity(list_answers['NER'][i], list_answers['NOUN'][i], list_answers_tab[to_do_idx[i]])
                    #print(ref)
                    if answers != None:
                        logs[to_do_idx[i]]['phrases'][to_do_sent_idx[i]]['qa_pairs'] = []
                        for a, k in zip(answers, keys):
                            ambiguite = self._calculate_ambiguity(a, logs[to_do_idx[i]]['table'])
                            qa_pair = {'answer': a, 'ambiguity': ambiguite, 'key':k}
                            logs[to_do_idx[i]]['phrases'][to_do_sent_idx[i]]['qa_pairs'].append(qa_pair)
                    else: 
                        logs[to_do_idx[i]]['phrases'][to_do_sent_idx[i]]['qa_pairs'] = answers

    """
    def _calculate_ambiguity(
        self, 
        a:str, 
        tab:str,
    ) -> int:
        values = extract_table_answers(tab)
        times = [1 if v==a else 0 for v in values]
        return sum(times)
    """

    def _compute_question_generation(
        self,
        logs: List[Dict],
    ) -> None:
        """
        Generate question for each (answer, linearised table) pairs.
        """

        to_do_exs, to_do_idx, to_do_sent_idx, to_do_qa_idx = [], [], [], []
        # log: dictionnary of hashcode of each sentence
        for idx, log in enumerate(logs):
            for sent_idx, sent in enumerate(log["phrases"]):
                for qa_idx, ans in enumerate(sent['qa_pairs']):
                    if ans != None:
                        if self.withkeys:
                            key = ans['key']
                            value = ans['answer']
                            asw = f'{key} [ {value} ]'
                            to_do_exs.append((asw, log['table']))
                        else: 
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
        """
        Identify the relevant phrase or paragraph in the corpus for each (answer, question, context) triples.
        """
        to_do_exs, to_do_sent, to_do_idx, to_do_sent_idx, to_do_qa_idx = [], [], [], [], []

        for idx, log in enumerate(logs):
            try:
                phrases = sent_tokenize(log['text'])
            except:
                phrases = log['text']
            for sent_idx, sent in enumerate(log["phrases"]):
                for qa_idx, ans in enumerate(sent['qa_pairs']):
                    if ans != None:
                        ans['scores'] = []
                        asw = ans['answer']
                    
                        if self.withquestion:
                            to_do_exs += [(ans['question'], asw)]*len(phrases)
                        else:
                            to_do_exs += [(' ',asw)]*len(phrases)
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
        elif answer_type == 'KEYS':
            list_answers = [extract_table_answers_with_keys(text) for text in texts]
                

        return list_answers
    

    def _predict_questions(
        self,
        to_do_exs: List[tuple], 
        batch_size: int=256, 
    ) -> List[str]:
        str_prefix = f'{self.qg_prefix} {self.sep} ' if self.qg_prefix is not None else ''
        formated_inputs = [f'{str_prefix}{asw} {self.sep} {context}' for asw, context in to_do_exs]
        question_texts = []

        for i in tqdm(range(0, len(formated_inputs), batch_size)):
            score, question_text = self.qg_model.predict(formated_inputs[i:i+batch_size])
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
        
        list_pred, list_amb = [], []
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
                        list_amb.append(qa['ambiguity'])
        
        assert len(list_pred) == len(list_amb)
        
        precision = sum(list_pred)/len(list_pred)
        amb_mal_pred = sum([1 if p==0 and a > 1 else 0 for (p, a) in zip(list_pred, list_amb)])/len(list_pred)
        non_amb_mal_pred = sum([1 if p==0 and a == 1 else 0 for (p, a) in zip(list_pred, list_amb)])/len(list_pred)
        non_amb_pred = sum([1 if p==1 and a == 1 else 0 for (p, a) in zip(list_pred, list_amb)])/len(list_pred)
        amb_pred = sum([1 if p==1 and a > 1 else 0 for (p, a) in zip(list_pred, list_amb)])/len(list_pred)

        return collections.OrderedDict([
            ('precision', 100 * precision), 
            ('ambiguous and negative', 100 * amb_mal_pred), 
            ('non ambiguous and positive', 100 * non_amb_pred), 
            ('ambiguous and positive', 100 * amb_pred), 
            ('non ambiguous and negative', 100 * non_amb_mal_pred)])
        

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

