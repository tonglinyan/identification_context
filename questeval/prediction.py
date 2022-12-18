from typing import List, Tuple
import os
import json
import numpy as np
from datasets import load_dataset, load_metric
import torch
from tqdm import tqdm
import torch
from utils import (
    API_BERT, 
    API_T2T,
    LinearizeDataInput,
    clean_table, 
    clean_obj,
    calculate_f1_squad, 
    calculate_BERTScore, 
    calculate_exact
)
import argparse
import sys
import collections
DIR = os.path.dirname(os.path.abspath(__file__))
HF_ORGANIZATION = 'ThomasNLG'

def parse_args():
    parser = argparse.ArgumentParser('Evaluation script for each model in questeval.')
    parser.add_argument('--task', choices=['QA_D2T', 'QG_D2T', 'QA_T2T', 'QG_T2T', 'BERT_Classification', 'BERT_Ranking', 'BERT_Classification_sent', 'BERT_Ranking_sent'], help='prediction of question answering, question generation or identification of context')
    parser.add_argument('--author', choices=['hf', 'trained'], default='trained')
    parser.add_argument('--dataset', choices=['squad', 'sqa', 'wtq', 'webnlg', 'squadv2'], help='datasets', default = 'squad')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--save_table', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

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

        self.T2T_DATASETS = ['squad', 'squadv2']
        self.D2T_DATASETS = ['sqa', 'wtq', 'webnlg']
        self.BERT_DATASETS = ['squad']
        
        if 'T2T' in self.task and self.dataset not in self.T2T_DATASETS:
            raise (
                f'Dataset {self.dataset} is not available for model {self.task}. The list of available question-answering datasets are: {self.T2T_DATASETS}.'
            )

        if 'D2T' in self.task and self.dataset not in self.D2T_DATASETS:
            raise (
                f'Dataset {self.dataset} is not available for model {self.task}. The list of available question-answering datasets are: {self.D2T_DATASETS}.'
            )

        if 'BERT' in self.task and self.dataset not in self.BERT_DATASETS:
            raise (
                f'Dataset {self.dataset} is not available for model {self.task}. The list of available question-answering datasets are: {self.BERT_DATASETS}.'
            )
        
        self._load_all_models()
        self._load_dataset()
        preds = self._prediction()

        with open(f'prediction_{self.task}.json', 'w') as f:
            data = {'class': preds}
            json.dump(data, f, indent=4)          
        
        self._save_json(preds)      
        print(self._evaluation(preds))


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
                self.model = self.get_model(model_name=f'{DIR}/t5_qa_{self.dataset}_en')
            if self.task == 'QA_T2T':
                self.model = self.get_model(model_name=f'{DIR}/t5_qa_squad2neg_en')
            if self.task == 'QG_D2T':
                self.model = self.get_model(model_name=f'{DIR}/t5_qg_{self.dataset}_en')
            if self.task == 'QG_T2T':
                self.model = self.get_model(model_name=f'{DIR}/t5_qg_squad1_en')
            if 'BERT' in self.task:
                self.model = self.get_model(model_name=f'{DIR}/{self.task.lower()}')

    def _load_dataset(self):

        if self.dataset == 'squadv2':
            self.text, self.question, self.answer = self._load_squad_v2()
        if self.dataset == 'squad':
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
        if 'BERT' in self.task:
            raw_datasets = load_dataset(
                'data/squad_cls.py',
                data_files={"validation":"dev-v1.1.json"},
            )
            test_set = raw_datasets['validation']
            self.sentence1, self.sentence2, self.label = test_set['sentence1'], test_set['sentence2'], test_set['label']
        if 'sent' in self.task:
            raw_datasets = load_dataset(
                'data/squad_rank_sent.py',
                data_files={"validation":"dev-v1.1.json"},
            )
            test_set = raw_datasets['validation']
            self.sentence1, self.sentence2, self.label = test_set['sentence1'], test_set['sentence2'], test_set['label'],

    def _load_squad_v2(self):
        raw_dataset = load_dataset('squad_v2')['validation']
        context = raw_dataset['context']
        question = raw_dataset['question']
        answer = [a['text'][0] if len(a["text"]) > 0 else "unanswerable" for a in raw_dataset['answers']]
        return context, question, answer


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

        def answer_generation(coor, table, header):
            row_ind = [c['row_index'] for c in coor]
            col_ind = [c['column_index'] for c in coor]

            asws = []
            for t, h, rows, cols in zip(table, header, row_ind, col_ind):
                asws.append(', '.join([f'{h[c]}[{t[r][c]}]' for (r, c) in zip(rows, cols)]))
            return asws
        #all_answer_idx = dataset['answer_coordinates']
        #all_answers = answer_generation(all_answer_idx, all_tables, all_headers)
        
        
        # finetuning SQA with all pairs
        #questions, headers, answers, tables = all_questions, all_headers, all_answers, all_tables
        
        # finetuning SQA filtering pairs
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

        """
        with open(f'ambiguity.json', 'r') as f:
            logs = json.load(f)   

        tables = [log['table'] for log in logs]
        headers = [log['header'] for log in logs]
        questions = [log['question'] for log in logs]
        answers = [log['answer'] for log in logs]
        """

        tables = [clean_table(t, h) for t, h in zip(tables, headers)]
        texts = [LinearizeDataInput(t[0], t[1:]) for t in tables]

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
        if 'Classification' in self.task:
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
            
        elif 'Ranking' in self.task:
            qa_pair, log = None, None
            for s1, s2, p, l in zip(self.sentence1, self.sentence2, preds, self.label):
                if s1 != qa_pair:
                    data.append(log)
                    log = {'question-answer:': s1, 'context_predicted': [s2], 'score':[p[0]]}
                    qa_pair = s1
                else:
                    log['context_predicted'].append(s2)
                    log['score'].append(p[0])
                if l == 1:
                    log['context'] = s2
            self.data = list(filter(None, data))

        else:
            for c, q, a, p in zip(self.text, self.question, self.answer, preds):
                log = {"table": c, "question": q, "answer": a, "prediction": p}        
                data.append(log)
            self.data = data

        with open(f'{self.task}_{self.dataset}_{self.author}.json', 'w') as f:
            json.dump(self.data, f, indent=4)        


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
        
        if 'BERT' in self.task:
            binary = ('Classification' in self.task)
            preds = []
            #self.label = self.label[:int(0.06*len(self.label))]
            for idx in tqdm(range(0, len(self.label), batch_size)):
                pred = self.model.predict(self.sentence1[idx:idx+batch_size], self.sentence2[idx:idx+batch_size], binary)
                preds += pred
            assert len(preds) == len(self.label)
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
        if self.author == 'hf':
            formated_inputs = [f'{question} {self.sep} {context}' for question, context in to_do_exs]
        else:
            formated_inputs = [f'question: {question} context: {context}' for question, context in to_do_exs]
        qa_scores, qa_texts = self.model.predict(formated_inputs)

        return qa_scores, qa_texts


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

        elif 'classification' in self.task.lower():
            #if self.binary:
            accuracy = sum([1 if l == p else 0 for l, p in zip(self.label, preds)])/len(self.label)
            list_correct = []
            for log in self.data:
                crct = True
                for c in log['context']:
                    crct = (c in log['prediction']) and crct
                if crct:
                    list_correct.append(1)
                else:
                    list_correct.append(0)
            precision = sum([1 if log['context'] in log['prediction'] else 0 for log in self.data])/len(self.data)
            precision = sum(list_correct)/len(self.data)
            return collections.OrderedDict([
                ('accuracy', 100.0 * accuracy),
                ('precision', 100 * precision)])

        else:
            def top_k_map(N, logs):
                ap = []
                for log in logs:
                    score = log['score']
                    label_ind = log['context_predicted'].index(log['context'])

                    sorted_id = sorted(range(len(score)), key=lambda k: score[k], reverse=True)
                    prec = 0

                    N = len(sorted_id) if len(sorted_id) < N else N
                    for i in range(N):
                        if sorted_id[i] == label_ind:
                            prec = 1/(i+1)
                    ap.append(prec) 

                return sum(ap)/len(logs)

            precision = sum([1 if log['context_predicted'][np.argmax(log['score'])]==log['context'] else 0 for log in self.data])/len(self.data)

            return collections.OrderedDict([
                ('precision', 100 * precision), 
                ('mAP', 100*top_k_map(3, self.data))])


        

    def get_model(self, model_name: str):
        keep_score_idx = None

        if 'QG' in self.task:
            self.qg_prefix = 'sv1'

        # batch size
        model_batch_size = self.qg_batch_size if 'QG' in self.task else self.clf_batch_size

        if 'BERT' in self.task:
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
    questeval = Evaluation(args)

