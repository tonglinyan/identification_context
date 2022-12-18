import numpy as np
from tqdm import tqdm
import sys
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import json
import collections

def parse_args():
    parser = argparse.ArgumentParser('Calculation of cosinus similarity of context and (answer, question) pair.')
    parser.add_argument('--dataset', help='datasets', default = 'rotowire')
    parser.add_argument('--verbose', '-v', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

args = parse_args()


def _load_dataset(args):
    with open(args.dataset, 'r') as f:
        data = json.load(f)  
    data = data['data']
    logs = []
    for passage in data:
        context = [p['context'] for p in passage['paragraphs']]
        log = {'passage': context, 'qa':[]}
        for p in passage['paragraphs']:
            for qa in p['qas']:
                qa_log = {'answer': ', '.join(list(set([a['text'] for a in qa['answers']]))) if len(qa['answers']) > 0 else '', 'question': qa['question'], 'context': p['context']}
                log['qa'].append(qa_log)
        logs.append(log)

    return logs

def _tfidf(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    return tfidf.toarray()


def cos_similarity(tfidf_Q, tfidf_D):
    sim = []
    for tfd in tfidf_D:
        numerator = sum([a*b for a, b in zip(tfidf_Q, tfd)])
        denomitor = (sum([a*a for a in tfidf_Q])*sum([a*a for a in tfd])) ** 2
        sim.append(numerator/denomitor)

    return sim

logs = _load_dataset(args)
for log in tqdm(logs):
    for qa_pair in log['qa']:

        tfidf = _tfidf(['%s. %s'%(qa_pair['answer'], qa_pair['question'])]+log['passage'])
        sim = cos_similarity(tfidf[0], tfidf[1:])
        qa_pair['score'] = sim
        qa_pair['context_predicted'] = log['passage'][np.argmax(sim)]


with open('ranking_baseline.json', 'w') as f:
    json.dump(logs, f, indent=4)  
#with open('ranking_baseline.json', 'r') as f:
#    logs = json.load(f)  

def evaluation(logs):
    
    def top_k_map(N, logs):
        ap = []
        for log in logs:
            for qa in log['qa']:

                score = qa['score']
                label_ind = log['passage'].index(qa['context'])

                sorted_id = sorted(range(len(score)), key=lambda k: score[k], reverse=True)
                prec = 0
                for i in range(N):
                    if sorted_id[i] == label_ind:
                        prec = 1/(i+1)

                ap.append(prec) 

        return sum(ap)/sum([len(log['qa']) for log in logs])

    precision = sum([1 if qa['context']==qa['context_predicted'] else 0 for log in logs for qa in log['qa'] ])/sum([len(log['qa']) for log in logs])
    return collections.OrderedDict([
        ('precision', 100 * precision), 
        ('mAP', 100 * top_k_map(3, logs))])

print(evaluation(logs))
