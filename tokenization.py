import string
import unicodedata
from datamaestro.definitions import AbstractDataset
import torch
import re
import numpy as np
from torch.utils.data import Dataset
import sys
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

def prepare_dataset(dataset_id: str):
    """Find a dataset given its id and download the resources"""

    ds = AbstractDataset.find(dataset_id)
    return ds.prepare(download=True)


def word2vec(OOVID, embedding_size=200):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage (embedding_size = [50,100,200,300])

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText) train
    - DataSet (FolderText) test
 
    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset(
        'edu.stanford.glove.6b.%d' % embedding_size).load()
    words = ['<unk>'] + words
    word2id = {word: ix+1 for ix, word in enumerate(words)}
    word2id['<unk>'] = OOVID
    
    id2word = dict(zip(word2id.values(),id2lettre.keys()))
    embeddings = np.vstack((np.zeros(embedding_size), embeddings))
    return word2id, id2word, embeddings, WORDS


## Token de padding (BLANK)
OOVID = 0
## Token de fin de séquence
#SEPID = -1
LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(1, len(LETTRES)+1), LETTRES))
id2lettre[OOVID] = '<unk>' #PAD
#id2lettre[SEPID] = '<sep>'
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))
NUM_LETTRES = len(lettre2id)
embedding_size = 200
word2id, id2word, embeddings, WORDS = word2vec(OOVID, embedding_size)
    

def normalize(s):
    """ enlève les accents et les caractères spéciaux"""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)

def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    # une séquence de phrases, représenté par les chiffres d'une lettre
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


class QuestionContextDataset(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""
    def __init__(self, questions, passages, answers):
        self.data = {'text':[], 'label':[]}
        
        import nltk
        nltk.download('punkt')

        for q, c, a in tqdm(zip(questions, passages, answers)):
            #phrases = re.split(r'(\.|\?|\!)', c)
            phrases = sent_tokenize(c)

            text = ['<sep>'.join([q, p]) for p in phrases]
            label = [1 if a in p else 0 for p in phrases]
            self.data['text'] += text
            self.data['label'] += label

        assert len(self.data['text']) == len(self.data['label'])
        
    def __len__(self):
        return len(self.data['text'])

    def __getitem__(self, ix):
        texts = self.data['text'][ix].split('<sep>')
        q, p = texts[0], texts[1]
        
        
        def split_tokens(t):
            return [x for x in re.findall(WORDS, t.lower())]

        def tokenizer(t):
            return torch.tensor([word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())])

        q_word_tok = tokenizer(q if isinstance(q, str) else q.read_text())
        p_word_tok = tokenizer(p if isinstance(p, str) else p.read_text())
        q_char_tok = [string2code(t) for t in split_tokens(q)]
        p_char_tok = [string2code(t) for t in split_tokens(p)]
    
        return q_word_tok, q_char_tok, p_word_tok, p_char_tok, self.data['label'][ix]
    


class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        """  Dataset pour les tweets de Trump
            * fname : nom du fichier
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        # décomposer le texte en phrases
        self.phrases = [re.sub(' +',' ',p[:maxlen]).strip() +"." for p in text.split(".") if len(re.sub(' +',' ',p[:maxlen]).strip())>0]
        if maxsent is not None:
            self.phrases = self.phrases[:maxsent]
        self.maxlen = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        return string2code(self.phrases[i])
