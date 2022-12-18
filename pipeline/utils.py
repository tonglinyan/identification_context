from typing import Dict, List, Optional, Tuple
import string
import re
import unidecode
import collections
import torch
import hashlib
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    AutoTokenizer, 
    AutoModelForSequenceClassification
)

# encryptographe
def text2hash(string: str) -> str:
    hash_object = hashlib.sha512(string.encode('utf-8'))
    hex_dig = hash_object.hexdigest()

    return hex_dig


def split_on_punct(doc):
    """
    From one spacy doc to a List of (sentence_text, (start, end))
    """
    start = 0
    seen_period = False
    start_idx = 0

    for i, token in enumerate(doc):
        if seen_period and not token.is_punct:
            yield doc[start: token.i].text, (start_idx, token.idx)
            start = token.i
            start_idx = token.idx
            seen_period = False
        elif token.text in [".", "!", "?"]:
            seen_period = True
    if start < len(doc):
        yield doc[start: len(doc)].text, (start_idx, len(doc.text))


def sentencize(
    text: str, spacy_pipeline
) -> List:
    preprocessed_context = spacy_pipeline(text)
    return [sentence_tuple[0] for sentence_tuple in split_on_punct(preprocessed_context)]


def clean_obj(
    s,
    lc: bool = False
):
    s = unidecode.unidecode(s)
    if lc: s = s.lower()
    s = re.sub('^"|"$', "", s)  # remove useless quotesigns
    s = re.sub('_', ' ', s)  # turn underscores to spaces
    return s


def clean_table(rows, header):

    def clean_row(row):

        new_row = []
        start = -1
        for i in range(len(row)):

            if '"' in row[i]:
                if start == -1:
                    start = i
                else:
                    words = ''.join([row[j] for j in range(start, i+1)])
                    start = -1
                    new_row.append(words)
            else:
                if start == -1:
                    new_row.append(row[i])
        return new_row
    
    header = clean_row(header)

    new_rows = [header]
    for row in rows:
        new_row = clean_row(row)
        assert len(header) == len(new_row)
        new_rows.append(new_row)

    return new_rows


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


# extract answer from: text [answer] text
def extract_table_answers_with_keys(
    text: str
) -> List[str]:

    asws = []

    asw_toks = []
    for tok in text.split():
        asw_toks.append(tok)
        
        if tok == ']':   
            asws.append(' '.join(asw_toks).strip())
            asw_toks = []

    return asws


def extract_table_answers(
    text: str
) -> List[str]:

    asws = []

    asw_toks = []
    is_asw = False
    for tok in text.split():

        if tok == ']':
            asws.append(' '.join(asw_toks))
            is_asw = False
            asw_toks = []

        if is_asw:
            asw_toks.append(tok)

        if tok == '[':
            is_asw = True
    return asws

class API_T2T:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_source_length: int,
        model_batch_size: int,
        keep_score_idx: int,  # Note: will work only if beamsize == 1
        device: str = "cuda"
    ) -> None:

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        #print(self.pretrained_model_name_or_path)
        # import tokenizer from hugging face
        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path = self.pretrained_model_name_or_path
        )
        
        self.model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path = self.pretrained_model_name_or_path
        )
        self.keep_score_idx = keep_score_idx

        if device == "cuda":
            self.model.cuda()
        self.max_source_length = max_source_length
        self.model_batch_size = model_batch_size

    def predict(
        self,
        sources, 
    ):
        ### sources should be question <s> context ###

        gen_texts = []
        keep_score_idx_scores = []
        
        # padding: true/longest/max_length/false
        # truncation: true (cut at the "max_length") / false 
        for i in range(0, len(sources), self.model_batch_size):

            inputs = self.tokenizer(
                sources[i: i+self.model_batch_size],
                max_length=self.max_source_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                verbose=False,
            )

            with torch.no_grad():
                source_ids, source_mask = inputs["input_ids"], inputs["attention_mask"]
                dict_generated_ids = self.model.generate(
                    input_ids=source_ids.to(self.model.device),
                    attention_mask=source_mask.to(self.model.device),
                    use_cache=True,
                    decoder_start_token_id=None,
                    num_beams=1,
                    num_return_sequences=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                gen_text = self.tokenizer.batch_decode(
                    dict_generated_ids['sequences'],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                gen_texts += gen_text

                keep_score_idx_score = (1 - dict_generated_ids['scores'][0].softmax(-1)[:, self.keep_score_idx])
                if len(gen_text) != 1:
                    keep_score_idx_score = keep_score_idx_score.squeeze()
                keep_score_idx_scores += keep_score_idx_score.tolist()

        return keep_score_idx_scores, gen_texts


class API_BERT:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_source_length: int,
        model_batch_size: int,
        keep_score_idx: int,  # Note: will work only if beamsize == 1
        device: str = "cuda"
    ) -> None:

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        #print(self.pretrained_model_name_or_path)
        # import tokenizer from hugging face
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = self.pretrained_model_name_or_path
        )
        
        # import model from hugging face
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path = self.pretrained_model_name_or_path
        )

        self.keep_score_idx = keep_score_idx

        if device == "cuda":
            self.model.cuda()
        self.max_source_length = max_source_length
        self.model_batch_size = model_batch_size

    def predict(
        self,
        st1,
        st2, 
        binary = True
    ):

        args = ((st1, st2))

        inputs = self.tokenizer(
            *args, 
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            verbose=False,
        )
    
        with torch.no_grad():
            logits = self.model(**inputs.to(self.model.device))['logits']
            if binary:
                preds = np.argmax(logits.to('cpu'), axis=-1).tolist()
            else:
                return logits.tolist()
        return preds


def linearize_e2e_input(
    input: str,
    format: str ='gem'
) -> str:
    """
    Linearize an E2E input for QuestEval.
    Input must be a string, in standard E2E format.
    Example:
        'name[The Eagle], eatType[coffee shop], food[Japanese]'
    lowercase=True indicates that you want all tokens to be lowercased.
    """
    if format != 'gem':
        raise ValueError(f'Unsupported format for now: {format}')

    if not isinstance(input, str):
        raise WrongE2EFormat(input)

    items = dict([s.strip()[:-1].split('[') for s in input.split(',')])

    return ' , '.join([
        f'{key} [ {value} ]'
        for key, value in items.items()
    ])


def LinearizeDataInput(
    header: List,
    table: List[List],
):
    
    entites = []
    if header != None:
        for r in table:
            entity = [f'{clean_obj(h)} [ {clean_obj(v)} ]' for h, v in zip(header, r)]
            entites.append(' , '.join(entity))
    else:
        for r in table:
            entity = [f'[ {clean_obj(v)} ]' for v in r]
            entites.append(' , '.join(entity))
        
    return '. '.join(entites)


class LinearizeWebnlgInput():

    def __init__(
        self,
        spacy_pipeline,
        lowercase=False,
        format: str ='gem',
    ):
        """
        Linearize a WebNLG input for QuestEval.
        Input must be a list of triples, each being a string with two "|".
        Example:
            [
                "entity 1 | relation | entity 2" 
                "(15788)_1993_SB | discoverer | Donal_O'Ceallaigh",
                "(15788)_1993_SB | epoch | 2006-03-06"
            ]
        lowercase=True indicates that you want all strings to be lowercased.
        """

        self.lowercase = lowercase
        self.format = format
        self.spacy_pipeline = spacy_pipeline

    def __call__(
        self,
        input: List[str]
    )-> str:

        if self.format != 'gem':
            raise ValueError(f'Unsupported format for now: {self.format}')

        # isinstance used to determine whether "input" is a "list"
        if not isinstance(input, list):
            raise WrongWebNlgFormat(input)

        triples = [Triple(triple,
                          spacy_pipeline=self.spacy_pipeline,
                          lower=self.lowercase)
                   for triple in input]

        table = dict()
        for triple in triples:
            table.setdefault(triple.sbj, list())
            table[triple.sbj].append((triple.obj, triple.prp))

        ret = list()
        for entidx, (entname, entlist) in enumerate(table.items(), 1):
            ret.append(f'entity [ {entname} ]')
            for values, key in entlist:
                ret.append(f'{key} [ {values} ]')

        return ' , '.join(ret)


# split the raw text into triple value 1 | entity | value 2
class Triple:
    def __init__(
        self,
        raw_text: str,
        spacy_pipeline,
        lower: bool = False,
    ):
        sbj, prp, obj = self.safe_split(raw_text)
        # strip removes the space at the beginning and at the end.
        # ' '.join(["a", "b", "c"]) --> "a b c"
        # spacy_pipeline(a) tokenizer a 
        obj = ' '.join([t.text for t in spacy_pipeline(self.clean_obj(obj.strip(), lc=lower))])
        prp = self.clean_prp(prp.strip())
        sbj = ' '.join([t.text for t in spacy_pipeline(self.clean_obj(sbj.strip(), lc=lower))])
        if prp == 'ethnicgroup':
            obj = obj.split('_in_')[0]
            obj = obj.split('_of_')[0]

        self.sbj = sbj
        self.obj = obj
        self.prp = prp

    @staticmethod
    def safe_split(
        raw_text
    ) -> List[str]:

        if not isinstance(raw_text, str):
            raise TypeError('A triple must be a string with two "|"'
                            f'but you gave: {raw_text}')

        split = raw_text.strip().split('|')
        if not len(split) == 3:
            raise TypeError('A triple must be a string with two "|"'
                            f'but you gave: {raw_text}')

        return split

    def __repr__(self):
        return f'{self.sbj} | {self.prp} | {self.obj}'

    @staticmethod
    def clean_obj(
        s,
        lc: bool = False
    ):
        s = unidecode.unidecode(s)
        if lc: s = s.lower()
        s = re.sub('^"|"$', "", s)  # remove useless quotesigns
        s = re.sub('_', ' ', s)  # turn underscores to spaces
        return s

    @staticmethod
    def clean_prp(
        s: str,
        lc: bool=False
    ) -> str:
        s = unidecode.unidecode(s)
        if lc: s = s.lower()
        s = re.sub('^"|"$', "", s)  # remove useless quotesigns
        s = re.sub('\s+', '_', s)  # turn spaces to underscores
        s = re.sub('\s+\(in metres\)', '_m', s)
        s = re.sub('\s+\(in feet\)', '_f', s)
        s = re.sub('\(.*\)', '', s)
        return s.strip()


class WrongE2EFormat(
    Exception
):
    def __init__(self, obj):
        err = """
            It seems you passed an objected weirdly formatted.
            For E2E, please give a Meaning Representation as a string, 
            formatted as below:
                input = 'name[The Eagle], eatType[coffee shop], food[Japanese]'
            Your object was: {}
        """
        super().__init__(err.format(obj))


class WrongWebNlgFormat(Exception):
    def __init__(self, obj):
        err = """
            It seems you passed an objected weirdly formatted.
            For webnlg, please give a list of triplets, where each
            triplet is a string with two '|'.
            For instance:
                input = [
                    "(15788)_1993_SB | discoverer | Donal_O'Ceallaigh",
                    "(15788)_1993_SB | epoch | 2006-03-06"
                ]
            Your object was: {}
        """
        super().__init__(err.format(obj))


def calculate_f1_squad(
    a_gold: str,
    a_pred: str
) -> float:

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


def calculate_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def calculate_BERTScore(
    model_predictions: List[str],
    gold_references: List[str],
    metric_BERTScore,
    device: str,
) -> List[float]:

    if len(model_predictions) == 0:
        return []

    metric_BERTScore.add_batch(predictions=model_predictions, references=gold_references)
    final_score = metric_BERTScore.compute(model_type='bert-base-multilingual-cased', device=device)
    """
    # set all unanswerable scores to 0
    for i, (pred) in enumerate(model_predictions):
        if pred == "unanswerable":
            final_score['f1'][i] = 0.0
    """
    
    return [f1 for f1 in final_score['f1']]
    
