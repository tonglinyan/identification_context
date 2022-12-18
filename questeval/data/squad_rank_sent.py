# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""



import csv
import os

import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    'train': "./train-v1.1.json", 
    'validation': "./dev-v1.1.json", 
}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class SQuADCLSDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="squad_rank", version=VERSION, description="Squad datasets used for passage ranking task"),
    ]

    DEFAULT_CONFIG_NAME = "train"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        features = datasets.Features(
            {
                "sentence1": datasets.Value("string"),
                "sentence2": datasets.Value("string"), 
                "label": datasets.Value("float32"),
                # These are the features of your dataset like images, labels ...
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir['train']), 
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir['validation']), 
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        import json
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        # bert classification 0.05
        dataset = data['data']
        key = 0
        from nltk.tokenize import sent_tokenize
        
        for d in dataset:
            
            paras = d['paragraphs']

            corpus, s1 = [], []
            #for p in paras:
            #    corpus.append(p['context'])

            for p in paras:  
                corpus = sent_tokenize(p['context'])
                
                for qa_pair in p['qas']:        
                    q = qa_pair['question']
                    answers = qa_pair['answers']
                    
                    start_idx = list(set([a['answer_start'] for a in answers]))
                    asw_text = ', '.join(list(set([a['text'] for a in answers]))) if len(answers) > 0 else " " 
                    s1 = f'question: {q} answer: {asw_text}'

                    len_sent = 0
                    for c in corpus:
                        label = 0
                        for s in start_idx:
                            if s >= len_sent and s < len_sent+len(c)+1:
                                label = 1

                        yield key, {
                            'sentence1': s1,
                            'sentence2': c,
                            'label': label,
                        }
                        key += 1
                        len_sent += len(c)+1

                            
            """
            corpus = list(set(context))
            labels, s1, s2 = [], [], []
            import random
            for i in range(len(pairs)):
                ctx = random.sample(corpus, 5)
                if context[i] not in ctx:
                    ctx.append(context[i])
                labs = [1 if c == context[i] else 0 for c in ctx]
                
                labels += labs
                s1 += [pairs] * len(ctx)
                s2 += ctx

            yield key, {
                    'sentence1': s1,
                    'sentence2': s2,
                    'label': l,
                }
            """