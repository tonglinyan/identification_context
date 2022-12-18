# Identification of context 
The idea of this project is to generate a dataset composed of tables (structured data)-text pairs in the biological domain. Given the structured data and the passages that describe the data, we propose to identify a paragraph or a sentence which corresponds to a value in the table. The pipeline consists of a Question-Generation (QG) system, generating a question from a linearized answer-table pair, and an inverse Question-Answering (QA)system, finding the correct paragraph which describes the question-answer pair.


## 1. Question generation (QuestEval)

The Question-generation component is referred to a part of metric [QuestEval](https://github.com/ThomasScialom/QuestEval).
### Training of QuestEval

The overall process of training in QuestEval is composed of 4 steps:
1. textual QG/QA: we train a textual QG and QA model on SQuAD.
2. Synthetic Questions: Given any data-to-text dataset, consisting of pairs (structured input, textual description), we generate synthetic questions for each textual description using QG textual.
3. Multimodal synthetic dataset: Each example in the dataset consists of
    - the source (i.e. the structured data),
    - the textual description,
    - the question-answer pairs generated in step 2.
  We then match each structured input to its corresponding synthetic question-answer pair to construct a multimodal dataset.
4. Multimodal QG/QA: With multimodal dataset, we train the multimodal QG and QA model with context-answer and context-question pairs respectively, where the context is the linearized table instead of the sentences.

We introduce how to fine-tune a T5 model on datasets SQuAD and WebNLG. 


##### QG/QA fine-tuned on SQuAD

The `run_seq2seq_qa.py` and `run_seq2seq_qg.py` scripts leverage the Trainer for fine-tuning on QG/QA. T5 learns to generate the correct answer, rather than predicting the start and end position of the tokens of the answer.

Command for fine-tuning QA on the SQuAD2.0 dataset.
```bash
python run_seq2seq_qa.py \
  --model_name_or_path t5-small \
  --dataset_name squad_v2 \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \vi
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir t5_qa_squad2neg_en
```
  
Command for fine-tuning QG on the SQuAD1.0 dataset.
```bash
python run_seq2seq_qg.py \
  --model_name_or_path t5-small \
  --dataset_name squad \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir t5_qg_squad1_en
```

##### Inference on WebNLG

The [qg_inference.py] is used to generate question with the sentences of WebNLG datasets using the model `t5_qg_squad2neg_en`, in order to build a question-answering dataset with triplets ans references of WebNLG. 

Command for inference 
```bash
python generation_webnlg.py
```

##### QG/QA fine-tuned on WebNLG
The last step is to fine-tune `t5_qa_squad2neg_en` and `t5_qg_squad_en` on WebNLG.

Command for fine-tuning QA on the synthetic WebNLG dataset.

```bash
python qa_finetuning.py \
  --model_name_or_path t5_qa_squad2neg_en \
  --loading_file webnlg.py
  --train_file webnlg_qgqa_train.json \
  --context_column linearization \
  --question_column question \
  --answer_column answer \
  --do_train \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir t5_qa_webnlg_en
```

Command for fine-tuning QG on the synthetic WebNLG dataset.
```bash
python qg_finetuning.py \
  --model_name_or_path t5_qg_squad1_en \
  --loading_file data/webnlg.py \
  --train_file webnlg_qgqa_train.json \
  --context_column linearization \
  --question_column question \
  --answer_column answer \
  --do_train \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir t5_qg_webnlg_en
```

##### QG fine-tuned on SQA
Also, by changing the parameters `dataset_name`, `question_column`, `context_column` and 
`answer_column`, we can finetune the pre-trained QG on the other dataset.

```bash
python qg_finetuning.py \
  --model_name_or_path t5_qg_squad1_en \
  --dataset_name msr_sqa \
  --question_column question \
  --context_column table_header \
  --answer_column answer_text \
  --do_train \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir t5_qg_sqa_en
```


## 2. Identification of relevant context

The system of identifying context is trained with the script `classifcation.py`. To train for the different tasks (classification or ranking), it's necessary to change the parameter `loading_file` with `data/squad_rank.py`, `squad_rank_sent.py`, `squad_cls.py` or `squad_cls_sent.py`, which imports different scripts for generating the training dataset.

Command for fine-tuning BERT on Classification or Ranking tasks.
```bash
python classification.py \
  --model_name_or_path bert-base-cased \
  --loading_file data/squad_rank.py \
  --train_file train-v1.1.json \
  --validation_file dev-v1.1.json \
  --do_train \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir bert_ranking \
```

The script `prediction.py` is used to evaluate the performance of all the models with specifying `task`, `dataset`. 
```bash
python prediction.py 
  --task QG_D2T \
  --dataset squad 
```
The alternative options for `--task` are: `QG_T2T`, `QG_D2T`, `QA_T2T`, `QA_D2T`, `BERT_Classification`, `BERT_Ranking`, `BERT_Classification_sent`, `BERT_Ranking_sent`.
The alternative options for `--dataset` are: `sqaud`, `sqaudv2`, `webnlg`, `sqa`, `wtq`.


## 3. Using the pipeline

The last step is to run the pipeline with command on the dataset `wiki_bio`, `totto` or `rotowire`. 
```bash
python pipeline.py -d wiki_bio  -f
```

