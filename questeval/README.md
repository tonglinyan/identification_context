# Question generation

This folder contains several scripts that showcase how to fine-tune a T5 model on datasets SQuAD and WebNLG. 

## Fine-tuning T5 on SQuAD1.0

The [`qg_training.py`](https://github.com/tonglinyan/identification_context/blob/main/question-generation/qg_training.py) script leverages the Trainer for fine-tuning. T5 learns to generate the correct answer, rather than predicting the start and end position of the tokens of the answer.

Command for fine-tuning T5 on the SQuAD1.0 dataset.

```bash
python qg_training.py \
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

## Inference on WebNLG

The [`qg_inference.py`] is used to generate question with the sentences of WebNLG datasets using the model `t5_qg_squad2neg_en`, in order to build a question-answering dataset with triplets ans references of WebNLG. 

Command for inference 
```bash
python qg_inference.py
```

## Fine-tuning T5 on WebNLG
The last step is to fine-tune `t5_qg_squad2neg_en` on WebNLG.

Command for fine-tuning T5 on the SQuAD2.0 dataset.
```bash
python qg_finetuning.py \
   --model_name_or_path t5_qg_squad1_en \
   --train_file webnlg_qgqa_train.json \
   --context_column context \
   --question_column question \
   --answer_column answer \
   --do_train \
   --per_device_train_batch_size 12 \
   --learning_rate 3e-5  \
   --num_train_epochs 2 \
   --max_seq_length 384 \
   --doc_stride 128 \
   --output_dir t5_qg_webnlg_en
```


question answering finetuning on squadv2:
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
  
question generation finetuning on squadv1: 
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
question generation finetuning on webnlg:
```bash
python qg_finetuning.py \
  --model_name_or_path t5_qg_squad1_en \
  --loading_file data/webnlg.py
  --train_file data/webnlg_qgqa_train.json \
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
question answering finetuning on webnlg:
```bash
python qa_finetuning.py \
  --model_name_or_path t5_qa_squad2neg_en \
  --loading_file data/webnlg.py
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
question answering finetuning on newsqa:
```bash
python qa_finetuning.py \
  --model_name_or_path t5_qa_squad2neg_en \
  --loading_file data/newsqa.py \
  --train_file newsqa_train.csv \
  --context_column context \
  --question_column question \
  --answer_column answer \
  --do_train \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir t5_qa_newsqa_en
```
question answering finetuning on SQA:
```bash
python qa_finetuning.py \
  --model_name_or_path t5_qa_squad2neg_en \
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
  --output_dir t5_qa_sqa_en
```

question generation finetuning on SQA:
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
```bash
python qa_finetuning.py   --model_name_or_path t5_qa_squad2neg_en   --dataset_name wikitablequestions   --context_column table   --question_column question   --answer_column answers   --do_train   --per_device_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 384   --doc_stride 128   --output_dir t5_qa_wtq_en
```
```bash
python prediction.py --task BERT_Classification --author trained --dataset squad 
```

ranking
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
  --output_dir bert_ranking_non_msmarco \
```

```bash
python run_glue_no_trainer.py \
  --model_name_or_path bert-base-cased \
  --loading_file data/squad_rank_sent.py \
  --train_file train-v1.1.json \
  --validation_file dev-v1.1.json \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir bert_ranking_sent1
```

```bash
python baseline.py --dataset data/dev-v1.1.json
python pipeline.py --dataset wiki_bio
```
