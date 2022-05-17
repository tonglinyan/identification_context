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

