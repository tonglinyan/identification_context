import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import Encoder
from tokenization import (
    QuestionContextDataset, 
    lettre2id,
    word2id)
import numpy as np
from torch.utils.data import DataLoader
import argparse
import sys
from datasets import load_dataset
import logging
import pathlib

PATH = pathlib.Path('./model')

def parse_args():
    parser = argparse.ArgumentParser('Parameters for training model')
    parser.add_argument('--dataset', default="squad", type=str)
    parser.add_argument('--batch_size', help='batch-size', default=32, type=int)
    parser.add_argument('--test_iterations', default=100, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--hidden_size', default=50, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def preprocess_data():
    raw_dataset = load_dataset("squad")
    
    def generate_pairs(data):
        questions = data["question"]
        answers = [d['text'][0] for d in data['answers']]
        context = data['context']
        return [f'question: {q} answer: {a}. ' for q, a in zip(questions, answers)], context, answers

    data = {'train':{}, 'test':{}}
    data['train']['questions'], data['train']['context'], data['train']['answers'] = generate_pairs(raw_dataset['train'][:int(0.1*len(raw_dataset['train']))])
    data['test']['questions'], data['test']['context'], data['test']['answers'] = generate_pairs(raw_dataset['validation'])

    return data['train'], data['test']


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PAD_word = word2id["<unk>"]
    PAD_char = lettre2id["<unk>"]
    model = Encoder(args.hidden_size, args.dropout)
    optim = torch.optim.Adam(model.parameters(),lr=1e-4)
    loss = torch.nn.CrossEntropyLoss()


    def collate(batch):
        """ Collate function for DataLoader """
        q_word = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True, padding_value=PAD_word)
        p_word = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True, padding_value=PAD_word)
        
        q_len = q_word.size(1) ## max number of words in questions
        p_len = p_word.size(1) ## max number of words in sentences

        q_char, p_char = [], []
        
        for item in batch:
            q_char += item[1]
            q_char += [torch.tensor([lettre2id['<unk>']])] * (q_len - len(item[1]))
            p_char += item[3]
            p_char += [torch.tensor([lettre2id['<unk>']])]  * (p_len - len(item[3]))

        q_char = torch.nn.utils.rnn.pad_sequence(q_char, batch_first=True, padding_value=PAD_char)
        p_char = torch.nn.utils.rnn.pad_sequence(p_char, batch_first=True, padding_value=PAD_char)

        labels = [item[4] for item in batch]
        return q_word, q_char, p_word, p_char, torch.LongTensor(labels)#, torch.Tensor(lens)
        
 

    def train(train_loader, model, device):
        """Run a model during `epochs` epochs"""
        iteration = 0
        writer = SummaryWriter(f"/tmp/runs/s-net")
        model = model.to(device)

        model.train()
        for epoch in tqdm(range(args.epochs)):
            # Iterate over batches
            cum_loss = 0
            for x in tqdm(train_loader):
                optim.zero_grad()
                yhat = model(x[0].to(device), x[1].to(device), x[2].to(device), x[3].to(device))
                l = loss(yhat, x[4].to(device))
                cum_loss += l/len(train_loader)
                l.backward()
                optim.step()
                writer.add_scalar('loss/train', l, iteration)
                writer.add_scalar('loss/total_train', cum_loss, iteration)
                
                iteration += 1

                if iteration % args.test_iterations == 0:
                    print(f"loss train : {float(cum_loss)} -- {iteration}")
                #logging.info("loss train : %f -- %f", cum_loss, iteration)

                """
                model.eval()
                with torch.no_grad():
                    cumloss = 0
                    for x in tqdm(test_loader):
                        yhat = model(x[0].to(device), x[1].to(device), x[2].to(device), x[3].to(device))
                        l = loss(yhat, x[4].to(device))
                        cum_loss += l/len(test_loader)
                    
                    writer.add_scalar('loss/test', cumloss, iteration)
                """
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(), 
                            'optimiser_state_dict': optim.state_dict(), 
                            'loss': cum_loss}, f'{PATH}/checkpoint_{iteration}.pt')
        torch.save(model, f'{PATH}/model.pt')
        torch.save(model.state_dict(), f'{PATH}/state_dict.pt')
    

    def eval(test_loader, model, device):
        iteration = 0
        writer = SummaryWriter(f"/tmp/runs/s-net")
        model = model.to(device)

        model.eval()

        cumloss = 0
        yhat, y = [], []
        for x in tqdm(test_loader):
            yh = model(x[0].to(device), x[1].to(device), x[2].to(device), x[3].to(device))
            l = loss(yh, x[4].to(device))
            cum_loss += l/len(test_loader)
            yhat += yh.tolist()
            y += x[4]

        
        writer.add_scalar('loss/test', cumloss, iteration)

        def evaluation(preds, truth):
            cls = [0 if p[0] > p[1] else 1 for p in preds]
            same = [1 if c == t else 0 for (c, t) in zip(cls, truth)]
            return same.sum()/len(same)

        print(evaluation(yhat, y))

    train_data, test_data = preprocess_data()

    if args.do_train:   
        try:
            model = torch.load()
        except:
            print("create a new model.")
            model = Encoder(args.hidden_size, args.dropout)
        else:
            print("sucessfully import the model")
        train_data = QuestionContextDataset(train_data['questions'], train_data['context'], train_data['answers'])
        train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, collate_fn=collate)
        train(train_loader, model, device)
        
    if args.do_eval:
        model = torch.load(f'{PATH}/model.pt')

        test_data = QuestionContextDataset(test_data['questions'], test_data['context'], test_data['answers'])
        test_loader = DataLoader(test_data, shuffle=True, batch_size=args.batch_size, collate_fn=collate)
        eval(test_loader, model, device)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)

