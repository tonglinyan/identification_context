import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import Encoder
from tokenization import (
    QuestionContextDataset, 
    word2id)
import numpy as np
from torch.utils.data import DataLoader
import argparse
import sys
from datasets import load_dataset


def preprocess_data():
    raw_dataset = load_dataset("squad")
    
    def generate_pairs(data):
        questions = data["question"]
        answers = [d['text'][0] for d in data['answers']]
        context = data['context']
        return [f'question: {q} answer: {a}' for q, a in zip(questions, answers)], context, answers

    data = {'train':{}, 'test':{}}
    data['train']['questions'], data['train']['context'], data['train']['answers'] = generate_pairs(raw_dataset['train'])
    data['test']['questions'], data['test']['context'], data['test']['answers'] = generate_pairs(raw_dataset['validation'])

    return data['train'], data['test']
    

def parse_args():
    parser = argparse.ArgumentParser('Evaluation script for each model in questeval.')
    parser.add_argument('--dataset', default="squad", type=str)
    parser.add_argument('--batch_size', help='batch-size', default=32, type=int)
    parser.add_argument('--test_iterations', default=20, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--emb_size', default=150, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(epochs, test_iterations, batch_size, emb_size, dropout):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PAD = word2id["<unk>"]
    model = Encoder(emb_size, dropout)
    optim = torch.optim.Adam(model.parameters(),lr=1e-4)
    loss = torch.nn.CrossEntropyLoss()

    train_data, test_data = preprocess_data()

    train_data = QuestionContextDataset(train_data['questions'], train_data['context'], train_data['answers'])
    test_data = QuestionContextDataset(test_data['questions'], test_data['context'], test_data['answers'])

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=collate)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, collate_fn=collate)

    def collate(batch):
        """ Collate function for DataLoader """
        data = [torch.LongTensor(item[0]) for item in batch]
        lens = [len(d) for d in data]
        labels = [item[1] for item in batch]
        return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value = PAD), torch.LongTensor(labels), torch.Tensor(lens)


    def train(train_loader, test_loader, epochs, test_iterations, device):
        """Run a model during `epochs` epochs"""
        iteration = 0
        writer = SummaryWriter(f"/tmp/runs/s-net")
        model = model.to(device)

        model.train()
        for epoch in tqdm(range(epochs)):
            # Iterate over batches
            cum_loss = 0
            for x in train_loader:
                optim.zero_grad()
                yhat = model(x[0].to(device), x[1].to(device), x[2].to(device), x[3].to(device))
                l = loss(yhat, x[4].to(device))
                cum_loss += l/len(train_loader)
                l.backward()
                optim.step()
                writer.add_scalar('loss/train', l, iteration)
                writer.add_scalar('loss/total_train', cum_loss, iteration)
                iteration += 1
                
                if iteration % test_iterations == 0:
                    model.eval()
                    with torch.no_grad():
                        lst_probs = []
                        cumloss = 0
                        for x, y, lens in test_loader:
                            yhat = model(x.to(device),lens.to(device))
                            cumloss += LossFunction(yhat, y.to(device))/len(test_loader)
                            
                        writer.add_scalar(
                            'loss/test', cumloss, iteration)
                        writer.add_histogram(f'entropy',torch.cat(lst_probs),iteration)
                        
                    model.train()

    train(train_loader, test_loader, epochs, test_iterations, device)

if __name__ == "__main__":
    args = parse_args()
    main(args.epochs, args.test_iterations, args.batch_size, args.emb_size, args.dropout)

