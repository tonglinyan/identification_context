import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn


class Learner:
    """Base class for supervised learning"""

    def __init__(self, model, model_id: str):
        super().__init__()
        self.model = model
        self.optim = torch.optim.Adam(model.parameters(),lr=1e-4)
        self.model_id = model_id
        self.iteration = 0

    def run(self,train_loader, test_loader, epochs, test_iterations, device,entropy_pen=0.):
        """Run a model during `epochs` epochs"""
        writer = SummaryWriter(f"/tmp/runs/{self.model_id}")
        model = self.model.to(device)
        loss = nn.CrossEntropyLoss()
        loss_nagg = nn.CrossEntropyLoss(reduction='sum')

        model.train()
        for epoch in tqdm(range(epochs)):
            # Iterate over batches
            for x, y,lens in train_loader:
                self.optim.zero_grad()
                yhat = model(x.to(device),lens.to(device))
                l = loss(yhat, y.to(device))
                probs = model.attention(model.emb(x.to(device)),lens.to(device))
                entrop = -(probs*(probs+1e-10).log()).sum(1).mean()
                total_l = l+entropy_pen*entrop
                total_l.backward()
                self.optim.step()
                writer.add_scalar('loss/train', l, self.iteration)
                writer.add_scalar('loss/entrop',entrop,self.iteration)
                writer.add_scalar('loss/total_train',total_l,self.iteration)
                self.iteration += 1
                
                if self.iteration % test_iterations == 0:
                    model.eval()
                    with torch.no_grad():
                        lst_probs = []
                        cumloss = 0
                        cumcorrect = 0
                        count = 0
                        for x, y, lens in test_loader:
                            yhat = model(x.to(device),lens.to(device))
                            cumloss += loss_nagg(yhat, y.to(device))
                            cumcorrect += (yhat.argmax(1) == y.to(device)).sum()
                            count += x.shape[0]
                            probs =  model.attention(model.emb(x.to(device)),lens.to(device))
                            lst_probs.append(-(probs*(probs+1e-10).log()).sum(1))

                        writer.add_scalar(
                            'loss/test', cumloss.item() / count, self.iteration)
                        writer.add_scalar(
                            'correct/test', cumcorrect.item() / count, self.iteration)
                        
                        writer.add_histogram(f'entropy',torch.cat(lst_probs),self.iteration)
                        
                    model.train()


def collate(batch):
    """ Collate function for DataLoader """
    data = [torch.LongTensor(item[0]) for item in batch]
    lens = [len(d) for d in data]
    labels = [item[1] for item in batch]
    return torch.nn.utils.rnn.pad_sequence(data, batch_first=True,padding_value = PAD), torch.LongTensor(labels), torch.Tensor(lens)

