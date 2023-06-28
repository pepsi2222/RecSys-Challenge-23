from ..basemodel import BaseRanker
from ..module import ctr
from ..loss_func import BCELoss
from recstudio.data.dataset import TripletDataset
import torch


class LR3(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.linear = SoftmaxLinearLayer(self.fields, train_data)

    def _get_loss_func(self):
        return BCELoss()

    def score(self, batch):
        return {'score' : self.linear(batch)}
    

class SoftmaxLinearLayer(torch.nn.Module):
    def __init__(self, fields, data):
        super().__init__()
        if not isinstance(data.frating, list):
            self.field2types = {f: data.field2type[f] for f in fields if f != data.frating}
        else:
            self.field2types = {f: data.field2type[f] for f in fields if f not in data.frating}
        self.weight = torch.nn.Parameter(torch.ones(len(self.field2types)))   # self.frating

    def forward(self, batch):
        probs = []
        for f, t in self.field2types.items():
            prob = batch[f]
            probs.append(prob)
        probs = torch.stack(probs, dim=-1)
        weighted_probs = probs * self.weight.softmax(0)
        sum_of_probs = torch.sum(weighted_probs, dim=-1)
        return sum_of_probs
