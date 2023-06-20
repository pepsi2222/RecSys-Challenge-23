from ..basemodel import BaseRanker
from ..module import ctr
from ..loss_func import BCEWithLogitLoss
from recstudio.data.dataset import TripletDataset
import torch


class LR2(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.linear = SoftmaxLinearLayer(self.fields, train_data)

    def _get_loss_func(self):
        return BCEWithLogitLoss()

    def score(self, batch):
        return {'score' : self.linear(batch)}
    

class SoftmaxLinearLayer(ctr.Embeddings):
    def __init__(self, fields, data):
        super().__init__(fields, 1, data)
        self.weight = torch.nn.Parameter(torch.ones(len(fields) - 1).softmax(0))   # self.frating

    def forward(self, batch):
        # input: [B, num_fields, 1]
        embs = super().forward(batch).squeeze(-1)
        weighted_embs = embs * self.weight.softmax(0)
        sum_of_embs = torch.sum(weighted_embs, dim=-1)
        return sum_of_embs
