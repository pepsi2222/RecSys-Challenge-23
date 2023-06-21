import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, MLPModule


class IFMMLP(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.linear = ctr.LinearLayer(self.fields, train_data)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        model_config = self.config['model']
        num_fields = self.embedding.num_features
        self.fen = ctr.MLPModule(
                        [num_fields * self.embed_dim] + model_config['mlp_layer'],
                        model_config['activation'],
                        model_config['dropout'],
                        batch_norm=model_config['batch_norm'])
        self.fen.add_modules(
                    nn.Linear(model_config['mlp_layer'][-1], num_fields, bias=False),
                    nn.Softmax(dim=-1))
        self.fm = ctr.FMLayer(reduction='sum')
        
        self.mlp = MLPModule(
                        [self.embedding.num_features*self.embed_dim]+model_config['deep_mlp_layer']+[1],
                        activation_func = model_config['activation'],
                        dropout = model_config['deep_dropout'],
                        batch_norm = model_config['batch_norm'],
                        last_activation = False, last_bn=False)
        
    def score(self, batch):
        emb = self.embedding(batch)
        weight = self.fen(emb.flatten(1))
        lr_score = (super(ctr.LinearLayer, self.linear).forward(batch).squeeze(-1) * weight).sum(-1) + self.linear.bias
        fm_score = self.fm(emb * weight.unsqueeze(-1))
        
        deep_score = self.mlp(emb.flatten(1)).squeeze(-1)
        
        return {'score' : lr_score + fm_score + deep_score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
