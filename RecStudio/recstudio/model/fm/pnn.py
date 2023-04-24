import torch
import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, MLPModule

r"""
PNN
######################

Paper Reference:
    Product-based Neural Networks for User Response Prediction (ICDM'16)
    https://ieeexplore.ieee.org/document/7837964
"""

class PNN(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        model_config = self.config['model']
        num_fields = self.embedding.num_features
        if model_config['stack_dim'] is None:
            if model_config['product_type'].lower() == 'inner':
                self.prod_layer = ctr.InnerProductLayer(num_fields)
                mlp_in_dim = num_fields * (num_fields - 1) // 2 + num_fields * self.embed_dim
            elif model_config['product_type'].lower() == 'outer':
                self.prod_layer = ctr.OuterProductLayer(num_fields)
                mlp_in_dim = (num_fields * (num_fields - 1) // 2) * self.embed_dim * self.embed_dim + num_fields * self.embed_dim
            else:
                raise ValueError(f'Expect product_type to be `inner` or `outer`, but got {model_config["product_type"]}.')
        else:
            self.Wz = nn.Parameter(torch.randn(num_fields * self.embed_dim, model_config['stack_dim']))
            if model_config['product_type'].lower() == 'inner':
                self.Thetap = nn.Parameter(torch.randn(num_fields, model_config['stack_dim']))
            elif model_config['product_type'].lower() == 'outer':
                self.Wp = nn.Parameter(torch.randn(self.embed_dim, self.embed_dim, model_config['stack_dim']))
            else:
                raise ValueError(f'Expect product_type to be `inner` or `outer`, but got {model_config["product_type"]}.')
            self.bias = nn.Parameter(torch.randn(model_config['stack_dim']))
            mlp_in_dim = 2 * model_config['stack_dim']
            
        self.mlp = MLPModule(
                    [mlp_in_dim] + model_config['mlp_layer'] + [1],
                    model_config['activation'], model_config['dropout'],
                    batch_norm=model_config['batch_norm'],
                    last_activation=False, last_bn=False)

    def score(self, batch):
        emb = self.embedding(batch)                                         # B x F x D
        if self.config['model']['stack_dim'] is None:
            lz = emb.flatten(1)                                             # B x F*D
            lp = self.prod_layer(emb)                                       # B x num_pairs
        else:
            lz = (self.Wz * emb.view(emb.size(0), -1, 1)).sum(1)            # B x S
            if self.config['model']['product_type'] == 'inner':
                delta = torch.einsum('fs,bfd->bfsd', [self.Thetap, emb])    # B x F x S x D
                lp = (delta.sum(1)**2).sum(-1)                              # B x S
            elif self.config['model']['product_type'] == 'outer':
                p = torch.einsum('bi,bj->bij', 2 * [emb.sum(1)])            # B x D x D
                lp = torch.einsum('bij,ijs->bs', [p, self.Wp])              # B x S
        mlp_in = torch.cat([lz, lp], dim=1)
        score = self.mlp(mlp_in).squeeze(-1)
        return {'score' : score}

    def _get_loss_func(self):
        return loss_func.BCEWithLogitLoss()
