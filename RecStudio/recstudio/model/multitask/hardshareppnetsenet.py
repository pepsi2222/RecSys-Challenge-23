import torch
import torch.nn as nn
from recstudio.data.dataset import TripletDataset
from .. import loss_func
from ..basemodel import BaseRanker
from ..module import ctr, MLPModule
from collections import defaultdict
import torch.nn.functional as F

class BCEWithLogitLossWithAux(loss_func.BCEWithLogitLoss):
    def forward(self, aux_score, label, pos_score):
        if aux_score is not None:
            return super().forward(label, aux_score) + super().forward(label, pos_score)
        else:
            return super().forward(label, pos_score)

class HardSharePPNetSEnet(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        self.mlp = MLPModule([self.embedding.num_features*self.embed_dim] + model_config['mlp_layer'] + [1],
                        model_config['activation'], 
                        model_config['dropout'],
                        last_activation=False, 
                        last_bn=False
                    )
        if model_config['id_fields'] is None:
            id_fields = []
            if self.fuid is not None:
                id_fields.append(self.fuid)
            if self.fiid is not None:
                id_fields.append(self.fiid)
            if len(id_fields) == 0:
                raise ValueError('Expect id_fields, but got None.')
        else:
            id_fields = model_config['id_fields']
        self.id_embedding = ctr.Embeddings(id_fields, model_config['id_embed_dim'], train_data)
        pp_hidden_dims1 = [self.embedding.num_features*self.embed_dim] + model_config['pp_hidden_dims1']
        self.ppnet1 = nn.ModuleList([
                        ctr.PPLayer(
                            pp_hidden_dims1[i : i + 2],
                            self.embedding.num_features*self.embed_dim + len(id_fields)*model_config['id_embed_dim'],
                            model_config['gate_hidden_dims1'][i],
                            model_config['activation'],
                            model_config['dropout1'],
                            model_config['batch_norm']) 
                        for i in range(len(pp_hidden_dims1) - 1)
                    ])
        pp_hidden_dims2 = [self.embedding.num_features*self.embed_dim] + model_config['pp_hidden_dims2']
        self.ppnet2 = nn.ModuleList([
                        ctr.PPLayer(
                            pp_hidden_dims2[i : i + 2],
                            self.embedding.num_features*self.embed_dim + len(id_fields)*model_config['id_embed_dim'],
                            model_config['gate_hidden_dims2'][i],
                            model_config['activation'],
                            model_config['dropout2'],
                            model_config['batch_norm']) 
                        for i in range(len(pp_hidden_dims2) - 1)
                    ])
        self.fc1 = nn.Linear(pp_hidden_dims1[-1], 1)
        self.fc2 = nn.Linear(pp_hidden_dims2[-1], 1)
        self.senet = ctr.SqueezeExcitation(
                        len(self.fields) - 2, 
                        model_config['reduction_ratio'], 
                        model_config['excitation_activation'])

    def score(self, batch):
        emb = self.senet(self.embedding(batch))
        mlp_score = self.mlp(emb.flatten(1)).squeeze(-1)
        
        id_emb = self.id_embedding(batch)
        gate_in = torch.cat([emb.flatten(1).detach(), id_emb.flatten(1)], dim=-1)
        mlp_in1 = mlp_in2 = emb.flatten(1).detach()
        for pplayer in self.ppnet1:
            mlp_in1 = pplayer(gate_in, mlp_in1)
        for pplayer in self.ppnet2:
            mlp_in2 = pplayer(gate_in, mlp_in2)
        ppnet_score1 = self.fc1(mlp_in1).squeeze(-1)
        ppnet_score2 = self.fc2(mlp_in2).squeeze(-1)
        score = defaultdict(dict)
        score['is_clicked']['score'] = ppnet_score1
        score['is_installed']['score'] = ppnet_score2
        score['is_clicked']['aux_score'] = None
        score['is_installed']['aux_score'] = mlp_score
        return score

    def _get_loss_func(self):
        return BCEWithLogitLossWithAux()
    
    def training_step(self, batch):
        y_h, output = self.forward(batch)
        loss = {
            'is_clicked': self.loss_fn(output['is_clicked']['aux_score'], **y_h['is_clicked']),
            'is_installed': self.loss_fn(output['is_installed']['aux_score'], **y_h['is_installed'])
            }
        
        weights = self.config['train'].get('weights', [1.0]*len(self.frating))
        if weights is None:
            weights = [1.0]*len(self.frating)
        assert len(weights) == len(self.frating), \
            f'Expect {len(self.frating)} float(s) for weights, but got {self.config["train"]["weights"]}.'
        weights = torch.tensor(weights, device=self.device).softmax(0)
        
        loss['loss'] = sum(w*v for w, (_, v) in zip(weights, loss.items()))
        return loss

