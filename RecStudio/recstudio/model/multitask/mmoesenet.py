import torch
import torch.nn as nn
from collections import defaultdict
from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule


class MMoESEnet(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        assert isinstance(self.frating, list), f'Expect rating_field to be a list, but got {self.frating}.'
        self.experts = nn.ModuleList([
                            MLPModule(
                                [self.embedding.num_features * self.embed_dim] + model_config['expert_mlp_layer'],
                                model_config['expert_activation'], 
                                model_config['expert_dropout'],
                                batch_norm=model_config['expert_batch_norm'])
                            for _ in range(model_config['num_experts'])
                        ])
        self.gates = nn.ModuleDict({
                            r: MLPModule(
                                [self.embedding.num_features * self.embed_dim] + model_config['gate_mlp_layer'] + [model_config['num_experts']],
                                model_config['gate_activation'], 
                                model_config['gate_dropout'],
                                batch_norm=model_config['gate_batch_norm'])
                            for r in self.frating
                        })
        for _, g in self.gates.items():
            g.add_modules(nn.Softmax(-1))
        self.towers = nn.ModuleDict({
                            r: MLPModule(
                                [model_config['expert_mlp_layer'][-1]] + model_config['tower_mlp_layer'] + [1],
                                model_config['tower_activation'], 
                                model_config['tower_dropout'],
                                batch_norm=model_config['tower_batch_norm'],
                                last_activation=False, 
                                last_bn=False)
                            for r in self.frating
                        })
        self.senet = ctr.SqueezeExcitation(
                        len(self.fields) - 1, 
                        model_config['reduction_ratio'], 
                        model_config['excitation_activation'])
            
    def score(self, batch):
        emb = self.senet(self.embedding(batch)).flatten(1)
        experts_out = torch.stack([e(emb) for e in self.experts], dim=1)        # B x E x De
        score = defaultdict(dict)
        for r, gate in self.gates.items():
            gate_out = gate(emb)                                                # B x E
            mmoe_out = (gate_out.unsqueeze(-1) * experts_out).sum(1)            # B x De
            score[r]['score'] = self.towers[r](mmoe_out).squeeze(-1)
        return score

    def _get_loss_func(self):
        return BCEWithLogitLoss()
    
    def training_step(self, batch):
        y_h, _ = self.forward(batch)
        loss = {}
        for r in self.frating:
            loss[r] = self.loss_fn(**y_h[r])
        
        weights = self.config['train'].get('weights', [1.0]*len(self.frating))
        if weights is None:
            weights = [1.0]*len(self.frating)
        assert len(weights) == len(self.frating), \
            f'Expect {len(self.frating)} float(s) for weights, but got {self.config["train"]["weights"]}.'
        weights = torch.tensor(weights, device=self.device).softmax(0)
        
        loss['loss'] = sum(w*v for w, (_, v) in zip(weights, loss.items()))
        return loss
