import torch
import torch.nn as nn
from collections import defaultdict
from recstudio.data.dataset import TripletDataset
from ..basemodel import BaseRanker
from ..loss_func import BCEWithLogitLoss
from ..module import ctr, MLPModule

class PLEDCNv2(BaseRanker):

    def _get_dataset_class():
        return TripletDataset

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        model_config = self.config['model']
        self.embedding = ctr.Embeddings(self.fields, self.embed_dim, train_data)
        assert isinstance(self.frating, list), f'Expect rating_field to be a list, but got {self.frating}.'
        self.extraction_layers = nn.Sequential(*[
                                    ExtractionLayer(
                                        self.embedding.num_features * self.embed_dim, # if i == 0 else model_config['expert_mlp_layer'][-1],
                                        model_config['specific_experts_per_task'],
                                        len(self.frating),
                                        model_config['num_shared_experts'],
                                        True if i != model_config['num_levels'] - 1 else False,
                                        model_config['expert_num_layers'],
                                        # model_config['expert_mlp_layer'],
                                        # model_config['expert_activation'],
                                        # model_config['expert_dropout'],
                                        model_config['gate_mlp_layer'],
                                        model_config['gate_activation'],
                                        model_config['gate_dropout'])
                                    for i in range(model_config['num_levels'])
                                ])
        self.towers = nn.ModuleDict({
                            r: MLPModule(
                                [self.embedding.num_features * self.embed_dim] + model_config['tower_mlp_layer'] + [1],
                                model_config['tower_activation'], 
                                model_config['tower_dropout'],
                                batch_norm=model_config['tower_batch_norm'],
                                last_activation=False, 
                                last_bn=False)
                            for r in self.frating
                        })
            
    def score(self, batch):
        emb = self.embedding(batch).flatten(1)
        extraction_out = self.extraction_layers([emb] * (len(self.frating) + 1))
        score = defaultdict(dict)
        for i, (r, tower) in enumerate(self.towers.items()):
            score[r]['score'] = tower(extraction_out[i]).squeeze(-1)
        return score

    def _get_loss_func(self):
        return BCEWithLogitLoss()
    
    def training_step(self, batch):
        '''Same feature fields for all ratings and no reweighting'''
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
    


class ExtractionLayer(nn.Module):
    def __init__(self, in_dim, specific_experts_per_task, num_task, num_shared_experts, share_gate,
                 expert_num_layers,
                #  expert_mlp_layer, expert_activation, expert_dropout, 
                 gate_mlp_layer, gate_activation, gate_dropout):
        super().__init__()
        self.specific_experts_per_task = specific_experts_per_task
        self.num_task = num_task
        self.num_shared_experts = num_shared_experts
        self.share_gate = share_gate
        self.specific_experts = nn.ModuleList([
                                    nn.ModuleList([
                                        ctr.CrossNetworkV2(in_dim, expert_num_layers)
                                        for _ in range(specific_experts_per_task)
                                    ])
                                    for _ in range(num_task)
                                ])
        self.shared_experts = nn.ModuleList([
                                ctr.CrossNetworkV2(in_dim, expert_num_layers)
                                for _ in range(num_shared_experts)
                            ])
        self.gates = nn.ModuleList([
                        MLPModule(
                            [in_dim] + gate_mlp_layer + [specific_experts_per_task + num_shared_experts],
                            gate_activation, 
                            gate_dropout)
                        for _ in range(num_task)
                    ])
        for g in self.gates:
            g.add_modules(nn.Softmax(-1))
            
        if share_gate:
            self.shared_gates = MLPModule(
                                    [in_dim] + gate_mlp_layer + [num_task * specific_experts_per_task + num_shared_experts],
                                    gate_activation, 
                                    gate_dropout)
            self.shared_gates.add_modules(nn.Softmax(-1))
    
    def forward(self, inputs):
        experts_out = []
        for i, experts_per_task in enumerate(self.specific_experts):
            experts_out.append(torch.stack([e(inputs[i]) for e in experts_per_task], dim=1))   # B x SpecificPerTask x De
                
        shared_e_out = torch.stack(
                        [shared_e(inputs[-1]) for shared_e in self.shared_experts], dim=1)      # B x Shared x De
        
        outputs = []
        for i, (g, e_out) in enumerate(zip(self.gates, experts_out)):
            gate_out = g(inputs[i])        # B x (SpecificPerTask + Shared)
            outputs.append((gate_out.unsqueeze(-1) * torch.cat([e_out, shared_e_out], dim=1)).sum(1))    # B x De
        
        if self.share_gate:
            shared_gate_out = self.shared_gates(inputs[-1])
            e_out = torch.cat(experts_out, dim=1)                                               # B x num_task*SpecificPerTask x De
            outputs.append((shared_gate_out.unsqueeze(-1) * torch.cat([e_out, shared_e_out], dim=1)).sum(1))
        return outputs
    
    def extra_repr(self):
        return f'specific_experts_per_task={self.specific_experts_per_task}, ' + \
                f'num_task={self.num_task}, num_shared_experts={self.num_shared_experts}'
