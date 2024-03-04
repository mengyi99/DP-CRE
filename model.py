import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel, BertConfig
import json
import os

class base_model(nn.Module):
    def __init__(self):
        super(base_model, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict=False)
        self.eval()

    def get_parameters(self, mode="numpy", param_dict=None):
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:
                res[param] = all_param_dict[param]
        return res

    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()


class Classifier(base_model):
    def __distance__(self, logits, rel):
        pairwise_distances_squared = torch.sum(logits ** 2, dim=1, keepdim=True) + \
                                 torch.sum(rel.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(logits, rel.t())
        error_mask = pairwise_distances_squared <= 0.0
        pairwise_distances = torch.clamp(pairwise_distances_squared, min=1e-16)#.sqrt()
        pairwise_distances = torch.mul(pairwise_distances, ~error_mask)
        return pairwise_distances


    def __init__(self, sentence_encoder,num_class, id2rel, config = None):
        super(Classifier, self).__init__()
        self.config = config
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.hidden_size = self.sentence_encoder.output_size
        self.id2rel = id2rel
        self.rel2id = {}
        self.features = {}
        self.memory_size = config.memory_size
        for id, rel in enumerate(id2rel):
            self.rel2id[rel] = id
        self.head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(self.hidden_size, config.contrasive_size))
        self.fc = nn.Linear(self.hidden_size, self.num_class, bias = False)

    def set_memorized_prototypes(self,protos):
        self.prototypes = protos

    def set_memorized_samples(self,relation,features):
        self.features[relation] = features

    def get_samples_dis(self, logits,relations):
        distance = [torch.max(-self.__distance__(logits, self.features[rel]), dim=1)[0] for rel in relations]
        distance = torch.stack(distance).t()
        return distance
    
    def get_mem_feature(self, logits):
        dis = self.__distance__(logits, self.prototypes)
        return dis
    
    def forward(self, sentences):
        logits = self.sentence_encoder(sentences) # (B, H)
        fc_rep = self.fc(logits)
        rep = self.head(logits)
        rep = F.normalize(rep, p=2, dim=1)
        return logits, rep , fc_rep

class Encoder(base_model):
#  load_config.attention_probs_dropout_prob = config.monto_drop_ratio
    # load_config.hidden_dropout_prob = config.monto_drop_ratio
    def __init__(self, config, attention_probs_dropout_prob=None, hidden_dropout_prob=None, drop_out=None): 
        super(Encoder, self).__init__()

        # load model
        self.encoder = BertModel.from_pretrained(config.bert_path).to(config.device)
        self.bert_config = BertConfig.from_pretrained(config.bert_path)
        self.device = config.device

        # for monto kalo
        if attention_probs_dropout_prob is not None:
            assert hidden_dropout_prob is not None and drop_out is not None
            self.bert_config.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.bert_config.hidden_dropout_prob = hidden_dropout_prob
            config.drop_out = drop_out

        # the dimension for the final outputs
        self.output_size = config.encoder_output_size

        self.drop = nn.Dropout(config.drop_out)

        # find which encoding is used
        if config.pattern in ['standard', 'entity_marker']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding.')

        if self.pattern == 'entity_marker':
            self.encoder.resize_token_embeddings(config.vocab_size + 4)
            self.linear_transform = nn.Linear(self.bert_config.hidden_size*2, self.output_size, bias=True)
        else:
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size, bias=True)
     
        self.layer_normalization = nn.LayerNorm([self.output_size])


    def get_output_size(self):
        return self.output_size

    def forward(self, inputs):
        '''
        :param inputs: of dimension [B, N]
        :return: a result of size [B, H*2] or [B, H], according to different strategy
        '''
        # generate representation under a certain encoding strategy
        if self.pattern == 'standard':
            # in the standard mode, the representation is generated according to
            #  the representation of[CLS] mark.
            output = self.encoder(inputs)[1]
        else:
            # in the entity_marker mode, the representation is generated from the representations of
            #  marks [E11] and [E21] of the head and tail entities.
            num = torch.arange(0, inputs.shape[1], 1).repeat(inputs.shape[0],1).to(self.device)
            e11 = torch.masked_select(num,(inputs == 30522))
            e21 = torch.masked_select(num,(inputs == 30524))

            # input the sample to BERT
            attention_mask = inputs != 0
            tokens_output = self.encoder(inputs, attention_mask=attention_mask)[0] # [B,N] --> [B,N,H]
            output = []

            # for each sample in the batch, acquire its representations for [E11] and [E21]
            for i in range(len(e11)):
                instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).to(self.device))
                instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).to(self.device))
                output.append(instance_output) # [B,N] --> [B,2,H]

            # for each sample in the batch, concatenate the representations of [E11] and [E21], and reshape
            output = torch.cat(output, dim=0)
            output = output.view(output.size()[0], -1) # [B,N] --> [B,H*2]
            
            # the output dimension is [B, H*2], B: batchsize, H: hiddensize
            output = self.drop(output)
            output = self.linear_transform(output)
            output = F.gelu(output)
            output = self.layer_normalization(output)
        return output
    

    



