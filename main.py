import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
from copy import deepcopy
import os
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.metrics import f1_score, classification_report
from config import get_config
from model import Encoder, Classifier 
from data_loader import get_data_loader
from sampler import data_sampler
from utils import set_seed, select_data, get_proto, SupConLoss

def evaluate(config,id2num, test_data, seen_relations, mode="cur", pid2name=None, model=None):
    model.eval()
    n = len(test_data)
    data_loader = get_data_loader(config,id2num, test_data, batch_size=128)
    gold = []
    pred = []
    correct = 0
    f = []
    
    with torch.no_grad():
        for _, (_, labels, sentences,_,_,_) in enumerate(data_loader):
            sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
            logits, _ ,_= model(sentences)
            f += logits.tolist()
            dis_logits = -model.get_mem_feature(logits)
            distance  = model.get_samples_dis(logits,seen_relations)*0.1*episode
            dis_logits = distance + dis_logits
            predicts = dis_logits.max(dim=-1)[1]  
            labels = labels.to(config.device)
            correct += (predicts == labels).sum().item()
            predicts = predicts.tolist()
            labels = labels.tolist()
            gold.extend(labels)
            pred.extend(predicts)
    micro_f1 = f1_score(gold, pred, average='micro')
    macro_f1 = f1_score(gold, pred, average='macro')

    if len(pid2name) != 0:
        seen_relations = [x+pid2name[x][0] for x in seen_relations]
    if mode == "total":
        print('\n' + classification_report(gold, pred, labels=range(len(seen_relations)), target_names=seen_relations, zero_division=0))
        print("Micro F1 {}".format(micro_f1))
        print("Macro F1 {}".format(macro_f1))
    return correct / n

def train_model(config,model,train_set,training_data,id2num,id2label,epochs,mode = 'initial',model_clone = None):
    optimizer = optim.Adam([{'params': model.sentence_encoder.parameters(), 'lr': 0.00001},
                            {'params': model.head.parameters(), 'lr': 0.001,'weight_decay':config.weight_decay},
                            {'params': model.fc.parameters(), 'lr': 0.001,'weight_decay':config.weight_decay},])
    for _ in range(epochs):
        n_batch = (len(train_set) + config.batch_size_per_step - 1) // config.batch_size_per_step
        random.shuffle(train_set)
        for step in tqdm(range(n_batch)):
            start = step * config.batch_size_per_step
            end = (step + 1) * config.batch_size_per_step
            batch_data = train_set[start:end]
            batch_data_2 = []
            for item in batch_data:
                error = 0
                label = id2label[item['relation']]
                no_self = training_data[label]
                try:
                    no_self.remove(item)
                except ValueError:
                    error = ~error
                item2 = random.choice(no_self)
                batch_data_2.append(item2)
                if (~error):
                    no_self.append(item)
            if episode == 0 or mode == 'initial':
                model = initial_loss(config,id2num, batch_data, batch_data_2, model, optimizer)
            else:
                model = loss_replay(config,id2num, batch_data, batch_data_2, model,model_clone,optimizer)
    torch.cuda.empty_cache()
    return model



def loss_replay(config,id2num,batch_data, batch_data_2, model,clone_model,optimizer):
    label_criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(device = config.device)
    data_loader = get_data_loader(config,id2num, batch_data, batch_data_2, shuffle=False)
    for  _,labels, sentences,sentences2,_,_,_ in data_loader:
        labels = labels.to(config.device)
        loss_mask = labels < len(id2num)-config.rel_per_task
        sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
        sentences2 = torch.stack([x.to(config.device) for x in sentences2], dim=0)
        _, rep ,fc = model(sentences)
        _, rep2 ,_ = model(sentences2)
        clone_model.eval()
        with torch.no_grad():
            _,c_rep,_ = clone_model(sentences)
            _,c_rep2,_ = clone_model(sentences2)

        Kloss = torch.sum(((rep - c_rep) - (rep2 - c_rep2))**2,dim=1)
        Kloss = 0 if torch.sum(loss_mask) == 0 else  torch.sum(Kloss * loss_mask)/torch.sum(loss_mask)*(episode**config.alpha)
        cross = label_criterion(fc, labels) +  model.fc.weight.square().mean() 
        rep_output = torch.cat([rep, rep2], dim=1)
        rep_feature = torch.reshape(torch.unsqueeze(rep_output, 1).to(config.device), (-1, 2, config.contrasive_size))
        con_loss = torch.sum(torch.mul(con_criterion(rep_feature, labels),~loss_mask))/(config.batch_size_per_step-torch.sum(loss_mask))

        grads = {}
        def save_grad(name):
            def hook(grad):
                grads[name] = grad
            return hook
        def BA(rep,loss, Kloss,cross):
            rep.register_hook(save_grad('z'))
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            theta1 = grads['z'].contiguous().view(-1)
            optimizer.zero_grad()
            Kloss.backward(retain_graph = True)
            theta2 = grads['z'].contiguous().view(-1)
            part1 = torch.matmul((theta2 - theta1), theta2.reshape(-1,1))
            part2 = torch.norm(theta1 - theta2, p = 2)
            part2.pow(2)
            alpha = torch.div(part1, part2)
            min = torch.ones_like(alpha)
            alpha = torch.where(alpha > 1, min, alpha)
            min = torch.zeros_like(alpha)
            alpha = torch.where(alpha < 0, min, alpha)
            alpha1 = alpha
            alpha2 = (1 - alpha)
            optimizer.zero_grad()
            MTLoss = loss + alpha2/alpha1 * Kloss
            (MTLoss+cross).backward()
        if Kloss != 0:
            BA(rep,con_loss,Kloss,cross)
        else:
            optimizer.zero_grad()
            (con_loss+cross+Kloss).backward()
        optimizer.step()
        torch.cuda.empty_cache()
    return model


def initial_loss(config,id2num,batch_data, batch_data_2,  model,optimizer):
    label_criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(device = config.device)
    data_loader = get_data_loader(config,id2num, batch_data, batch_data_2, shuffle=False)
    model.train()
    for  _,labels, sentences,sentences2,_,_,_ in data_loader:
        labels = labels.to(config.device)
        sentences = torch.stack([x.to(config.device) for x in sentences], dim=0)
        sentences2 = torch.stack([x.to(config.device) for x in sentences2], dim=0)
      
        _, rep ,fc = model(sentences)
        _, rep2 ,_ = model(sentences2)

        rep_output = torch.cat([rep, rep2], dim=1)
        rep_feature = torch.reshape(torch.unsqueeze(rep_output, 1).to(config.device), (-1, 2, config.contrasive_size))
        cross =label_criterion(fc, labels)+ model.fc.weight.square().mean() 
        con_loss = con_criterion(rep_feature, labels).mean()
        loss = con_loss+cross
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    return model


        
if __name__ == '__main__':
    config = get_config()
    tokenizer = BertTokenizer.from_pretrained(config.bert_path, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    test_cur_record = []
    test_total_record = []
    pid2name = json.load(open('data/pid2name.json', 'r')) if  config.task_name.lower() == 'fewrel' else {}
    for i in range(config.total_round):
        test_cur = []
        test_total = []
        set_seed(config.seed + i * 100)
        # sampler setup
        sampler = data_sampler(config=config, seed=config.seed + i * 100, tokenizer=tokenizer)
        
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
        encoder = Encoder(config=config).to(config.device)
        model = Classifier(encoder, num_class = len(sampler.id2rel), id2rel = sampler.id2rel,  config = config).to(config.device)

        memorized_samples = {}
        id2num = {}
        num = 0
        for episode, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
            print(current_relations)
            feat_mem = []
            proto_mem = []

            train_data_for_initial = []
            current_dataset = {}
            for relation in current_relations:
                train_data_for_initial += training_data[relation]
                current_dataset[relation] = training_data[relation]              

            model_clone = deepcopy(model)
            for relation in current_relations:
                id2num[training_data[relation][0]['relation']] = num
                num +=1
            print('Initial Learning')
            model = train_model(config, model, train_data_for_initial,current_dataset,id2num,id2rel,1,mode='initial')
            
            print('Memory Replaying')
            train_data_for_replay = {}
            replaydata = []
            
            for relation in seen_relations:
                if relation in current_relations:
                    train_data_for_replay[relation] = training_data[relation]
                    replaydata += training_data[relation]
                else:
                    train_data_for_replay[relation] = memorized_samples[relation]
                    replaydata += train_data_for_replay[relation]

            model = train_model(config, model, replaydata,train_data_for_replay, id2num,id2rel, 4,mode='replay',model_clone = model_clone)

            print('Select Samples')
            for relation in tqdm(current_relations):
                training_data[relation],memorized_samples[relation],_, _,_= select_data(config, model.sentence_encoder, training_data[relation],id2num)

        
            protos = []
            for relation in seen_relations:
                features,weight = get_proto(config, model.sentence_encoder, memorized_samples[relation],id2num)
                prototype = torch.sum(torch.mul(features,weight), dim=0,keepdim=True)
                protos.append(prototype)
                model.set_memorized_samples(relation,features)
            protos = torch.cat(protos, dim=0).detach()
            model.set_memorized_prototypes(protos) 

            
            print('[Evaluation]')
            test_data_1 = []
            for relation in current_relations:
                test_data_1 += test_data[relation]

            test_data_2 = []
            for relation in seen_relations:
                test_data_2 += historic_test_data[relation]
            cur_acc = evaluate(config,id2num, test_data_1,seen_relations, mode="cur", pid2name=pid2name, model= model)
            total_acc =  evaluate(config,id2num, test_data_2 ,seen_relations, mode="total", pid2name=pid2name, model= model)
   
            print(f'Restart Num {i+1}')
            print(f'task--{episode + 1}:')
            test_cur.append(cur_acc)
            test_total.append(total_acc)
            print(f'history test acc:{test_total}')
            print(f'current test acc:{test_cur}')
        
        test_cur_record.append(test_cur)
        test_total_record.append(test_total)

        print(f'Round history test acc:{test_total}')
        print(f'Round current test acc:{test_cur}')
        torch.cuda.empty_cache()


    test_cur_record = torch.tensor(test_cur_record)
    test_total_record = torch.tensor(test_total_record)

    test_cur_record = torch.mean(test_cur_record, dim=0).tolist()
    test_total_record = torch.mean(test_total_record, dim=0).tolist()

    print(f'Avg current test acc: {test_cur_record}')
    print(f'Avg total test acc: {test_total_record}')