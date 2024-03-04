import argparse
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='./logs', type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--total_round', default=5, type=int)
    parser.add_argument('--seed', default=2021, type=int)
    
    parser.add_argument('--num_of_train', default=420, type=int)
    parser.add_argument('--num_of_val', default=140, type=int)
    parser.add_argument('--num_of_test', default=140, type=int)
    parser.add_argument("--max_grad_norm", default=10, type=float)
    parser.add_argument('--memory_size', default=10, type=int)
    
    parser.add_argument('--task_name', default='FewRel', type=str)
    
    config = parser.parse_args()
    if config.task_name == 'FewRel':
      parser.add_argument('--num_of_relation', default=80, type=int)
      parser.add_argument('--rel_per_task', default=8, type=int)
      parser.add_argument('--data_file', default='./data/data_with_marker.json', type=str)
      parser.add_argument('--cache_file', default='./data/fewrel_data.pt', type=str)
      parser.add_argument('--relation_file', default='./data/id2rel.json', type=str)
      parser.add_argument('--drop_out', default=0.2, type=float)
      parser.add_argument('--weight_decay', default=0.001, type=float)
      parser.add_argument('--alpha', default=1.0, type=float)
    else:
      parser.add_argument('--num_of_relation', default=40, type=int)
      parser.add_argument('--rel_per_task', default=4, type=int)
      parser.add_argument('--data_file', default='./data/data_with_marker_tacred.json', type=str)
      parser.add_argument('--cache_file', default='./data/TACRED_data.pt', type=str)
      parser.add_argument('--relation_file', default='./data/id2rel_tacred.json', type=str)
      parser.add_argument('--drop_out', default=0, type=float)
      parser.add_argument('--weight_decay', default=0.01, type=float)
      parser.add_argument('--alpha', default=0.8, type=float)
      
    parser.add_argument('--bert_path', default='./bert-base-uncased', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--vocab_size', default=30522, type=int)
    parser.add_argument('--batch_size_per_step', default=32, type=int)
    parser.add_argument('--contrasive_size', default=64, type=str)
    parser.add_argument('--pattern', default='entity_marker', type=str)
    parser.add_argument('--key_size', default=256, type=int)
    parser.add_argument('--head_size', default=768, type=int)
    parser.add_argument('--encoder_output_size', default=768, type=int)
    parser.add_argument("--optim", default='adam', type=str)


    config = parser.parse_args()
    return config

