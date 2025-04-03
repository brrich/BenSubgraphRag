import os
import torch

from datasets import load_dataset, Dataset
import json
from tqdm import tqdm

from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset

def save_dataset_to_disk(dataset, save_path):
    """Save a dataset to disk in JSON format"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(dataset.to_dict(), f)

def load_dataset_from_disk(load_path):
    """Load a dataset from disk"""
    with open(load_path, 'r') as f:
        data_dict = json.load(f)
    return Dataset.from_dict(data_dict)

def get_dataset(input_file, split, local_dir='data_files/datasets'):
    """Get dataset either from disk or download it"""
    # Create filename based on input_file and split
    filename = f"{input_file.replace('/', '_')}_{split}.json"
    local_path = os.path.join(local_dir, filename)
    
    # Try to load from disk first
    if os.path.exists(local_path):
        print(f"Loading {split} set from {local_path}")
        return load_dataset_from_disk(local_path)
    
    # If not found locally, download and save
    print(f"Downloading {split} set from {input_file}")
    dataset = load_dataset(input_file, split=split)
    save_dataset_to_disk(dataset, local_path)
    return dataset

def get_emb(subset, text_encoder, save_file):
    emb_dict = dict()
    for i in tqdm(range(len(subset))):
        id, q_text, text_entity_list, relation_list = subset[i]
        
        q_emb, entity_embs, relation_embs = text_encoder(
            q_text, text_entity_list, relation_list)
        emb_dict_i = {
            'q_emb': q_emb,
            'entity_embs': entity_embs,
            'relation_embs': relation_embs
        }
        emb_dict[id] = emb_dict_i
    
    torch.save(emb_dict, save_file)

def main(args):
    # Modify the config file for advanced settings and extensions.
    config_file = f'configs/emb/gte-large-en-v1.5/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    torch.set_num_threads(config['env']['num_threads'])

    if args.dataset == 'cwq':
        input_file = os.path.join('rmanluo', 'RoG-cwq')
    else:
        input_file = os.path.join('ml1996', 'webqsp')

    # Load datasets with local caching
    train_set = get_dataset(input_file, 'train')
    val_set = get_dataset(input_file, 'validation')
    test_set = get_dataset(input_file, 'test')

    entity_identifiers = []
    with open(config['entity_identifier_file'], 'r') as f:
        for line in f:
            entity_identifiers.append(line.strip())
    entity_identifiers = set(entity_identifiers)
    
    save_dir = f'data_files/{args.dataset}/processed'
    os.makedirs(save_dir, exist_ok=True)

    train_set = EmbInferDataset(
        train_set,
        entity_identifiers,
        os.path.join(save_dir, 'train.pkl'))

    val_set = EmbInferDataset(
        val_set,
        entity_identifiers,
        os.path.join(save_dir, 'val.pkl'))

    test_set = EmbInferDataset(
        test_set,
        entity_identifiers,
        os.path.join(save_dir, 'test.pkl'),
        skip_no_topic=False,
        skip_no_ans=False)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    text_encoder_name = config['text_encoder']['name']
    if text_encoder_name == 'gte-large-en-v1.5':
        from src.model.text_encoders import GTELargeEN
        text_encoder = GTELargeEN(device)
    else:
        raise NotImplementedError(text_encoder_name)
    
    emb_save_dir = f'data_files/{args.dataset}/emb/{text_encoder_name}'
    os.makedirs(emb_save_dir, exist_ok=True)
    
    get_emb(train_set, text_encoder, os.path.join(emb_save_dir, 'train.pth'))
    get_emb(val_set, text_encoder, os.path.join(emb_save_dir, 'val.pth'))
    get_emb(test_set, text_encoder, os.path.join(emb_save_dir, 'test.pth'))

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    args = parser.parse_args()
    
    main(args)
