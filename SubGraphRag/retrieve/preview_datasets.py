import os
import torch
import random
import pickle
from datasets import load_dataset, Dataset
import json
from argparse import ArgumentParser

from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset

def save_dataset_to_disk(dataset, save_path):
    """Save a dataset to disk in JSON format"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(dataset.to_dict(), f)
    print(f"Saved dataset to {save_path}")

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
        print(f"✅ Found existing {split} set at {local_path}")
        return load_dataset_from_disk(local_path)
    
    # If not found locally, download and save
    print(f"❌ Dataset not found locally. Downloading {split} set from {input_file}")
    print(f"This may take some time, please be patient...")
    try:
        # Download the dataset
        dataset = load_dataset(input_file, split=split, verification_mode='no_checks')
        print(f"✅ Download complete! Saving raw JSON file...")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Save as raw JSON
        save_dataset_to_disk(dataset, local_path)
        print(f"✅ Raw JSON saved successfully to {local_path}")
        return dataset
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        raise

def download_raw_datasets(dataset_name):
    """Download only the raw datasets without processing them"""
    print(f"\n🔽 Downloading raw {dataset_name} dataset...")
    
    if dataset_name == 'cwq':
        input_file = os.path.join('rmanluo', 'RoG-cwq')
    else:
        input_file = os.path.join('ml1996', 'webqsp')
    
    print(f"\n📥 Downloading/checking {dataset_name} dataset from {input_file}...")

    # Load datasets with local caching
    print("\nDownloading/checking train set...")
    train_set = get_dataset(input_file, 'train')
    print("\nDownloading/checking validation set...")
    val_set = get_dataset(input_file, 'validation')
    print("\nDownloading/checking test set...")
    test_set = get_dataset(input_file, 'test')
    
    print(f"\n✅ Raw datasets for {dataset_name} have been downloaded or verified")
    return train_set, val_set, test_set

def check_embedded_datasets(dataset_name, text_encoder_name='gte-large-en-v1.5'):
    """Check if the embedded datasets exist"""
    emb_dir = f'data_files/{dataset_name}/emb/{text_encoder_name}'
    
    train_emb_path = os.path.join(emb_dir, 'train.pth')
    val_emb_path = os.path.join(emb_dir, 'val.pth')
    test_emb_path = os.path.join(emb_dir, 'test.pth')
    
    train_exists = os.path.exists(train_emb_path)
    val_exists = os.path.exists(val_emb_path)
    test_exists = os.path.exists(test_emb_path)
    
    print(f"\nEmbedded datasets status for {dataset_name} ({text_encoder_name}):")
    print(f"  - Train embeddings: {'✅ Found' if train_exists else '❌ Not found'} at {train_emb_path}")
    print(f"  - Validation embeddings: {'✅ Found' if val_exists else '❌ Not found'} at {val_emb_path}")
    print(f"  - Test embeddings: {'✅ Found' if test_exists else '❌ Not found'} at {test_emb_path}")
    
    return train_exists, val_exists, test_exists

def check_processed_datasets(dataset_name):
    """Check if processed datasets exist"""
    processed_dir = f'data_files/{dataset_name}/processed'
    
    train_path = os.path.join(processed_dir, 'train.pkl')
    val_path = os.path.join(processed_dir, 'val.pkl')
    test_path = os.path.join(processed_dir, 'test.pkl')
    
    train_exists = os.path.exists(train_path)
    val_exists = os.path.exists(val_path)
    test_exists = os.path.exists(test_path)
    
    print(f"\nProcessed datasets status for {dataset_name}:")
    print(f"  - Train processed data: {'✅ Found' if train_exists else '❌ Not found'} at {train_path}")
    print(f"  - Validation processed data: {'✅ Found' if val_exists else '❌ Not found'} at {val_path}")
    print(f"  - Test processed data: {'✅ Found' if test_exists else '❌ Not found'} at {test_path}")
    
    return train_exists, val_exists, test_exists

def load_processed_dataset(dataset_name, split):
    """Load a processed dataset from disk"""
    processed_path = f'data_files/{dataset_name}/processed/{split}.pkl'
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed dataset not found at {processed_path}")
    
    print(f"Loading processed {dataset_name} {split} dataset from {processed_path}")
    with open(processed_path, 'rb') as f:
        return pickle.load(f)

def preview_webqsp(num_examples=5, download_only=False):
    """Preview random examples from WebQSP dataset"""
    if download_only:
        download_raw_datasets('webqsp')
        return None
    return preview_dataset('webqsp', num_examples)

def preview_cwq(num_examples=5, download_only=False):
    """Preview random examples from CWQ dataset"""
    if download_only:
        download_raw_datasets('cwq')
        return None
    return preview_dataset('cwq', num_examples)

def preview_dataset(dataset_name, num_examples=5):
    """Preview random examples from a dataset"""
    print(f"\n📊 Previewing {dataset_name} dataset...")
    
    # Check if processed datasets exist
    train_exists, val_exists, test_exists = check_processed_datasets(dataset_name)
    
    # Also check if embedded datasets exist (informational only)
    emb_train_exists, emb_val_exists, emb_test_exists = check_embedded_datasets(dataset_name)
    
    if not (train_exists and val_exists and test_exists):
        print(f"\n🔄 Some processed datasets for {dataset_name} are missing. Downloading and processing...")
        download_and_process(dataset_name)
    else:
        print(f"\n✅ All processed datasets for {dataset_name} are available.")
    
    # Load processed dataset (using test split by default for preview)
    processed_data = load_processed_dataset(dataset_name, 'test')
    
    # Select random examples
    total_examples = len(processed_data)
    if num_examples > total_examples:
        num_examples = total_examples
        print(f"⚠️ Requested more examples than available. Showing all {total_examples} examples.")
    
    print(f"\n🔍 Showing {num_examples} random examples from {dataset_name} test set ({total_examples} total examples):")
    
    random_indices = random.sample(range(total_examples), num_examples)
    
    # Display examples
    examples = []
    for i, idx in enumerate(random_indices):
        sample = processed_data[idx]
        example = {
            'id': sample['id'],
            'question': sample['question'],
            'question_entities': sample['q_entity'],
            'answer_entities': sample['a_entity']
        }
        examples.append(example)
        
        print(f"\n--- Example {i+1}/{num_examples} (index {idx}) ---")
        print(f"Question: {sample['question']}")
        print(f"Question entities: {', '.join(sample['q_entity'])}")
        print(f"Answer: {', '.join(sample['a_entity'])}")
        print(f"Total entities in graph: {len(sample['text_entity_list']) + len(sample['non_text_entity_list'])}")
        print(f"Total relations in graph: {len(sample['relation_list'])}")
        print(f"Total triples in graph: {len(sample['h_id_list'])}")
    
    return examples

def download_and_process(dataset_name):
    """Download and process a dataset (similar to emb.py)"""
    print(f"\n🔽 Preparing to download and process {dataset_name} dataset...")
    
    # Load configuration
    config_file = f'configs/emb/gte-large-en-v1.5/{dataset_name}.yaml'
    print(f"Loading configuration from {config_file}")
    config = load_yaml(config_file)
    
    torch.set_num_threads(config['env']['num_threads'])
    print(f"Set torch threads to {config['env']['num_threads']}")

    if dataset_name == 'cwq':
        input_file = os.path.join('rmanluo', 'RoG-cwq')
    else:
        input_file = os.path.join('ml1996', 'webqsp')
    
    print(f"\n📥 Downloading/loading {dataset_name} dataset from {input_file}...")

    # Load datasets with local caching
    print("\nDownloading/loading train set...")
    train_set = get_dataset(input_file, 'train')
    print("\nDownloading/loading validation set...")
    val_set = get_dataset(input_file, 'validation')
    print("\nDownloading/loading test set...")
    test_set = get_dataset(input_file, 'test')

    print(f"\nLoading entity identifiers from {config['entity_identifier_file']}")
    entity_identifiers = []
    with open(config['entity_identifier_file'], 'r') as f:
        for line in f:
            entity_identifiers.append(line.strip())
    entity_identifiers = set(entity_identifiers)
    print(f"Loaded {len(entity_identifiers)} entity identifiers")
    
    save_dir = f'data_files/{dataset_name}/processed'
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n💾 Processing datasets and saving to {save_dir}...")

    print("\nProcessing train set...")
    train_set = EmbInferDataset(
        train_set,
        entity_identifiers,
        os.path.join(save_dir, 'train.pkl'))

    print("\nProcessing validation set...")
    val_set = EmbInferDataset(
        val_set,
        entity_identifiers,
        os.path.join(save_dir, 'val.pkl'))

    print("\nProcessing test set...")
    test_set = EmbInferDataset(
        test_set,
        entity_identifiers,
        os.path.join(save_dir, 'test.pkl'),
        skip_no_topic=False,
        skip_no_ans=False)
    
    print(f"\n✅ Datasets for {dataset_name} have been processed and saved to {save_dir}")
    return train_set, val_set, test_set

def main():
    parser = ArgumentParser('Preview Datasets for SubGraphRag')
    parser.add_argument('-ds', '--dataset', type=str, required=False, 
                        choices=['webqsp', 'cwq', 'both'], default='both',
                        help='Dataset name (default: both)')
    parser.add_argument('-n', '--num_examples', type=int, default=5,
                        help='Number of examples to preview (default: 5)')
    parser.add_argument('-d', '--download_only', action='store_true',
                        help='Only download raw datasets without processing or previewing')
    args = parser.parse_args()
    
    print("\n===== SubGraphRag Dataset Preview Tool =====")
    
    if args.download_only:
        print("Running in DOWNLOAD ONLY mode - will only download raw datasets without processing")
        print(f"Storage path for raw datasets: data_files/datasets/")
        print("===========================================\n")
        
        if args.dataset == 'webqsp' or args.dataset == 'both':
            print("\n=== WebQSP Raw Dataset Download ===")
            download_raw_datasets('webqsp')
        
        if args.dataset == 'cwq' or args.dataset == 'both':
            print("\n=== CWQ Raw Dataset Download ===")
            download_raw_datasets('cwq')
            
        print("\n✅ Raw dataset download complete")
        return
        
    # Normal preview mode
    print("This tool checks for datasets and displays random examples")
    print(f"Storage paths:")
    print(f"  - Raw downloaded datasets: data_files/datasets/")
    print(f"  - Processed datasets: data_files/[dataset_name]/processed/")
    print(f"  - Embedded datasets: data_files/[dataset_name]/emb/gte-large-en-v1.5/")
    print("===========================================\n")
    
    if args.dataset == 'webqsp' or args.dataset == 'both':
        print("\n=== WebQSP Dataset Preview ===")
        preview_webqsp(args.num_examples)
    
    if args.dataset == 'cwq' or args.dataset == 'both':
        print("\n=== CWQ Dataset Preview ===")
        preview_cwq(args.num_examples)

if __name__ == '__main__':
    main() 