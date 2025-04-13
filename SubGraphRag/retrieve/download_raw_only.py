import os
import json
from datasets import load_dataset, Dataset
from argparse import ArgumentParser

def save_dataset_to_json(dataset, save_path):
    """Save a dataset directly to disk as JSON"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving raw dataset to {save_path}...")
    with open(save_path, 'w') as f:
        json.dump(dataset.to_dict(), f)
    print(f"✅ Saved raw JSON to {save_path}")

def download_dataset(input_file, split, local_dir='data_files/datasets'):
    """Download dataset and save as raw JSON"""
    # Create filename based on input_file and split
    filename = f"{input_file.replace('/', '_')}_{split}.json"
    local_path = os.path.join(local_dir, filename)
    
    # Check if file already exists
    if os.path.exists(local_path):
        print(f"✅ Raw {split} set already exists at {local_path}")
        return True
    
    # Download and save if not found
    print(f"⏳ Downloading {split} set from {input_file}...")
    print(f"This may take some time, please be patient...")
    
    try:
        # Download the dataset
        dataset = load_dataset(input_file, split=split, verification_mode='no_checks')
        print(f"✅ Download complete!")
        
        # Save as raw JSON
        save_dataset_to_json(dataset, local_path)
        return True
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        return False

def download_cwq():
    """Download only CWQ raw dataset files"""
    print("\n=== Downloading CWQ Raw Dataset ===")
    input_file = os.path.join('rmanluo', 'RoG-cwq')
    
    # Ensure the directory exists
    os.makedirs('data_files/datasets', exist_ok=True)
    
    # Download each split
    train_ok = download_dataset(input_file, 'train')
    val_ok = download_dataset(input_file, 'validation')
    test_ok = download_dataset(input_file, 'test')
    
    if train_ok and val_ok and test_ok:
        print("\n✅ All CWQ raw dataset files downloaded/verified successfully")
    else:
        print("\n⚠️ Some CWQ dataset files could not be downloaded")

def download_webqsp():
    """Download only WebQSP raw dataset files"""
    print("\n=== Downloading WebQSP Raw Dataset ===")
    input_file = os.path.join('ml1996', 'webqsp')
    
    # Ensure the directory exists
    os.makedirs('data_files/datasets', exist_ok=True)
    
    # Download each split
    train_ok = download_dataset(input_file, 'train')
    val_ok = download_dataset(input_file, 'validation')
    test_ok = download_dataset(input_file, 'test')
    
    if train_ok and val_ok and test_ok:
        print("\n✅ All WebQSP raw dataset files downloaded/verified successfully")
    else:
        print("\n⚠️ Some WebQSP dataset files could not be downloaded")

def main():
    parser = ArgumentParser('Download Raw Datasets Only')
    parser.add_argument('-ds', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq', 'both'],
                        help='Dataset to download: webqsp, cwq, or both')
    
    args = parser.parse_args()
    
    print("\n===== Raw Dataset Downloader =====")
    print("This tool only downloads and saves raw dataset JSON files")
    print("No processing or embedding is performed")
    print(f"Files will be saved to: data_files/datasets/")
    print("=====================================\n")
    
    if args.dataset == 'cwq' or args.dataset == 'both':
        download_cwq()
    
    if args.dataset == 'webqsp' or args.dataset == 'both':
        download_webqsp()
    
    print("\n✅ Download process completed!")

if __name__ == '__main__':
    main() 