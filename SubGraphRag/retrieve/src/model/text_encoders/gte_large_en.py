import torch
import torch.nn.functional as F
import os
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

class GTELargeEN:
    def __init__(self,
                 device,
                 normalize=True,
                 cache_dir=None):
        self.device = device
        self.normalize = normalize
        
        # Set up model paths
        model_name = 'Alibaba-NLP/gte-large-en-v1.5'
        if cache_dir is None:
            # Get the directory of the current file
            current_dir = Path(__file__).resolve().parent
            # Navigate up to the retrieve root directory
            retrieve_root = current_dir.parent.parent.parent
            cache_dir = str(retrieve_root / 'model_cache')
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Using cache directory: {cache_dir}")
        
        try:
            # First try to load from local cache
            print(f"Attempting to load model from local cache: {cache_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
                cache_dir=cache_dir
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                local_files_only=True,
                cache_dir=cache_dir,
                trust_remote_code=True,
                unpad_inputs=True
            ).to(device)
            
        except Exception as e:
            print(f"Could not load from cache: {str(e)}")
            print("Attempting to download model from HuggingFace...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
                self.model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    unpad_inputs=True
                ).to(device)
            except Exception as download_error:
                raise RuntimeError(
                    f"Failed to load model both locally and from HuggingFace. "
                    f"If offline, ensure model is downloaded to {cache_dir}. "
                    f"Error: {str(download_error)}"
                )
        
        self.model.eval()  # Set model to evaluation mode

    @torch.no_grad()
    def embed(self, text_list):
        if len(text_list) == 0:
            return torch.zeros(0, 1024, device=self.device)
        
        batch_dict = self.tokenizer(
            text_list, max_length=8192, padding=True,
            truncation=True, return_tensors='pt')
        
        # Move to device efficiently
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        
        outputs = self.model(**batch_dict).last_hidden_state
        emb = outputs[:, 0]
        
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        
        return emb.cpu()

    def __call__(self, q_text, text_entity_list, relation_list):
        q_emb = self.embed([q_text])
        entity_embs = self.embed(text_entity_list)
        relation_embs = self.embed(relation_list)
        
        return q_emb, entity_embs, relation_embs

if __name__ == "__main__":
    print("Initializing GTELargeEN to download and cache the model...")
    
    # Try to use CUDA if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize the model - this will force a download if not in cache
    model = GTELargeEN(device=device)
    
    # Test the model with a simple example
    print("\nTesting model with a simple example...")
    test_query = "What is the capital of France?"
    test_entities = ["Paris is the capital of France.", "London is the capital of the UK."]
    test_relations = ["is capital of"]
    
    q_emb, entity_embs, relation_embs = model(test_query, test_entities, test_relations)
    
    print(f"\nTest successful!")
    print(f"Query embedding shape: {q_emb.shape}")
    print(f"Entity embeddings shape: {entity_embs.shape}")
    print(f"Relation embeddings shape: {relation_embs.shape}")
    print("\nModel is now cached and ready for offline use.")
