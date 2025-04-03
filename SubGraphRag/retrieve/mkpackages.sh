#!/bin/bash

# 1. Create a directory to store packages
mkdir -p offline_packages

# 2. Download required packages
pip download -d offline_packages datasets
pip download -d offline_packages -r requirements/gte_large_en_v1-5.txt
pip download -d offline_packages xformers --index-url https://download.pytorch.org/whl/cu121

# 3. Create a tar archive
tar -czvf embedding_packages.tar.gz offline_packages

# 4. Now transfer the embedding_packages.tar.gz to your VM
# (e.g., using scp, rsync, or any other file transfer method)
