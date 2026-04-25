import os
from torchvision.models import resnet101, ResNet101_Weights, resnet50
from pathlib import Path
from torch.nn import Module
from typing import List, Tuple
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoModel, AutoTokenizer
import torch
from get_dsets import load_all_in_tree

# Load model and tokenizer
#model_name = "mistralai/Mistral-7B-Instruct"
#model_name = "mistralai/Mistral-7B-v0.1"
# Tokenize input
dset = 'im'

if dset=='ng20':
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    #model_name = 'llamafactory/tiny-random-Llama-3'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    texts = newsgroups.data
    labels = newsgroups.target
    bs = 1
    n_batches = math.ceil(len(texts)/bs)
    os.makedirs(doc_embs_dir:='data/ng20-featvecs', exist_ok=True)
    for batch_idx in tqdm(range(n_batches)):
        t = texts[batch_idx*bs:(batch_idx+1)*bs]
        #out_fps = [f'{doc_embs_dir}/{batch_idx+i}.npy' for i in range(bs)]
        out_fp = f'{doc_embs_dir}/{batch_idx}.npy'
        #if batch_idx!=11166 and all(os.path.exists(fp) for fp in out_fps):
        if os.path.exists(out_fp):
            doc_ar = np.load(out_fp)
            if not np.isinf(doc_ar).any():
                continue

        print(f'recomputing for {batch_idx}')
        inputs = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=2048).to(model.device)

        # Get hidden states
        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling (ignoring padding tokens)
        embeddings = outputs.last_hidden_state.detach().cpu().float()  # Shape: (batch_size, seq_length, hidden_size)
        attention_mask = inputs["attention_mask"].cpu().unsqueeze(-1)  # Shape: (batch_size, seq_length, 1)
        doc_embedding = (embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        doc_ar = doc_embedding.numpy()
        np.save(out_fp, doc_ar)
elif dset=='im':
    imgs, img_paths = load_all_in_tree('data/imagenette2')

    model_ = resnet50(num_classes=365)

    # Load the state dict
    checkpoint = torch.load('resnet50_places365.pth.tar', map_location='cpu')
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model_.load_state_dict(state_dict)
    # Remove the final classification layer
    feature_extractor = nn.Sequential(*list(model_.children())[:-1])
    feature_extractor.eval()
    feature_extractor = feature_extractor.cuda()

    # Create base output directory
    base_output_dir = Path('data/im-featvecs')
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Group images by class
    class_indices = {}
    for idx, path in enumerate(img_paths):
        class_name = Path(path).parent.name
        if class_name not in class_indices:
            class_indices[class_name] = []
        class_indices[class_name].append(idx)

        # Create class directory if it doesn't exist
        class_dir = base_output_dir / class_name
        class_dir.mkdir(exist_ok=True)

    # Process each class
    batch_size = 32
    with torch.no_grad():
        imgs = torch.tensor(imgs).permute(0,3,1,2).float()/255
        print(imgs.max(), imgs.min(), imgs.mean())
        for class_name, indices in class_indices.items():
            print(f"Processing class: {class_name}")

            # Process in batches
            for i in tqdm(range(0, len(indices), batch_size)):
                batch_indices = indices[i:i + batch_size]
                batch_imgs = torch.stack([imgs[idx] for idx in batch_indices])

                # Extract features
                batch_features = feature_extractor(batch_imgs.cuda())
                batch_features = batch_features.squeeze(-1).squeeze(-1).cpu().numpy()

            # Save individual feature vectors
                for idx, feat_vec in zip(batch_indices, batch_features):
                    output_path = base_output_dir / class_name / f"{Path(img_paths[idx]).stem}.npy"
                    np.save(output_path, feat_vec)

            print(f"Processed batch {i//batch_size + 1}/{(len(indices)-1)//batch_size + 1}")
