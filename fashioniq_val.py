import json
import os
import pickle
from argparse import ArgumentParser
from typing import List, Dict, Tuple
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTokenizer
import clip
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data_utils import collate_fn, PROJECT_ROOT, targetpad_transform
from src.datasets import FashionIQDataset, VAWValSubset, COCOValSubset
from src.utils import fashioniq_extract_image_features, device


@torch.no_grad()
def fiq_generate_val_predictions(clip_model, relative_val_dataset):
    """
    Generates features predictions for the validation set of Fashion IQ.
    """

    # Create data loader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=10,
                                     pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    predicted_features_captions_list = []
    target_names_list = []

    # Compute features
    for batch in tqdm(relative_val_loader):
        target_names = batch['target_name']
        mental_images = batch['mental_images'].to(device)
        mental_images_des = batch['mental_images_des']
        mental_images_des = clip.tokenize(mental_images_des, context_length=77, truncate=True).to(device)
        relative_captions = batch['mod_text']

        flattened_captions: list = np.array(relative_captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ')} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        input_captions_reversed = [
            f"{flattened_captions[i + 1].strip('.?, ')} and {flattened_captions[i].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]

        input_captions = clip.tokenize(input_captions, context_length=77).to(device)
        input_captions_reversed = clip.tokenize(input_captions_reversed, context_length=77).to(device)

        lam = 0.3
        with torch.no_grad():
            mental_images_feature = clip_model.encode_image(mental_images)
            mental_images_des_feature = clip_model.encode_text(mental_images_des)
            input_captions_feature = clip_model.encode_text(input_captions)
            input_captions_reversed_feature = clip_model.encode_text(input_captions_reversed)

        predicted_features = lam * F.normalize(mental_images_feature + mental_images_des_feature)

        predicted_features_captions = (1 - lam) * F.normalize((input_captions_feature+input_captions_reversed_feature)/2)

        predicted_features_list.append(predicted_features)
        predicted_features_captions_list.append(predicted_features_captions)
        target_names_list.extend(target_names)

    predicted_features = torch.vstack(predicted_features_list)
    predicted_features_captions = torch.vstack(predicted_features_captions_list)
    return predicted_features, predicted_features_captions, target_names_list


@torch.no_grad()
def fiq_compute_val_metrics(relative_val_dataset, clip_model, index_features, index_names):
    """
    Compute the retrieval metrics on the FashionIQ validation set given the dataset, pseudo tokens and the reference names
    """
    # Generate the predicted features
    predicted_features, predicted_features_captions, target_names = fiq_generate_val_predictions(clip_model, relative_val_dataset)

    # Move the features to the device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)
    predicted_features_captions = predicted_features_captions.to(device)

    # Normalize the features
    index_features = F.normalize(index_features.float())

    # Compute the distances
    distances = 1 - predicted_features @ index_features.T
    distances_captions = 1 - predicted_features_captions @ index_features.T
    sorted_indices = torch.argsort(distances + distances_captions, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Check if the target names are in the top 10 and top 50
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return {'fiq_recall_at10': recall_at10,
            'fiq_recall_at50': recall_at50}


@torch.no_grad()
def fiq_val_retrieval(dataset_path, dress_type, clip_model, preprocess):
    """
    Compute the retrieval metrics on the FashionIQ validation set given the pseudo tokens and the reference names
    """
    # Extract the index features
    classic_val_dataset = FashionIQDataset(dataset_path, 'val', [dress_type], 'classic', preprocess)
    index_features, index_names = fashioniq_extract_image_features(classic_val_dataset, clip_model)

    # Define the relative dataset
    relative_val_dataset = FashionIQDataset(dataset_path, 'val', [dress_type], 'relative', preprocess)

    return fiq_compute_val_metrics(relative_val_dataset, clip_model, index_features, index_names)

@torch.no_grad()
def get_recall(indices, targets):
    if len(targets.size()) == 1:
        # One hot label branch
        targets = targets.view(-1, 1).expand_as(indices)
        hits = (targets == indices).nonzero()
        if len(hits) == 0: return 0
        n_hits = (targets == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / targets.size(0)
        return recall
    else:
        # Multi hot label branch
        recall = []
        for preds, gt in zip(indices, targets):
            max_val = torch.max(torch.cat([preds, gt])).int().item()
            preds_binary = torch.zeros((max_val + 1,), device=preds.device, dtype=torch.float32).scatter_(0, preds, 1)
            gt_binary = torch.zeros((max_val + 1,), device=gt.device, dtype=torch.float32).scatter_(0, gt.long(), 1)
            success = (preds_binary * gt_binary).sum() > 0
            recall.append(int(success))
        return torch.Tensor(recall).float().mean()



def main():
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, help="Experiment to evaluate")
    parser.add_argument("--dataset", default="fashioniq", type=str, choices=['fashioniq'], help="Dataset to use")
    parser.add_argument("--dataset-path", default="./data_CIR/", type=str, help="Path to the dataset")
    parser.add_argument("--eval-type", default="clip_vitb", type=str, choices=['clip_vitb', 'clip_vitl', 'openclip_vitb', 'openclip_vitl'])

    args = parser.parse_args()

    if args.eval_type == 'clip_vitb':
        clip_model_name = 'ViT-B/32'
        clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
        clip_model = clip_model.float().eval().requires_grad_(False)
    elif args.eval_type == 'clip_vitl':
        clip_model_name = 'ViT-L/14'
        clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
        clip_model = clip_model.float().eval().requires_grad_(False)
    if args.eval_type == 'openclip_vitb':
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        clip_model = clip_model.eval().requires_grad_(False).to(device)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        clip_model.tokenizer = tokenizer
        clip_model = clip_model.float().eval().requires_grad_(False)
    elif args.eval_type == 'openclip_vitl':
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
        clip_model = clip_model.eval().requires_grad_(False).to(device)
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        clip_model.tokenizer = tokenizer
        clip_model = clip_model.float().eval().requires_grad_(False)

    if 'openclip' in args.eval_type:
        preprocess = targetpad_transform(1.25, clip_model.visual.image_size[0])
    else:
        preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)

    if args.dataset.lower() == 'fashioniq':
        recalls_at10 = []
        recalls_at50 = []
        for dress_type in ['shirt', 'dress', 'toptee']:
            fiq_metrics = fiq_val_retrieval(args.dataset_path, dress_type, clip_model, preprocess)

            recalls_at10.append(fiq_metrics['fiq_recall_at10'])
            recalls_at50.append(fiq_metrics['fiq_recall_at50'])

            for k, v in fiq_metrics.items():
                print(f"{dress_type}_{k} = {v:.2f}")
            print("\n")

        print(f"average_fiq_recall_at10 = {np.mean(recalls_at10):.2f}")
        print(f"average_fiq_recall_at50 = {np.mean(recalls_at50):.2f}")


if __name__ == '__main__':
    main()
