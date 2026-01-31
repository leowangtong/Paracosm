import pickle
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import clip
from src.data_utils import collate_fn

if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32


@torch.no_grad()
def circo_extract_image_features(dataset: Dataset, clip_model: CLIP, batch_size: Optional[int] = 32,
                           num_workers: Optional[int] = 10) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    # Create data loader
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    index_features = []
    index_names = []

    try:
        print(f"extracting image features {dataset.__class__.__name__} - {dataset.split}")
    except Exception as e:
        pass

    # Extract features
    for batch in tqdm(loader):
        images = batch.get('image').to(device)
        synthetic_img = batch.get('synthetic_img').to(device)
        names = batch.get('image_name')

        with torch.no_grad():
            img_feature = clip_model.encode_image(images)
            synthetic_img_feature = clip_model.encode_image(synthetic_img)

            batch_features = F.normalize(img_feature) + F.normalize(synthetic_img_feature)

            index_features.append(batch_features.cpu())
            index_names.extend(names)

    index_features = torch.vstack(index_features)

    return index_features, index_names




@torch.no_grad()
def cirr_extract_image_features(dataset: Dataset, clip_model: CLIP, batch_size: Optional[int] = 32,
                           num_workers: Optional[int] = 10) -> Tuple[torch.Tensor, List[str]]:
    # def extract_image_features(dataset: Dataset, clip_model: CLIP, batch_size: Optional[int] = 32,
    #                                num_workers: Optional[int] = 10) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    # Create data loader
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    index_features = []
    index_names = []
    try:
        print(f"extracting image features {dataset.__class__.__name__} - {dataset.split}")
    except Exception as e:
        pass

    # Extract features
    for batch in tqdm(loader):
        images = batch.get('image').to(device)
        names = batch.get('image_name')
        synthetic_img = batch.get('synthetic_img').to(device)

        with torch.no_grad():
            img_feature = clip_model.encode_image(images)
            synthetic_img_feature = clip_model.encode_image(synthetic_img)

            batch_features = F.normalize(img_feature) + F.normalize(synthetic_img_feature)

        index_features.append(batch_features.cpu())
        index_names.extend(names)

    index_features = torch.vstack(index_features)

    return index_features, index_names


@torch.no_grad()
def fashioniq_extract_image_features(dataset: Dataset, clip_model: CLIP, batch_size: Optional[int] = 32,
                           num_workers: Optional[int] = 10) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    # Create data loader
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    index_features = []
    index_names = []
    try:
        print(f"extracting image features {dataset.__class__.__name__} - {dataset.split}")
    except Exception as e:
        pass

    # Extract features
    for batch in tqdm(loader):
        images = batch.get('image').to(device)
        synthetic_img = batch.get('synthetic_img').to(device)
        names = batch.get('image_name')

        with torch.no_grad():
            img_feature = clip_model.encode_image(images)
            synthetic_img_feature = clip_model.encode_image(synthetic_img)

            batch_features = F.normalize(img_feature) + F.normalize(synthetic_img_feature)

            index_features.append(batch_features.cpu())
            index_names.extend(names)

    index_features = torch.vstack(index_features)
    return index_features, index_names

