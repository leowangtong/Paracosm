import json
from argparse import ArgumentParser
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip

from src.data_utils import PROJECT_ROOT, targetpad_transform
from src.datasets import CIRRDataset, CIRCODataset
from src.utils import circo_extract_image_features, cirr_extract_image_features, device, collate_fn


@torch.no_grad()
def cirr_generate_test_submission_file(dataset_path, clip_model, preprocess, submission_name) -> None:
    """
    Generate the test submission file for the CIRR dataset given the pseudo tokens
    """

    # Compute the index features
    classic_test_dataset = CIRRDataset(dataset_path, 'test1', 'classic', preprocess)
    index_features, index_names = cirr_extract_image_features(classic_test_dataset, clip_model)

    relative_test_dataset = CIRRDataset(dataset_path, 'test1', 'relative', preprocess)

    # Get the predictions dicts
    pairid_to_retrieved_images, pairid_to_group_retrieved_images = cirr_generate_test_dicts(relative_test_dataset, clip_model, index_features, index_names)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_retrieved_images)
    group_submission.update(pairid_to_group_retrieved_images)

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'cirr'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{submission_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"subset_{submission_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def cirr_generate_test_dicts(relative_test_dataset, clip_model, index_features, index_names):
    """
    Generate the test submission dicts for the CIRR dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, predicted_features_modi, reference_names, pairs_id, group_members = cirr_generate_test_predictions(clip_model, relative_test_dataset)

    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = index_features.to(device)
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    distances_modi = 1 - predicted_features_modi @ index_features.T
    sorted_indices = torch.argsort(distances + distances_modi, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))

    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_retrieved_images = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                                  zip(pairs_id, sorted_index_names)}
    pairid_to_group_retrieved_images = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                        zip(pairs_id, sorted_group_names)}

    return pairid_to_retrieved_images, pairid_to_group_retrieved_images


def cirr_generate_test_predictions(clip_model, relative_test_dataset: CIRRDataset):
    """
    Generate the test prediction features for the CIRR dataset given the pseudo tokens
    """

    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=10,
                                      pin_memory=False)

    predicted_features_list = []
    predicted_features_modi_list = []
    pair_id_list = []
    group_members_list = []
    reference_names_list = []

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        group_members = batch['group_members']
        group_members = np.array(group_members).T.tolist()
        pairs_id = batch['pair_id']
        modify_text = batch['relative_caption']
        modify_text = clip.tokenize(modify_text, context_length=77, truncate=True).to(device)
        mental_image_des = batch['mental_image_des']
        mental_image_des = clip.tokenize(mental_image_des, context_length=77, truncate=True).to(device)
        mental_image = batch['mental_image'].to(device)

        with torch.no_grad():
            mental_image_features = clip_model.encode_image(mental_image)
            mental_image_des_features = clip_model.encode_text(mental_image_des)
            modify_text_features = clip_model.encode_text(modify_text)

        lam = 0.3
        predicted_features = mental_image_features + mental_image_des_features
        predicted_features = lam*(F.normalize(predicted_features))
        predicted_features_modi = (1 - lam) * F.normalize(modify_text_features)

        predicted_features_list.append(predicted_features)
        predicted_features_modi_list.append(predicted_features_modi)
        pair_id_list.extend(pairs_id)
        reference_names_list.extend(reference_names)
        group_members_list.extend(group_members)

    predicted_features = torch.vstack(predicted_features_list)
    predicted_features_modi = torch.vstack(predicted_features_modi_list)
    return predicted_features, predicted_features_modi, reference_names_list, pair_id_list, group_members_list


@torch.no_grad()
def circo_generate_test_submission_file(dataset_path, clip_model, preprocess, submission_name) -> None:
    """
    Generate the test submission file for the CIRCO dataset given the pseudo tokens
    """

    # Compute the index features
    classic_test_dataset = CIRCODataset(dataset_path, 'test', 'classic', preprocess)
    index_features, index_names = circo_extract_image_features(classic_test_dataset, clip_model)

    relative_test_dataset = CIRCODataset(dataset_path, 'test', 'relative', preprocess)

    # Get the predictions dict
    queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, clip_model, index_features, index_names)

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'circo'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{submission_name}.json", 'w+') as file:
        json.dump(queryid_to_retrieved_images, file, sort_keys=True)


def circo_generate_test_predictions(clip_model, relative_test_dataset):
    """
    Generate the test prediction features for the CIRCO dataset given the pseudo tokens
    """

    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=10,
                                      pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    query_ids_list = []
    predicted_features_modi_list = []

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        relative_captions = batch['relative_caption']
        text_modify = batch['text_composed']
        text_modify = clip.tokenize(text_modify, context_length=77, truncate=True).to(device)

        query_ids = batch['query_id']
        mental_img = batch['mental_img'].to(device)
        mental_img_des = batch['mental_img_des']
        tokenized_mental_img_des = clip.tokenize(mental_img_des, context_length=77, truncate=True).to(device)

        with torch.no_grad():
            mental_img_des_features = clip_model.encode_text(tokenized_mental_img_des)
            mental_img_features = clip_model.encode_image(mental_img)
            text_modify_features = clip_model.encode_text(text_modify)

        lamd = 0.3
        predicted_features = mental_img_des_features + mental_img_features
        predicted_features = lamd * F.normalize(predicted_features)
        predicted_features_modi = (1 - lamd) * F.normalize(text_modify_features)

        predicted_features_list.append(predicted_features)
        predicted_features_modi_list.append(predicted_features_modi)
        query_ids_list.extend(query_ids)

    predicted_features = torch.vstack(predicted_features_list)
    predicted_features_modi = torch.vstack(predicted_features_modi_list)
    return predicted_features, predicted_features_modi, query_ids_list


def circo_generate_test_dict(relative_test_dataset, clip_model, index_features, index_names):
    """
    Generate the test submission dicts for the CIRCO dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, predicted_features_modi, query_ids = circo_generate_test_predictions(clip_model, relative_test_dataset)

    # Normalize the index features
    index_features = index_features.to(device)
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the similarity
    similarity = predicted_features @ index_features.T
    similarity_modi = predicted_features_modi @ index_features.T
    sorted_indices = torch.topk(similarity+similarity_modi, dim=-1, k=50).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                   (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)}

    return queryid_to_retrieved_images




def main():
    parser = ArgumentParser()
    parser.add_argument("--submission-name", default="cirr_clipvitb", type=str, help="Filename of the generated submission file")
    parser.add_argument("--exp-name", type=str, help="Experiment to evaluate")
    parser.add_argument("--dataset", default="cirr", type=str, choices=['cirr', 'circo'], help="Dataset to use")
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

    if args.dataset.lower() == 'cirr':
        cirr_generate_test_submission_file(args.dataset_path, clip_model, preprocess, args.submission_name)
    elif args.dataset.lower() == 'circo':
        circo_generate_test_submission_file(args.dataset_path, clip_model, preprocess, args.submission_name)
    else:
        raise ValueError("Dataset not supported")


if __name__ == '__main__':
    main()
