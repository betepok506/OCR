import os
import hydra
import torch
import logging
from src.data.data import extract_data, create_loaders, fix_annotations
from src.utils.utils import mean_std_for_loader, dataset_statistics
import torchvision.transforms as transforms
from src.enities.training_pipeline_params import TrainingPipelineParams
from src.logger_config.config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()


@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def statistics_pipeline(params: TrainingPipelineParams):
    if params.create_annotations_file:
        logger.info("Correction of the annotation file")
        fix_annotations(params.path_to_annotations, params.path_to_data, params.path_to_annotations)

    logger.info("Loading data...")
    image_file_paths, labels_encoded, token2ind, ind2token, blank_token, blank_ind, num_classes = extract_data(
        params.path_to_annotations)
    logger.info("Load completed!")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(params.img_size),
        transforms.ToTensor()
    ])
    # train_loader = create_loaders(image_file_paths,
    #                               labels_encoded,
    #                               transform,
    #                               split=False,
    #                               batch_size=params.batch_size)
    # means, stds = mean_std_for_loader(train_loader)
    # logger.info(f"Means: {means}")
    # logger.info(f"Stds: {stds}")
    logger.info("-" * 60)
    dataset_stat = dataset_statistics(params.path_to_data)
    logger.info(f"Total images: {dataset_stat['total_images']}")
    logger.info("-" * 60)
    logger.info(f"Average height of all images: {dataset_stat['height_total_images']}")
    logger.info(f"Average width of all images: {dataset_stat['width_total_images']}")
    logger.info("-" * 60)
    logger.info(f"Average height of inverted images: {dataset_stat['height_inv_images']}")
    logger.info(f"Average width of inverted images: {dataset_stat['width_inv_images']}")
    logger.info("-" * 60)
    logger.info(f"Average height of normal images: {dataset_stat['height_normal_images']}")
    logger.info(f"Average width of normal images: {dataset_stat['width_normal_images']}")



if __name__ == "__main__":
    statistics_pipeline()
