import os
import hydra
import torch
import logging
from src.data.data import augmentation_data
from src.utils.utils import mean_std_for_loader, dataset_statistics
import torchvision.transforms as transforms
from src.enities.training_pipeline_params import TrainingPipelineParams
from src.logger_config.config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()


@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def statistics_pipeline(params: TrainingPipelineParams):
    augmentation_data(params.path_to_annotations, "./datasets/train/aug_train", "./datasets/train/aug_annotations.csv")


if __name__ == "__main__":
    statistics_pipeline()
