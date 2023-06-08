import torch
import argparse
from src.model.model import OCR
from src.data.data import create_loaders, create_list_files
import torchvision.transforms as transforms
import pandas as pd
import json
import os
import hydra
import logging

from src.enities.prediction_pipeline_params import PredictingPipelineParams
from src.logger_config.config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()


@hydra.main(version_base=None, config_path='../configs', config_name='predict_config')
def predict_pipeline(params: PredictingPipelineParams):
    if params.create_annotations:
        create_list_files(params.path_to_data, params.path_to_annotations)
        logger.info(f"Annotations created!")

    annotations = pd.read_csv(params.path_to_annotations)
    image_file_paths = annotations.iloc[:, 0].tolist()

    with open(os.path.join(params.path_to_info_for_model, "params.json"), "r") as infile:
        trainer_params = json.load(infile)

    trainer_params["ind2token"] = {int(k): v for k, v in trainer_params["ind2token"].items()}
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Grayscale(num_output_channels=1),
        # transforms.Resize((64, 224)),
        transforms.Resize(params.img_size),
        transforms.ToTensor()
    ])

    predict_loader = create_loaders(image_file_paths,
                                    transform=transform,
                                    split=False,
                                    batch_size=params.batch_size)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f'Current device {device}')
    logger.info(f"Model name: {params.model_name}")

    model = OCR(model_name=params.model_name, blank_token=trainer_params["blank_token"],
                blank_ind=trainer_params["blank_ind"],
                ind2token=trainer_params["ind2token"],
                token2ind=trainer_params["token2ind"], num_classes=trainer_params["num_classes"])
    model.to(device)

    logger.info(f"Loading model... Path to model: {params.path_to_model}")
    model.load(params.path_to_model)

    model.predict(predict_loader, params.output_dir)


if __name__ == "__main__":
    predict_pipeline()
