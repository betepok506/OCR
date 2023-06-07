import os
import hydra
import torch
import logging
import argparse
from src.model.model import OCR
from src.data.data import extract_data, create_loaders
import torchvision.transforms as transforms
import json
import datetime
from src.enities.training_pipeline_params import TrainingPipelineParams
from src.logger_config.config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger()


@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def train_pipeline(params: TrainingPipelineParams):
    logger.info("Loading data...")
    image_file_paths, labels_encoded, token2ind, ind2token, blank_token, blank_ind, num_classes = extract_data(
        params.path_to_annotations)
    logger.info("Load completed!")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize(params.img_size),
        transforms.ToTensor()
    ])

    params.output_dir = os.path.join(params.output_dir, str(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")))
    os.makedirs(params.output_dir, exist_ok=True)
    if params.is_load:
        logger.info("Loading information from previous training")
        with open(os.path.join(params.path_to_info_for_model, "params.json"), "r") as infile:
            trainer_params = json.load(infile)
        trainer_params["ind2token"] = {int(k): v for k, v in trainer_params["ind2token"].items()}
        logger.info("Load completed!")
    else:
        trainer_params = {"blank_token": blank_token, "blank_ind": blank_ind, "token2ind": token2ind,
                          "ind2token": ind2token, "num_classes": num_classes}
    with open(os.path.join(params.output_dir, "params.json"), "w") as outfile:
        json.dump(trainer_params, outfile)

    logger.info(f"Training information is saved: {params.output_dir}")
    params.output_dir = os.path.join(params.output_dir, "checkpoints")
    train_loader, test_loader = create_loaders(image_file_paths,
                                               labels_encoded,
                                               transform,
                                               split=True,
                                               batch_size=params.batch_size,
                                               test_size=0.2)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f'Current device {device}')

    model = OCR(model_name=params.model_name,
                blank_token=trainer_params["blank_token"], blank_ind=trainer_params["blank_ind"],
                ind2token=trainer_params["ind2token"], token2ind=trainer_params["token2ind"],
                num_classes=trainer_params["num_classes"], output_dir=params.output_dir)
    model.to(device)
    if params.is_load:
        logger.info("Loading model")
        model.load(params.path_to_load)
        logger.info("Load completed!")

    logger.info("Starting training...")
    model.train(train_loader,
                test_loader,
                num_epochs=params.num_epochs)
    print("Training Done!")
    # print(model.evaluations(test_loader))


# def get_args():
#     parser = argparse.ArgumentParser(description='Train the OCR on images and target text')
#     parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10000, help='Number of epochs')
#     parser.add_argument('--path_to_data', type=str, default="./data/train/train", help='Path to folder with images')
#     parser.add_argument('--path_to_annotations', type=str, default="./data/list_images.csv",
#                         help='Path to annotations')
#     parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=768, help='Batch size')
#     parser.add_argument('--load', '-f', type=str, default=False,  # "./model/checkpoints/Epoch_129_loss_-0.11890.pt"
#                         help='Load model from a .pth file')
#     parser.add_argument('--output-dir', type=str, default="./model/",
#                         help='Load model from a .pth file')
#
#     return parser.parse_args()


if __name__ == "__main__":
    # args = get_args()
    # create_annotations(args.path_to_data, args.path_to_annotations)
    # fix_annotations(args.path_to_annotations, args.path_to_data, "./data/train/annotations.csv")
    # args.path_to_annotations = "../data/train/annotations.csv"
    train_pipeline()
