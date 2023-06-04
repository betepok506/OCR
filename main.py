import torch
import argparse
from src.model import OCR
from src.data import extract_data, create_loaders, create_annotations, fix_annotations, collate_fn
import torchvision.transforms as transforms


def train_pipeline(args):
    # image_file_paths, labels_encoded, LabelEncoder = extract_data(args.path_to_annotations)
    image_file_paths, labels_encoded, LabelEncoder, blank, num_classes = extract_data(args.path_to_annotations)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((50,400)),
        transforms.ToTensor()
    ])

    train_loader, test_loader = create_loaders(image_file_paths,
                                               labels_encoded,
                                               transform,
                                               args.batch_size,
                                               test_size=0.2)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Current device {device}')

    model = OCR(blank, num_classes)
    model.to(device)
    if args.load:
        print("Loading model")
        model.load(args.load)

    model.train(train_loader,
                test_loader,
                LabelEncoder,
                num_epochs=args.epochs)
    print(model.evaluations(test_loader, LabelEncoder))


def get_args():
    parser = argparse.ArgumentParser(description='Train the OCR on images and target text')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10000, help='Number of epochs')
    parser.add_argument('--path_to_data', type=str, default="./data/train/train", help='Path to folder with images')
    parser.add_argument('--path_to_annotations', type=str, default="./data/train_labels.csv",
                        help='Path to annotations')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=128, help='Batch size')
    parser.add_argument('--load', '-f', type=str, default=False,
                        help='Load model from a .pth file')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # create_annotations(args.path_to_data, args.path_to_annotations)
    # fix_annotations(args.path_to_annotations, args.path_to_data, "./data/train/annotations.csv")
    args.path_to_annotations = "./data/train/annotations.csv"
    train_pipeline(args)
