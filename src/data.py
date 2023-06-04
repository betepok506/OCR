import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from skimage import io
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os


def show_landmarks(image):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.pause(0.001)  # pause a bit so that plots are updated


class CaptchaDataset(Dataset):
    """Класс содержит собственную реализацию Dataset унаследованную от `torch.utils.data.Dataset`"""

    def __init__(self, paths_to_images, targets_encoded, transforms=None):
        self.paths_to_images = paths_to_images
        self.targets = targets_encoded
        self.transform = transforms

    def __len__(self):
        return len(self.paths_to_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.paths_to_images[idx]
        image = io.imread(img_name)

        target = self.targets[idx]
        tensorized_target = torch.tensor(target, dtype=torch.int)
        # tensorized_target = target

        if self.transform:
            image = self.transform(image)

        return dict(image=image, seqs=tensorized_target)


def create_loaders(paths_to_images: list,
                   labels: list,
                   transform: transforms,
                   batch_size: int = 16,
                   test_size: float = 0.2):
    """
    Функция реализует создание наборов данных

    Parameters
    ------------
    paths_to_images: `list`
        Массив, содержащий пути до изображений
    labels: `np.array`
        Массив, содержащий закодированные метки
    transform: `transforms`
        Преобразования, которые необходимо применить к данным
    batch_size: `int`
        Размер батча
    test_size: `float`
        Размер тестовой части

    Returns
    ------------
    `DataLoader`, `DataLoader`
        Загрузчик данных обучающей части датасета, загрузчик данных тестовой части датасета
    """

    train_img, test_img, train_targets, test_targets = train_test_split(paths_to_images,
                                                                        labels,
                                                                        test_size=test_size,
                                                                        random_state=7)

    train_dataset = CaptchaDataset(train_img, train_targets, transforms=transform)
    test_dataset = CaptchaDataset(test_img, test_targets, transforms=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=collate_fn)

    return train_loader, test_loader


def extract_data(path_to_file: str, blank_token="<BLANK>"):
    """
    Функция для чтения файла аннотации и преобразования данных в формат датасета.
    Для кодирования меток используется `LabelEncoder`

    Parameters
    ------------
    path_to_file: `str`
        Путь до файла с аннотацией

    Returns
    ------------
    `list`, `np.array`, `LabelEncoder`
        Массив путей до изображений, массив закодированных меток, кодировщик меток
    """

    annotations = pd.read_csv(path_to_file)
    paths_to_images = annotations.iloc[:, 0].tolist()
    targets_orig = annotations.iloc[:, 1]

    labels = targets_orig.tolist()
    # targets = []
    #
    # for ind, x in enumerate(labels):
    #     t = []
    #     try:
    #         for c in x:
    #             t.append(c)
    #     except:
    #         print(f"{ind=} {x=}")
    #         continue
    #
    #     targets.append(t)
    targets = [[c for c in x] for x in labels]
    # targets_flat = [c for clist in targets for c in clist]
    targets_flat = set()
    for clist in targets:
        for c in clist:
            targets_flat.add(c)

    targets_flat = [blank_token] + sorted(list(targets_flat))
    # label_encoder = preprocessing.LabelEncoder()
    # label_encoder.fit(targets_flat)
    token2ind = {target: ind for ind, target in enumerate(targets_flat)}
    ind2token = {ind: target for ind, target in enumerate(targets_flat)}
    targets_enc = [[token2ind[token] for token in list_token] for list_token in targets]

    # token2ind = {token: label_encoder.transform([token])[0] for token in targets_flat}
    # token2ind = {}
    # for token in targets_flat:
    #     token2ind[token] = label_encoder.transform([token])[0]
    # targets_enc = [label_encoder.transform(x) for x in targets]
    # targets_enc = as_matrix(targets, label_encoder, blank_token=BLANK)

    # targets_enc = [token2ind[x] for x in targets]
    # targets_enc = np.array(targets_enc)

    return paths_to_images, targets_enc, \
           token2ind, ind2token, \
           blank_token, token2ind[blank_token], len(token2ind)
    # return paths_to_images, targets_enc, label_encoder, label_encoder.transform([BLANK])[0], len(token2ind)
    # return paths_to_images, targets_enc


def collate_fn(batch):
    """Function for torch.utils.data.Dataloader for batch collecting.

    Args:
        - batch: List of dataset __getitem__ return values (dicts).

    Returns:
        Dict with same keys but values are either torch.Tensors of batched images or sequences or so.
    """

    images, seqs, seq_lens = [], [], []
    # images, seqs, seq_lens, texts = [], [], [], []
    for item in batch:
        images.append(item["image"])

        seq_lens.append(len(item["seqs"]))
        seqs.extend(item["seqs"])

    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()
    batch = {"images": images, "seq": seqs, "seq_len": seq_lens}
    return batch


def as_matrix(sequences, encoder, max_len=None, blank_token="<BLANK>"):
    """ Convert a list of tokens into a matrix with padding """
    if isinstance(sequences[0], str):
        sequences = list(map(str.split, sequences))
    # sequences = [encoder.transform(x) for x in sequences]

    max_len = min(max(map(len, sequences)), max_len or float('inf'))

    matrix = np.full((len(sequences), max_len), np.int32(encoder.transform([blank_token])[0]))
    for i, seq in enumerate(sequences):
        row_ix = [encoder.transform([word])[0] for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix

    return matrix


def create_annotations(path_to_dataset: str, path_to_save: str):
    """
    Функция для создания файла аннотации

    Parameters
    ------------
    path_to_dataset: `str`
        Путь к папке с изображениями
    path_to_save: `str`
        Путь, где будет сохранен файл с аннотацией
    """
    paths_to_images = []
    for file_name in os.listdir(path_to_dataset):
        if len(file_name.split('.')) == 2:
            decoding = file_name.split('.')[0]
            paths_to_images.append([os.path.join(path_to_dataset, file_name), decoding])

    pd.DataFrame(data=paths_to_images).to_csv(path_to_save, index=False, header=False)


def fix_annotations(path_to_annotations: str, path_to_dataset: str, path_to_save: str):
    annotations = pd.read_csv(path_to_annotations)
    initial_length = len(annotations)
    annotations = annotations.dropna()
    print(f"The length of the dataset after deleting NaN: {initial_length - len(annotations)}")
    paths_to_images = annotations.iloc[:, 0].tolist()
    new_paths_to_images = []
    for ind, cur_path in enumerate(paths_to_images):
        name = os.path.basename(cur_path)
        new_paths_to_images.append(os.path.join(path_to_dataset, name))

    pd.DataFrame({"id": new_paths_to_images, "labels": annotations.iloc[:, 1].tolist()}).to_csv(path_to_save,
                                                                                                header=False,
                                                                                                index=False)
