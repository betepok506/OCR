import os

import pandas as pd
from torch import nn
import torch
from torchmetrics import CharErrorRate
from torch.utils.data import DataLoader
from src.utils.utils import decode_batch_outputs, encoding, split_text
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
from torch.nn import Module, Sequential, Conv2d, AvgPool2d, GRU, Linear
from torchvision import models


class FeatureExtractor(Module):

    def __init__(self, input_size=(64, 224), output_len=20):
        super(FeatureExtractor, self).__init__()

        h, w = input_size
        resnet = getattr(models, 'resnet18')(pretrained=True)
        self.cnn = Sequential(*list(resnet.children())[:-2])

        self.pool = AvgPool2d(kernel_size=(h // 32, 1))
        self.proj = Conv2d(w // 32, output_len, kernel_size=1)

        self.num_output_features = self.cnn[-1][-1].bn2.num_features

    def apply_projection(self, x):
        """Use convolution to increase width of a features.

        Args:
            - x: Tensor of features (shaped B x C x H x W).

        Returns:
            New tensor of features (shaped B x C x H x W').
        """
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        return x

    def forward(self, x):
        # Apply conv layers
        features = self.cnn(x)

        # Pool to make height == 1
        features = self.pool(features)

        # Apply projection to increase width
        features = self.apply_projection(features)

        return features


class SequencePredictor(Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False):
        super(SequencePredictor, self).__init__()

        self.num_classes = num_classes
        self.rnn = GRU(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       dropout=dropout,
                       bidirectional=bidirectional)

        fc_in = hidden_size if not bidirectional else 2 * hidden_size
        self.fc = Linear(in_features=fc_in,
                         out_features=num_classes)

    def _init_hidden(self, batch_size):
        """Initialize new tensor of zeroes for RNN hidden state.

        Args:
            - batch_size: Int size of batch

        Returns:
            Tensor of zeros shaped (num_layers * num_directions, batch, hidden_size).
        """
        num_directions = 2 if self.rnn.bidirectional else 1

        # YOUR CODE HERE
        h = torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size)
        # END OF YOUR CODE

        return h

    def _reshape_features(self, x):
        """Change dimensions of x to fit RNN expected input.

        Args:
            - x: Tensor x shaped (B x (C=1) x H x W).

        Returns:
            New tensor shaped (W x B x H).
        """

        # YOUR CODE HERE
        x = x.squeeze(1)
        x = x.permute(2, 0, 1)
        # END OF YOUR CODE

        return x

    def forward(self, x):
        x = self._reshape_features(x)

        batch_size = x.size(1)
        h_0 = self._init_hidden(batch_size)
        h_0 = h_0.to(x.device)
        x, h = self.rnn(x, h_0)

        x = self.fc(x)
        return x


class CRNN(Module):

    def __init__(self, alphabet,
                 cnn_input_size=(64, 224), cnn_output_len=20,
                 rnn_hidden_size=128,
                 rnn_num_layers=2,
                 rnn_dropout=0.3,
                 rnn_bidirectional=False):
        super(CRNN, self).__init__()
        self.alphabet = alphabet
        self.features_extractor = FeatureExtractor(input_size=cnn_input_size, output_len=cnn_output_len)
        self.sequence_predictor = SequencePredictor(input_size=self.features_extractor.num_output_features,
                                                    hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                                                    num_classes=len(alphabet), dropout=rnn_dropout,
                                                    bidirectional=rnn_bidirectional)

    def forward(self, x):
        features = self.features_extractor(x)
        sequence = self.sequence_predictor(features)
        return sequence


class CNN_GRU(nn.Module):
    def __init__(self, output_size):
        super(CNN_GRU, self).__init__()
        self.dropout_percentage = 0.5
        self.conv_layers = nn.Sequential(
            # BLOCK-1 (starting block) input=(224x224) output=(56x56)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),

            # BLOCK-2 (1) input=(56x56) output = (56x56)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.Dropout(p=self.dropout_percentage),
        )

        self.linear_1 = nn.Linear(1472, 536)
        self.lstm = nn.GRU(536, 64, bidirectional=True, batch_first=True)
        self.linear_2 = nn.Linear(128, output_size)

    def forward(self, x):
        bs, _, _, _ = x.size()
        x = self.conv_layers(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        x = self.linear_1(x)
        x = nn.functional.relu(x)
        x, h = self.lstm(x)
        x = self.linear_2(x)
        x = x.permute(1, 0, 2)
        return x


class OCR:
    """
    Класс содержит реализация модели для распознования изображений с капчей
    """

    def __init__(self, model_name, blank_token, blank_ind, ind2token, token2ind, num_classes, output_dir=None):
        if model_name == "custom":
            self.model = CNN_GRU(num_classes)
        elif model_name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            self.model = CRNN(alphabet=[k for k in token2ind.keys()], cnn_output_len=num_classes)
        else:
            raise NotImplementedError

        self._device = "cpu"
        self._epoch = 0
        self._eval_loss = float("inf")
        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=3E-4)
        # self._criterion = nn.CTCLoss(blank=blank - 1, zero_infinity=True)
        self._criterion = nn.CTCLoss(zero_infinity=True)
        self._path_save_checkpoint = output_dir

        self.blank_token = blank_token
        self.blank_ind = blank_ind
        self.ind2token = ind2token
        self.token2ind = token2ind

    def train(self,
              train_loader,
              test_loader,
              num_epochs=500,
              save_checkpoint=True,
              visualize_learning=True,
              visualize_each=1):
        """
        Функция для обучения модели

        Parameters
        ------------
        train_loader: `DataLoader`
            Загрузчки данных для обучения
        test_loader: `DataLoader`
            Загрузчки данных для валидации
        encoder: `LabelEncoder`
            Кодировщик меток
        num_epochs: `int`
            Количество эпох обучения
        save_checkpoint: `bool`
            True если необходимо сохранять наилучщую модель при  обучении, иначе False
        visualize_learning: `bool`
            True если необходимо печатать процесс обучения, иначе False
        visualize_each: `int`
            Отвечает через сколько эпох печатать результат обучения

        Returns
        ------------
        `list`, `list`
            Массив потерь при обучении, массив потерь при валидации
        """

        training_loss = []
        evaluations_loss = []
        for epoch in range(num_epochs):
            train_loss, train_cer, train_examples = self._train_epoch(train_loader)
            eval_loss, eval_cer, eval_examples = self._validation_epoch(test_loader)
            training_loss.append(training_loss)
            evaluations_loss.append(eval_loss)
            self._epoch += 1

            if (epoch + 1) % visualize_each == 0:
                print(
                    f"Epoch: {self._epoch} Train loss: {train_loss} CER: {train_cer} Examples: {train_examples[:min(len(train_examples), 5)]}"
                    f"\n\tValidation: loss: {eval_loss} CER: {eval_cer} Examples: {eval_examples[:min(len(eval_examples), 5)]}")

            if save_checkpoint and self._eval_loss > eval_loss:
                self._eval_loss = eval_loss
                self.save(os.path.join(self._path_save_checkpoint,
                                       f"Epoch_{self._epoch}_loss_{self._eval_loss:.5f}.pt"),
                          True)

        return training_loss, evaluations_loss

    def _train_epoch(self, train_loader: DataLoader, visualize_learning: bool = True):
        """
        Функция для тренировки модели на наборе данных

        Parameters
        ------------
        train_loader: `DataLoader`
            Загрузчки набора данных

        Returns
        ------------
        `float`
            Ошибка при обучении
        """
        self.model.train()
        final_loss, cer = 0, 0
        result_learning = []
        for batch in tqdm(train_loader, desc="Training...", ncols=160):
            self._optimizer.zero_grad()
            images = batch["images"].to(self._device)
            targets = batch["seq"]
            seq_lens_gt = batch["seq_len"]
            decoded_batch_text = ''.join(encoding(targets.detach().cpu().tolist(), self.ind2token))
            texts = split_text(decoded_batch_text, seq_lens_gt.detach().cpu().tolist())

            output = self.model(images).cpu()
            batch_predictions_labels, _ = decode_batch_outputs(output, self.ind2token, self.blank_ind,
                                                               self.blank_token)
            cer += self._evaluations_cer(texts, batch_predictions_labels).item()
            if visualize_learning:
                result_learning.extend(
                    [(text, predict) for text, predict in zip(texts, batch_predictions_labels)])

            log_probs = nn.functional.log_softmax(output, 2)
            seq_lens_pred = torch.Tensor([output.size(0)] * output.size(1)).int()
            loss = self._criterion(log_probs=log_probs,  # (T, N, C)
                                   targets=targets,  # N, S or sum(target_lengths)
                                   input_lengths=seq_lens_pred,  # N
                                   target_lengths=seq_lens_gt)  # N

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            final_loss += loss.item()
            loss.detach()

        train_loss = final_loss / len(train_loader)
        return train_loss, cer, result_learning

    def _validation_epoch(self, test_loader, visualize_learning=True):
        """
        Функция для валидации модели на наборе данных

        Parameters
        ------------
        test_loader: `DataLoader`
            Загрузчки набора данных

        Returns
        ------------
        `float`
            Ошибка при валидации
        """

        self.model.eval()
        final_loss, cer = 0, 0
        index = np.random.choice(len(test_loader), 1, replace=False)
        result_learning = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validating...", ncols=160):
                images = batch["images"].to(self._device)
                targets = batch["seq"]
                seq_lens_gt = batch["seq_len"]
                decoded_batch_text = ''.join(encoding(targets.detach().cpu().tolist(), self.ind2token))
                texts = split_text(decoded_batch_text, seq_lens_gt.detach().cpu().tolist())

                outputs = self.model(images)

                batch_predictions_labels, _ = decode_batch_outputs(outputs, self.ind2token, self.blank_ind,
                                                                   self.blank_token)
                cer += self._evaluations_cer(texts, batch_predictions_labels).item()
                if visualize_learning:
                    result_learning.extend(
                        [(text, predict) for text, predict in zip(texts, batch_predictions_labels)])

                log_probs = nn.functional.log_softmax(outputs, 2)

                seq_lens_pred = torch.Tensor([outputs.size(0)] * outputs.size(1)).int()
                loss = self._criterion(log_probs=log_probs,  # (T, N, C)
                                       targets=targets,  # N, S or sum(target_lengths)
                                       input_lengths=seq_lens_pred,  # N
                                       target_lengths=seq_lens_gt)  # N
                # loss.requires_grad = True
                final_loss += loss.item()

        eval_loss = final_loss / len(test_loader)
        return eval_loss, cer, result_learning

    def to(self, device: str = "cpu"):
        """
        Функция для задачи устройства для обучения модели

        Parameters
        ------------
        device: `str`
            Устройство для обучения модели
        """
        self._device = device
        self.model.to(device)

    def evaluations(self,
                    test_loader: DataLoader,
                    each_image: bool = False):
        """
        Функция для оценки модели по метрике CharErrorRate

        Parameters
        ------------
        test_loader: `DataLoader`
            Загрузчки набора данных для тестирования
        encoder: `LabelEncoder`
            Кодировщик меток

        Returns
        ------------
        `float`
            Показатель оценки по метрике CharErrorRate
        """
        self.model.eval()
        # outputs = []
        with torch.no_grad():
            for batch in test_loader:
                images = batch["images"].to(self._device)
                targets = batch["seq"]
                seq_lens_gt = batch["seq_len"]
                decoded_batch_text = ''.join(encoding(targets.detach().cpu().tolist(), self.ind2token))
                texts = split_text(decoded_batch_text, seq_lens_gt.detach().cpu().tolist())

                batch_outputs = self.model(images)
                batch_predictions_labels, _ = decode_batch_outputs(batch_outputs, self.ind2token, self.blank_ind,
                                                                   self.blank_token)
                for text, predict in zip(texts, batch_predictions_labels):
                    print((text, predict))


    def _evaluations_cer(self, labels_true, labels_pred):
        """
        Функция для оценки строк или набора строк по метрике CharErrorRate

        Parameters
        ------------
        labels_true: `list[str]` or `str`
            Массив истинных меток
        labels_pred: `list[str]` or `str`
            Массив предсказанных меток

        Returns
        ------------
        `float`
            Показатель оценки по метрике CharErrorRate
        """
        cer = CharErrorRate()
        return cer(labels_true, labels_pred)

    def predict(self, loader: DataLoader, path_to_save: str):
        self.model.eval()

        images_name, predict_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader, total=len(loader), desc="Predicting ", ncols=160):
                images = batch["images"].to(self._device)
                images_name.extend([os.path.basename(name) for name in batch["img_name"]])
                batch_outputs = self.model(images)
                batch_predictions_labels, _ = decode_batch_outputs(batch_outputs, self.ind2token, self.blank_ind,
                                                                   self.blank_token)
                predict_labels.extend(batch_predictions_labels)

        df = pd.DataFrame(data={"Id": images_name, "Predicted": predict_labels})
        df.to_csv(path_to_save, index=False, encoding='utf8')

    def error_calculation_each_image(self, test_loader):
        pass

    def save(self, path_to_save: str, training: bool = False):
        """
        Функция для сохранения модели

        Parameters
        ------------
        path_to_save: `str`
            Путь, куда будет сохранена модель
        training: `bool`
            Флаг означающий стоит ли сохранять информацию, необходимую для продолжения обучения
        """
        path_to_folder = os.path.split(path_to_save)
        if len(path_to_folder) == 2:
            os.makedirs(path_to_folder[0], exist_ok=True)
        state = {
            'model_state_dict': self.model.state_dict()
        }
        if training:
            state['optimizer_state_dic'] = self._optimizer.state_dict()
            state['epoch'] = self._epoch
            state['loss'] = self._eval_loss

        torch.save(state, path_to_save)

    def load(self, path_to_model):
        """
        Функция для загрузки модели

        Parameters
        ------------
        path_to_model: `str`
            Путь до загружаемое модели
        """
        checkpoint = torch.load(path_to_model)
        if "optimizer_state_dic" in checkpoint:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dic"])

        if "epoch" in checkpoint:
            self._epoch = checkpoint["epoch"]

        if "epoch" in checkpoint:
            self._eval_loss = checkpoint["loss"]

        self.model.load_state_dict(checkpoint["model_state_dict"])
