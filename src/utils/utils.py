import os
import cv2 as cv
from tqdm import tqdm
import torch


def encoding(labels, dict_encoding):
    result = []
    for ch in labels:
        result.append(dict_encoding[ch])
    return result


def split_text(text, seq_lens):
    last_ind = 0
    results = []
    for cur_len in seq_lens:
        results.append(text[max(0, last_ind): last_ind + cur_len])
        last_ind = last_ind + cur_len
    return results


def encoding_ind2token():
    pass


def decode_model_output(model_output, ind2token, blank_ind, blank_token):
    """
    Функция декодирования вывода модели

    Parameters
    ------------
    model_output: `tensor`
        Вывод модели
    encoder: `LabelEncoder`
        Кодировщик меток

    Returns
    ------------
    `str`, `list`
        Предсказанная метка, массив предсказанных символов
    """
    model_output_permuted = model_output.permute(1, 0, 2)
    model_output_converted_to_probabilities = torch.softmax(model_output_permuted, 2)
    model_output_BPA_applied_gpu = torch.argmax(model_output_converted_to_probabilities, 2)
    model_output_BPA_applied = model_output_BPA_applied_gpu.detach().cpu().numpy().squeeze()

    model_ouput_label_decoded = []
    for ind in model_output_BPA_applied:
        if ind == blank_ind:
            model_ouput_label_decoded.append(blank_token)
        else:
            c = ind2token[ind]
            model_ouput_label_decoded.append(c)

    model_ouput_without_dublicates = []
    for i in range(len(model_ouput_label_decoded)):
        if i == 0:
            model_ouput_without_dublicates.append(model_ouput_label_decoded[i])
        else:
            if model_ouput_without_dublicates[-1] != model_ouput_label_decoded[i]:
                model_ouput_without_dublicates.append(model_ouput_label_decoded[i])

    model_ouput_without_blanks = []
    for ch in model_ouput_without_dublicates:
        if ch != blank_token:
            model_ouput_without_blanks.append(ch)
    prediction = "".join(model_ouput_without_blanks)

    return prediction, model_ouput_label_decoded


def decode_batch_outputs(batch_outputs, ind2token, blank_ind, blank_token):
    """
    Функция для декодирования батча

    Parameters
    ------------
    batch_outputs: `tensor`
    encoder: `LabelEncoder`

    Returns
    ------------
    """
    predictions_ctc, predictions_labels = [], []

    for j in range(batch_outputs.shape[1]):
        temp = batch_outputs[:, j, :].unsqueeze(1)
        prediction_label, prediction_ctc = decode_model_output(temp, ind2token=ind2token,
                                                               blank_ind=blank_ind, blank_token=blank_token)
        predictions_ctc.append(prediction_ctc)
        predictions_labels.append(prediction_label)

    return predictions_labels, predictions_ctc


def mean_std_for_loader(loader: torch.utils.data.DataLoader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for batch in tqdm(loader, total=len(loader)):
        images = batch["images"]
        for d in range(3):
            mean[d] += images[:, d, :, :].mean()
            std[d] += images[:, d, :, :].std()

    mean.div_(len(loader))
    std.div_(len(loader))
    return list(mean.numpy()), list(std.numpy())


def dataset_statistics(path_to_data: str):
    stat = {
        "inv_images": 0,
        "height_inv_images": 0,
        "width_inv_images": 0,
        "total_images": 0,
        "height_total_images": 0,
        "width_total_images": 0,
        "normal_images": 0,
        "height_normal_images": 0,
        "width_normal_images": 0,
    }

    for name_file in tqdm(os.listdir(path_to_data)):
        img = cv.imread(os.path.join(path_to_data, name_file))
        h, w, _ = img.shape
        if h > w * 1.3:
            type_img = "inv"
        else:
            type_img = "normal"

        for type_ in ["total", type_img]:
            stat[f"{type_}_images"] += 1
            stat[f"height_{type_}_images"] += h
            stat[f"width_{type_}_images"] += w

    for type_ in ["total", "inv", "normal"]:
        stat[f"height_{type_}_images"] /= stat[f"{type_}_images"]
        stat[f"width_{type_}_images"] /= stat[f"{type_}_images"]

    return stat
