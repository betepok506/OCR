import torch


def decode_model_output(model_output, encoder):
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
    for n in model_output_BPA_applied:
        if n == 19:
            model_ouput_label_decoded.append("_")
        else:
            c = encoder.inverse_transform([n])[0]

            model_ouput_label_decoded.append(c)

    model_ouput_without_dublicates = []
    for i in range(len(model_ouput_label_decoded)):
        if i == 0:
            model_ouput_without_dublicates.append(model_ouput_label_decoded[i])
        else:
            if model_ouput_without_dublicates[-1] != model_ouput_label_decoded[i]:
                model_ouput_without_dublicates.append(model_ouput_label_decoded[i])

    model_ouput_without_blanks = []
    for e in model_ouput_without_dublicates:
        if e != "_":
            model_ouput_without_blanks.append(e)
    prediction = "".join(model_ouput_without_blanks)

    return prediction, model_ouput_label_decoded


def decode_batch_outputs(batch_outputs, encoder):
    """
    Функция для декодирования батча

    Parameters
    ------------
    batch_outputs: `tensor`
    encoder: `LabelEncoder`

    Returns
    ------------
    """
    predictions_ctc = []
    predictions_labels = []
    for j in range(batch_outputs.shape[1]):
        temp = batch_outputs[:, j, :].unsqueeze(1)
        #         [25,20] > [25,1,20]
        prediction_label, prediction_ctc = decode_model_output(temp, encoder)
        predictions_ctc.append(prediction_ctc)
        predictions_labels.append(prediction_label)

    return predictions_labels, predictions_ctc


