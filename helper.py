import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch


def plot_waveforms(dataset, title):
    labels = dataset.class_mapping
    length = len(dataset)
    n_rows = int((len(labels.keys()) / 2))

    fig, axs = plt.subplots(nrows=n_rows, ncols=2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.6)
    fig.suptitle(title, y=0.93)

    for lbl, ax in enumerate(axs.ravel()):
        for idx in range(0, length):
            _, _, label = dataset[idx]
            if label == lbl:
                waveform, sample_rate, _ = dataset[idx]
                num_channels, num_frames = waveform.shape
                time_axis = torch.arange(0, num_frames) / sample_rate
                ax.plot(time_axis, waveform[0])
                ax.set_title(label)
                ax.set_xlabel('Time (sec)')
                ax.set_ylabel('Amplitude (a.u.)')
                key = [k for k, v in labels.items() if v == label]
                ax.set_title(key)
                ax.set_xlabel('frame')
                ax.set_ylabel('freq_bin')
                break

def plot_features(dataset, title):
    labels = dataset.class_mapping
    length = len(dataset)
    n_rows = int((len(labels.keys()) / 2))

    fig, axs = plt.subplots(nrows=n_rows, ncols=2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.6)
    fig.suptitle(title, y=0.93)

    for lbl, ax in enumerate(axs.ravel()):
        for idx in range(0, length):
            _, _, label = dataset[idx]
            if label == lbl:
                waveform, _, _ = dataset[idx]
                im = ax.imshow(librosa.power_to_db(waveform[0].cpu()), origin="lower", aspect="auto")
                fig.colorbar(im, ax=ax)
                key = [k for k, v in labels.items() if v == label]
                ax.set_title(key)
                ax.set_xlabel('frame')
                ax.set_ylabel('freq_bin')
                break


def plot_histories(histories, epochs, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for experiment in histories:
        train_losses = histories[experiment][0]
        val_losses = histories[experiment][1]
        train_accuracies = histories[experiment][2]
        val_accuracies = histories[experiment][3]

        x_axis = range(1, epochs + 1)
        ax1.plot(x_axis, train_losses, label='train loss:' + str(experiment))
        ax1.plot(x_axis, val_losses, label='validation loss:' + str(experiment))
        ax1.set_xlabel("Epoch #")
        ax1.set_ylabel("Loss (a.u.)")
        # ax1.legend()
        ax2.plot(x_axis, train_accuracies, label='train accuracy:' + str(experiment))
        ax2.plot(x_axis, val_accuracies, label='validation accuracy:' + str(experiment))
        ax2.set_xlabel("Epoch #")
        ax2.set_ylabel("Accuracy (a.u.)")
        ax2.legend()
        fig.suptitle(title)


def plot_history(histories, experiment, epochs, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    train_losses = histories[experiment][0]
    val_losses = histories[experiment][1]
    train_accuracies = histories[experiment][2]
    val_accuracies = histories[experiment][3]

    x_axis = range(1, epochs + 1)
    ax1.plot(x_axis, train_losses, label='train loss:' + str(experiment))
    ax1.plot(x_axis, val_losses, label='validation loss:' + str(experiment))
    ax1.set_xlabel("Epoch #")
    ax1.set_ylabel("Loss (a.u.)")
    # ax1.legend()
    ax2.plot(x_axis, train_accuracies, label='train accuracy:' + str(experiment))
    ax2.plot(x_axis, val_accuracies, label='validation accuracy:' + str(experiment))
    ax2.set_xlabel("Epoch #")
    ax2.set_ylabel("Accuracy (a.u.)")
    ax2.legend()
    fig.suptitle(title + " - " + str(experiment))

    # plt.savefig("ESC10_MLP-mels-base_" + str(experiment) + ".png")


class Scaler(object):
    """ a simple minmax rescaler between 0 and 1
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        _input = sample

        min, max = _input.min(), _input.max()
        new_min, new_max = 0, 1

        _input = (_input - min) / (max - min) * (new_max - new_min) + new_min

        return _input


def conv2d_output_size(input_size, out_channels, padding, kernel_size, stride, dilation=None):
    """According to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    if dilation is None:
        dilation = (1, ) * 2
    if isinstance(padding, int):
        padding = (padding, ) * 2
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, ) * 2
    if isinstance(stride, int):
        stride = (stride, ) * 2

    output_size = (
        out_channels,
        np.floor((input_size[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) /
                 stride[0] + 1).astype(int),
        np.floor((input_size[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) /
                 stride[1] + 1).astype(int)
    )
    return output_size
