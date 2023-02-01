import pandas as pd
import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import librosa


class ESC10Prep:

    def __init__(self, data_path, device, preprocess=False, transform=None, resample_rate=22050, number_of_samples=22050):

        # Constructor
        self.data_path = os.path.join(data_path, 'audio')
        self.device = device
        self.preprocess = preprocess
        self.transform = transform.to(self.device)
        self.resample_rate = resample_rate
        self.number_of_samples = number_of_samples

        self.class_mapping = {"Dog Bark": 0,
                              "Rain": 1,
                              "Sea Waves": 2,
                              "Baby cry": 3,
                              "Clock tick": 4,
                              "Person sneeze": 5,
                              "Helicopter": 6,
                              "Chainsaw": 7,
                              "Rooster": 8,
                              "Fire Cracking": 9
                              }

    def __len__(self):
        return len(self._get_file_list())

    def __getitem__(self, idx):
        items = self._get_file_list()
        filename, label = items[idx]
        waveform, sample_rate = torchaudio.load(filename)
        waveform = waveform.to(self.device)

        # preprocess if necessary
        if self.preprocess:
            # keep only one channel
            number_of_channels = waveform.shape[0]
            if number_of_channels > 1:
                waveform = self._calc_mean_of_channels(waveform)
            # make sure that signal rate is equal
            if sample_rate != self.resample_rate:
                waveform = self._resampling(waveform, self.resample_rate)
                sample_rate = self.resample_rate
            # make sure that number of samples is equal either by downsampling or upsampling
            num_of_samples = waveform.shape[1]
            if num_of_samples > self.number_of_samples:
                waveform = self._down_sample(waveform)
            if num_of_samples < self.number_of_samples:
                waveform = self._up_sample(waveform)

        # transform if necessary
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample_rate, label

    def _get_file_list(self):
        items = []
        for (dirpath, _, filenames) in os.walk(self.data_path):
            if dirpath is not self.data_path:
                # get label
                label = os.path.basename(dirpath)[6:]
                # get files
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    items.append([filepath, label])
        return items

    def _resampling(self, signal, sr):
        resampler = T.Resample(sr, self.resample_rate)
        resampled_waveform = resampler(signal)
        return resampled_waveform

    def _calc_mean_of_channels(self, signal):
        signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _down_sample(self, signal):
        signal = signal[:, :self.number_of_samples]
        return signal

    def _up_sample(self, signal):
        num_of_samples = signal.shape[1]
        num_of_missing_samples = self.number_of_samples - num_of_samples
        signal = torch.nn.functional.pad(signal, (0, num_of_missing_samples))
        return signal

class ESC50prep(ESC10Prep):
    def __init__(self):
        super().__init__()

        self.class_mapping = {"Dog Bark": 0,
                              "Rain": 1,
                              "Sea Waves": 2,
                              "Baby cry": 3,
                              "Clock tick": 4,
                              "Person sneeze": 5,
                              "Helicopter": 6,
                              "Chainsaw": 7,
                              "Rooster": 8,
                              "Fire Cracking": 9
                              }

if __name__ == "__main__":

    # # general parameters
    PATH = "data/ESC-50"
    # SAMPLE_RATE = 44100  # targeted sample for all files
    # NUM_OF_SAMPLES = 44100  # targeted number of sample of each file
    #
    # # parameters for feature extraction (melspectrograms and mfcc)
    # N_FFT = 1024
    # HOP_LENGTH = 512
    # N_MELS = 64
    # N_MFCC = 13
    #
    dataset = ESC50Prep(PATH)
    # df = dataset.
    labels = dataset.class_mapping
    print(labels)
    #
    # sample_idx = 10
    # for lbl in labels.keys():
    #     index = df.index[df['class'] == lbl].tolist()
    #     wave, sr, lbl = dataset.__getitem__(index[sample_idx])
    #     print(wave)
        # mel_spectro = dataset.calc_mel_spec(waveform)
        # mfcc = dataset.calc_mfcc(waveform)

        # dataset.plot_spectrogram(mfcc[0])
        # plt.show()
    # sample_rate = 22050
    #
    # n_fft = 1024
    # win_length = None
    # hop_length = 512
    # n_mels = 64
    #
    # mel_spectrogram = T.MelSpectrogram(
    #     sample_rate=sample_rate,
    #     n_fft=n_fft,
    #     win_length=win_length,
    #     hop_length=hop_length,
    #     center=True,
    #     pad_mode="reflect",
    #     power=2.0,
    #     norm='slaney',
    #     onesided=True,
    #     n_mels=n_mels,
    #     mel_scale="htk",
    # )
    #
    # dataset_melspecs = UrbanSoundPrep(PATH, preprocess=True, transform=mel_spectrogram, resample_rate=22050, number_of_samples=22050)
    #
    # train = DataLoader(dataset_melspecs, batch_size=64)
    # print(train.dataset)

    # items = [(str(file), file.name.split('-')[-1].replace('.wav', '')) for file in files]

    # items = []
    # for (dirpath, _, filenames) in os.walk(PATH):
    #     if dirpath is not PATH:
    #         # get label
    #         label = os.path.basename(dirpath)[6:]
    #         # get files
    #         for filename in filenames:
    #             filepath = os.path.join(dirpath, filename)
    #             items.append([label, filepath])

                # print(filepath)

