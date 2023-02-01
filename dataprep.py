import pandas as pd
import os
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa


class ESC10Prep:

    def __init__(self, data_path, device='cpu', preprocess=False, transform=None, resample_rate=22050,
                 number_of_samples=22050):

        # Constructor
        self.data_path = os.path.join(data_path, 'audio')
        self.device = device
        self.preprocess = preprocess
        if transform is not None:
            self.transform = transform.to(self.device)
        else:
            self.transform = transform
        self.resample_rate = resample_rate
        self.number_of_samples = number_of_samples

        self.class_mapping = {"Dog bark": 0,
                              "Rain": 1,
                              "Sea waves": 2,
                              "Baby cry": 3,
                              "Clock tick": 4,
                              "Person sneeze": 5,
                              "Helicopter": 6,
                              "Chainsaw": 7,
                              "Rooster": 8,
                              "Fire crackling": 9
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

        return waveform, sample_rate, self.class_mapping[label]

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


class ESC50Prep(ESC10Prep):

    def __init__(self, data_path, device, preprocess=False, transform=None, resample_rate=22050,
                 number_of_samples=22050):

        super().__init__(data_path, device, preprocess, transform, resample_rate,
                 number_of_samples)

        self.class_mapping = {"Hand saw": 0,
                              "Fireworks": 1,
                              "Airplane": 2,
                              "Church bells": 3,
                              "Train": 4,
                              "Engine": 5,
                              "Car horn": 6,
                              "Siren": 7,
                              "Chainsaw": 8,
                              "Helicopter": 9,
                              "Glass breaking": 10,
                              "Clock tick": 11,
                              "Clock alarm": 12,
                              "Vacuum cleaner": 13,
                              "Washing machine": 14,
                              "Can opening": 15,
                              "Door - wood creaks": 16,
                              "Keyboard typing": 17,
                              "Mouse click": 18,
                              "Door knock": 19,
                              "Drinking - sipping": 20,
                              "Snoring": 21,
                              "Brushing teeth": 22,
                              "Laughing": 23,
                              "Footsteps": 24,
                              "Coughing": 25,
                              "Breathing": 26,
                              "Clapping": 27,
                              "Sneezing": 28,
                              "Crying baby": 29,
                              "Thunderstorm": 30,
                              "Toilet flush": 31,
                              "Pouring water": 32,
                              "Wind": 33,
                              "Water drops": 34,
                              "Chirping birds": 35,
                              "Crickets": 36,
                              "Crackling fire": 37,
                              "Sea waves": 38,
                              "Rain": 39,
                              "Crow": 40,
                              "Sheep": 41,
                              "Insects": 42,
                              "Hen": 43,
                              "Frog": 44,
                              "Cat": 45,
                              "Cow": 46,
                              "Pig": 47,
                              "Rooster": 48,
                              "Dog": 49
                              }


class UrbanSoundDatasetPrep(ESC10Prep):
    def __init__(self,
                 data_path,
                 device,
                 fold,
                 train=True,
                 preprocess=False,
                 transform=None,
                 resample_rate=22050,
                 number_of_samples=22050):

        super().__init__(data_path,
                         device,
                         preprocess,
                         transform,
                         resample_rate,
                         number_of_samples)

        self.data_path = data_path
        metadata_path = os.path.join(self.data_path, "metadata/UrbanSound8K.csv")
        self.fold = fold
        metadata_all = pd.read_csv(metadata_path)

        # split to train and val datasets according to folds
        if train:
            self.metadata = metadata_all[metadata_all['fold'] != fold]
        else:
            self.metadata = metadata_all[metadata_all['fold'] == fold]

        self.class_mapping = {"air_conditioner": 0,
                              "car_horn": 1,
                              "children_playing": 2,
                              "dog_bark": 3,
                              "drilling": 4,
                              "engine_idling": 5,
                              "gun_shot": 6,
                              "jackhammer": 7,
                              "siren": 8,
                              "street_music": 9
                              }
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # load an audio file along with its label
        filepath = self._get_audio_file_path(idx)
        label = self.metadata.iloc[idx, 6]
        waveform, sample_rate = torchaudio.load(filepath)

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

    def _get_audio_file_path(self, idx):
        filename = self.metadata.iloc[idx, 0]
        fold_number = self.metadata.iloc[idx, 5]
        filepath = os.path.join(self.data_path, "audio/fold" + str(fold_number), str(filename))
        return filepath


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(device)
    PATH = "data/ESC-50"
    dataset = ESC50Prep(PATH, device)
    print(dataset.class_mapping)
    length = len(dataset)

    print(dataset[1])
    #
    # # parameters for feature extraction (melspectrograms and mfcc)
    # N_FFT = 1024
    # HOP_LENGTH = 512
    # N_MELS = 64
    # N_MFCC = 13
    #
    # print(len(dataset))
    # print(dataset.class_mapping)

    RESAMPLE_RATE = 22050
    NUMBER_OF_SAMPLES = 22050
    #
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 64

    mels = T.MelSpectrogram(
        sample_rate=RESAMPLE_RATE,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )

    dataset = ESC50Prep(PATH,
                        device,
                        preprocess=True,
                        transform=mels,
                        resample_rate=RESAMPLE_RATE,
                        number_of_samples=NUMBER_OF_SAMPLES)


    print(dataset[1])

    labels = dataset.class_mapping
    # length = len(dataset)
    # print(labels)
    # print(length)
    n_rows = int((len(labels.keys()) / 2))

    fig, axs = plt.subplots(nrows=n_rows, ncols=2, figsize=(15, 24))
    plt.subplots_adjust(hspace=0.6)
    fig.suptitle("ESC10: melspectograms", y=0.93)

    for idx, ax in zip(range(0, length, 40), axs.ravel()):
        waveform, sample_rate, label = dataset[idx]

        im = ax.imshow(librosa.power_to_db(waveform[0]), origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax)

        ax.set_title(label)
        ax.set_xlabel('frame')
        ax.set_ylabel('freq_bin')

    # plt.savefig("figures/ESC10-processed_melspecs.png", bbox_inches='tight')

    #
    # n_rows = int((len(labels.keys()) / 2))
    #
    # fig, axs = plt.subplots(nrows=n_rows, ncols=2, figsize=(15, 12))
    # plt.subplots_adjust(hspace=0.6)
    # fig.suptitle("ESC10: mfcc", y=0.93)
    #
    # for idx, ax in zip(range(0, length, 40), axs.ravel()):
    #     waveform, sample_rate, label = dataset[idx]
    #
    #     im = ax.imshow(librosa.power_to_db(waveform[0].cpu()), origin="lower", aspect="auto")
    #     fig.colorbar(im, ax=ax)
    #
    #     key = [k for k, v in labels.items() if v == label]
    #     ax.set_title(key)
    #     ax.set_xlabel('frame')
    #     ax.set_ylabel('freq_bin')
    #
    plt.show()