import pandas as pd
import os
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa


class UrbanSoundPrep:

    def __init__(self,
                 data_path,
                 resample_rate=22050,
                 number_of_samples=44100,
                 n_fft=1024,
                 hop_length=512,
                 n_mels=64,
                 n_mfcc=13):

        # Constructor
        self.data_path = data_path
        metadata_path = os.path.join(self.data_path, "metadata/UrbanSound8K.csv")
        self.metadata = pd.read_csv(metadata_path)

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

        self.resample_rate = resample_rate
        self.number_of_samples = number_of_samples
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

    def get_dataset_length(self):
        length = len(self.metadata)
        return length

    def _get_audio_file_path(self, idx):
        filename = self.metadata.iloc[idx, 0]
        fold_number = self.metadata.iloc[idx, 5]
        filepath = os.path.join(self.data_path, "audio/fold" + str(fold_number), str(filename))
        return filepath

    def get_raw_waveform(self, idx):
        filepath = self._get_audio_file_path(idx)
        label = self.metadata.iloc[idx, 6]
        waveform, sr = torchaudio.load(filepath)

        return waveform, sr, label

    def get_processed_waveform(self, idx):
        waveform, sr, label = self.get_raw_waveform(idx)

        # keep only one channel
        number_of_channels = waveform.shape[0]
        if number_of_channels > 1:
            waveform = self._calc_mean_of_channels(waveform)

        # make sure that signal rate is equal
        if sr != self.resample_rate:
            waveform = self._resampling(waveform, self.resample_rate)

        # make sure that number of samples is equal either by downsampling or upsampling
        num_of_samples = waveform.shape[1]
        if num_of_samples > self.number_of_samples:
            waveform = self._down_sample(waveform)
        if num_of_samples < self.number_of_samples:
            waveform = self._up_sample(waveform)

        return waveform, self.resample_rate, label

    def _resampling(self, waveform, sr):
        resampler = T.Resample(sr, self.resample_rate)
        resampled_waveform = resampler(waveform)
        return resampled_waveform

    def _calc_mean_of_channels(self, waveform):
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _down_sample(self, waveform):
        waveform = waveform[:, :self.number_of_samples]
        return waveform

    def _up_sample(self, waveform):
        num_of_samples = waveform.shape[1]
        num_of_missing_samples = self.number_of_samples - num_of_samples
        waveform = torch.nn.functional.pad(waveform, (0, num_of_missing_samples))
        return waveform

    def calc_mfcc(self, waveform):
        mfcc_transform = T.MFCC(
            sample_rate=self.resample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'n_mels': self.n_mels,
                'hop_length': self.hop_length,
                'mel_scale': 'htk',
            }
        )

        mfcc = mfcc_transform(waveform)

        return mfcc

    def calc_mel_spec(self, waveform):

        # define transformation
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.resample_rate,
            n_fft=self.n_fft,
            win_length=None,
            hop_length=self.hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=self.n_mels,
            mel_scale="htk",
        )
        # Perform transformation
        mel_spec = mel_spectrogram(waveform)

        return mel_spec

    def plot_waveform(self, waveform, sample_rate, label):

        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c + 1}")

        key = [k for k, v in self.class_mapping.items() if v == label]
        figure.suptitle(key)

        return figure

    def plot_spectrogram(self, spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
        fig, axs = plt.subplots(1, 1)
        axs.set_title(title or 'Spectrogram (db)')
        axs.set_ylabel(ylabel)
        axs.set_xlabel('frame')
        im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
        if xmax:
            axs.set_xlim((0, xmax))
        fig.colorbar(im, ax=axs)
        plt.show(block=False)

    def plot_specgram(self, waveform, title="Spectrogram", xlim=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / self.resample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=self.resample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
        figure.suptitle(title)
        plt.show(block=False)


if __name__ == "__main__":
    sample_rate = 16000
    dataset = UrbanSoundPrep("data")
    df = dataset.metadata

    labels = dataset.class_mapping
    print(labels)

    sample_idx = 10
    for label in labels.keys():
        index = df.index[df['class'] == label].tolist()
        waveform, sample_rate, label = dataset.get_processed_waveform(index[sample_idx])
        mel_spectro = dataset.calc_mel_spec(waveform)
        mfcc = dataset.calc_mfcc(waveform)

        dataset.plot_spectrogram(mfcc[0])
        plt.show()