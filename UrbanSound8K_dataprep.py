import pandas as pd
import os
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt


class UrbanSoundPrep:

    def __init__(self,
                 data_path,
                 resample_rate=22050,
                 number_of_samples=22050,
                 n_fft=1024,
                 hop_length=512,
                 mfcc_num=64):

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
        self.mfcc_num = mfcc_num

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

    # def calc_stft(self, waveform, sr):
    def calc_mfcc(self, waveform, sr):

        # define transformation
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.n_fft,
            win_length=None,
            hop_length=self.hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=self.mfcc_num,
            mel_scale="htk",
        )
        # Perform transformation
        mfcc = mel_spectrogram(waveform)

        return mfcc

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

    def plot_spectrogram(self, waveform, sample_rate, label):

        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=sample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c + 1}")
        key = [k for k, v in self.class_mapping.items() if v == label]
        figure.suptitle(key)


if __name__ == "__main__":
    sample_rate = 16000
    dataset = UrbanSoundPrep("data")
