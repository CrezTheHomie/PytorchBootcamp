import os
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch

class UrbanSoundDataset(Dataset):

    def __init__(self,
    annotations_file: str,
    audio_dir: str,
    transformation: str,
    target_sr: int,
    num_samples: int,
    device: str) -> None:
        super().__init__()
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sr
        self.num_samples = num_samples


    def __len__(self) -> int:
        return len(self.annotations)

    def _get_audio_sample_path(self, index: int) -> str:
        fold = f"fold{self.annotations.iloc[index, 5]}"
        audio_file_name = self.annotations.iloc[index, 0]
        path = os.path.join(self.audio_dir, fold, audio_file_name)
        return path

    def _get_audio_sample_label(self, index: int) -> int:
        return self.annotations.iloc[index, 6]

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _resample_if_necessary(self, signal, sr: int):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _resize_if_necessary(self, signal):
        # signal -> Tensor (channels, num samples)
        # cut if too large
        if signal.shape[1] > self.num_samples:
            signal = signal[:, self.num_samples]
        # pad if too small
        elif signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            signal = torch.nn.functional.pad(signal, (0, num_missing_samples))

        return signal

    def __getitem__(self, index: int):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._resize_if_necessary(signal)

        signal = self.transformation(signal)

        return signal, label


if __name__ == "__main__":
    ANNOTATIONS_FILE = "D:\\Code\PytorchBootcamp\\PytorchBootcamp\\data\\UrbanSound8K\\metadata\\UrbanSound8K.csv"
    AUDIO_DIR = "D:\\Code\\PytorchBootcamp\\PytorchBootcamp\\data\\UrbanSound8K\\audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"We are using a {device} device")
    print(f"Device is a {torch.cuda.get_device_name()}")

    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft= 1024,
        hop_length=512,
        n_mels=64
    )

    USD = UrbanSoundDataset(annotations_file=ANNOTATIONS_FILE,
    audio_dir=AUDIO_DIR,
    transformation=mel_spectogram,
    target_sr=SAMPLE_RATE,
    num_samples=NUM_SAMPLES,
    device=device)
    print(f"There are {len(USD)} samples in dataset")

    signal, label = USD[0]

    print(f"We're done here.")