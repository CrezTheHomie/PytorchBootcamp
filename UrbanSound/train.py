import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from UrbanSound_Data import UrbanSoundDataset
from UrbanSound_CNN import UrbanSoundCNN
from GPUtil import showUtilization as gpu_usage


BATCH_SIZE = 512

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}\n")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print(f"Train\n" + torch.cuda.memory_summary(device="cuda", abbreviated=False))
    print(f"Training has finished")


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):

    running_loss = 0
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        print(f"Train 1\n")
        gpu_usage()
        predictions = model(input)
        loss = loss_fn(predictions, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Running Loss: {running_loss}")


if __name__ == "__main__":
    ANNOTATIONS_FILE = "D:\\Code\PytorchBootcamp\\PytorchBootcamp\\data\\UrbanSound8K\\metadata\\UrbanSound8K.csv"
    AUDIO_DIR = "D:\\Code\\PytorchBootcamp\\PytorchBootcamp\\data\\UrbanSound8K\\audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # device = "cpu"
    
    print(f"We are using a {device} device")
    print(f"Device is a {torch.cuda.get_device_name()}")
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary(device=device, abbreviated=False))

    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft= 1024,
        hop_length=512,
        n_mels=64
    )

    train_data = UrbanSoundDataset(annotations_file=ANNOTATIONS_FILE,
    audio_dir=AUDIO_DIR,
    transformation=mel_spectogram,
    target_sr=SAMPLE_RATE,
    num_samples=NUM_SAMPLES,
    device=device)
    print(f"Train data downloaded.")

    # create a data loader for train set
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    cnn = UrbanSoundCNN().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=.001)


    train(cnn, train_data_loader, loss_fn=loss_fn, optimizer=optimizer, device=device, epochs=10)

    torch.save(cnn.state_dict(), "cnn.pth")
    print(f"Model trained and stored at cnn.pth")