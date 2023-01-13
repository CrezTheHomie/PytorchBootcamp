import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128

class feed_forward_net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            # input size (flattened, neuron outputs) Linear = Dense layer
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flat_data = self.flatten(input_data)
        logits = self.dense_layers(flat_data)
        predictions = self.softmax(logits)
        return predictions

def download_mnist():
    mnist_train_data = datasets.MNIST(root="MNIST_data",
                                download=True,
                                train=True,
                                transform=ToTensor()
                                )

    mnist_val_data = datasets.MNIST(root="MNIST_data",
                                download=True,
                                train=False,
                                transform=ToTensor()
                                )
    return mnist_train_data, mnist_val_data

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}\n")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
    print(f"Training has finished")


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        predictions = model(input)
        loss = loss_fn(predictions, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    train_data, _ = download_mnist()
    print(f"Train data downloaded.")

    # create a data loader for train set
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"using {device} device")
    ff_net = feed_forward_net().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ff_net.parameters(), lr=.001)


    train(ff_net, train_data_loader, loss_fn=loss_fn, optimizer=optimizer, device=device, epochs=10)

    torch.save(ff_net.state_dict(), "data\\ff_net.pth")
    print(f"Model trained and stored at ff_net.pth")