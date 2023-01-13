import torch
from train import feed_forward_net, download_mnist


CLASS_MAPPING = [str(x) for x in range(10)]

def predict(model, input, target, class_mapping):
    model.eval()
    
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [[.1, .1, ..., .8]]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected
    

if __name__ == "__main__":
    # load model
    ff_net = feed_forward_net()
    state_dict = torch.load("Data\\ff_net.pth")
    # load dataset
    _ , validation_data = download_mnist()

    input, target = validation_data[0][0], validation_data[0][1]


    predicted, expected = predict(ff_net, input, target, CLASS_MAPPING)
    print(f"Predicted: {predicted}\nExpected: {expected}")