import torch
import torchaudio
from UrbanSound_Data import UrbanSoundDataset, ANNOTATIONS_FILE, AUDIO_DIR, NUM_SAMPLES, SAMPLE_RATE
from UrbanSound_CNN import UrbanSoundCNN

CLASS_MAPPING = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "Gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

def predict(model, input, target, class_mapping):
    model.eval()
    
    with torch.no_grad():
        predictions = model(input)
        print(f"Predictions: {predictions}")
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected
    

if __name__ == "__main__":
    # load model
    cnn = UrbanSoundCNN()
    state_dict = torch.load("cnn.pth")
    cnn.load_state_dict(state_dict)
    
    # load dataset
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft= 1024,
        hop_length=512,
        n_mels=64
    )

    usd_data = UrbanSoundDataset(annotations_file=ANNOTATIONS_FILE,
    audio_dir=AUDIO_DIR,
    transformation=mel_spectogram,
    target_sr=SAMPLE_RATE,
    num_samples=NUM_SAMPLES,
    device="cpu")
    

    
    input, target = usd_data[0][0], usd_data[0][1]  
    input.unsqueeze_(0)

    predicted, expected = predict(cnn, input, target, CLASS_MAPPING)
    print(f"Predicted: {predicted}\nExpected: {expected}")