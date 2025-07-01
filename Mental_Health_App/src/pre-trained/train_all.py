import torch
import joblib
from sentence_transformers import SentenceTransformer

from EmotionAnalyserTraining import EmotionAnalyserTrainer
from mentality_training import MentalityAnalyserTrainer

def test_emotion_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    print("\n🧪 Running test on Emotion model...")

    # Load model and preprocessing tools
    model = torch.jit.load("models/emotion.pt", map_location=device)
    model.eval()

    encoder = SentenceTransformer("all-MiniLM-L12-v2")
    label_encoder = joblib.load("models/emotion_labels.pkl")

    while True:
        test_input = input("Enter a sentence: ")
        input_vec = encoder.encode([test_input])
        input_tensor = torch.tensor(input_vec, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_label = label_encoder.inverse_transform([pred_idx])[0]

        print(f"Predicted emotion: {pred_label}")
        continue_input = input("Continue? (y?): ")
        if continue_input != "y":
            break


def test_mentality_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    print("\n🧪 Running test on Mentality model...")

    # Load model and preprocessing tools
    model = torch.jit.load("models/mentality.pt", map_location=device)
    model.eval()

    encoder = SentenceTransformer("all-MiniLM-L12-v2")
    label_encoder = joblib.load("models/mentality_labels.pkl")

    while True:
        test_input = input("Enter a sentence: ")
        input_vec = encoder.encode([test_input])
        input_tensor = torch.tensor(input_vec, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_label = label_encoder.inverse_transform([pred_idx])[0]

        print(f"Do the person feel stressed?: {pred_label}")
        continue_input = input("Continue? (y?): ")
        if continue_input != "y":
            break
def main():
    trainers = [
        #("models/emotion.pt", EmotionAnalyserTrainer()),
        # ("models/habit.pt", HabitAnalyserTrainer()),
        # ("models/health.pt", HealthAnalyserTrainer()),
        # ("models/leisure.pt", LeisureAnalyserTrainer()),
        ("models/mentality.pt", MentalityAnalyserTrainer())
        # ("models/productivity.pt", ProductivityAnalyserTrainer())
    ]

    for path, trainer in trainers:
        print(f"🔁 Training {path}...")
        trainer.run_pipeline(path)
        print(f"🔁 Training {path} finished.")

if __name__ == "__main__":
    #main()
    #test_emotion_model()
    test_mentality_model()