import torch
import joblib
from sentence_transformers import SentenceTransformer

from EmotionAnalyserTraining import EmotionAnalyserTrainer

def test_emotion_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    print("\n🧪 Running test on Emotion model...")

    # Load model and preprocessing tools
    model = torch.jit.load("Models/emotion.pt", map_location=device)
    model.eval()

    encoder = SentenceTransformer("all-MiniLM-L12-v2")
    label_encoder = joblib.load("Models/emotion_labels.pkl")

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
def main():
    trainers = [
        ("Models/emotion.pt", EmotionAnalyserTrainer())
        # ("Models/habit.pt", HabitAnalyserTrainer()),
        # ("Models/health.pt", HealthAnalyserTrainer()),
        # ("Models/leisure.pt", LeisureAnalyserTrainer()),
        # ("Models/mentality.pt", MentalityAnalyserTrainer()),
        # ("Models/productivity.pt", ProductivityAnalyserTrainer())
    ]

    for path, trainer in trainers:
        print(f"🔁 Training {path}...")
        trainer.run_pipeline(path)
        print(f"🔁 Training {path} finished.")

if __name__ == "__main__":
    # main()
    #
    test_emotion_model()

"""
Sad:
I woke up with that heavy kind of silence that follows a long cry. The air in my room felt like a weight, and my heart dragged behind me like a chain. I moved through the day like a ghost, surrounded by people yet entirely invisible. Nothing hurt in a sharp way—it was more like everything was dulled, as if the colors of the world had faded into grayscale. I didn’t want to talk, not because I was angry, but because I didn’t think my voice could hold itself together long enough to form words.

Happy:
I danced around the kitchen while making breakfast, half-singing, half-laughing at how silly I must’ve looked. But it didn’t matter. There was no audience, no performance—just joy for the sake of it. Even my coffee tasted better, like it knew I was in a good mood and wanted to contribute. Little things kept adding up like confetti: a kind message from a friend, a sunny window seat, a perfect song at the perfect time.

Love:
I’ve loved before, but never like this. This feels like building a home inside someone’s soul, and watching them do the same in mine. I want to grow with them, not just through the good—but through the messy, raw, real parts of life. And somehow, loving them makes me love myself a little more too.

Angry:
I am not just irritated—I am simmering. Beneath every polite nod and forced smile is a scream waiting to be let out. I replay the conversation in my head, over and over, each time catching a new moment where I should’ve said something, where I should’ve stood up, where I let it slide. I hate that I kept quiet. I hate that they think I’m okay with it.


"""
