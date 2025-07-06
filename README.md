💬 HappyMate: NLP Mental Health & Emotion Detection
This project implements two Natural Language Processing (NLP) classifiers for:
✅ Stress detection — predict if text indicates stress
✅ Emotion detection — classify text into emotional categories (e.g. happy, sad, angry)

Both models were trained using supervised learning on labeled datasets, achieving strong accuracy.

🚀 Project Highlights

✅ Stress Detection
Algorithm: Feed-forward neural network (PyTorch)
Vectorization: Sentence embeddings using sentence-transformers
Dataset: Kaggle StressScan (NLP stress detection text dataset)
Accuracy: 🌟 92% test accuracy

✅ Emotion Detection
Algorithm: Feed-forward neural network (PyTorch)
Vectorization: Sentence embeddings using sentence-transformers
Dataset: Hugging Face dair-ai/emotion unsplit dataset
Accuracy: 🌟 77% test accuracy

📂 Model Pipeline
✅ Data pipeline for both models:
1️⃣ Input: Raw text samples with labels
2️⃣ Embedding: Convert text to dense vectors using sentence-transformers (all-MiniLM-L12-v2)
3️⃣ Classifier: A PyTorch neural network
4️⃣ Training:
Train-validation-test split (90-5-5 or similar)
Early stopping or fixed epochs (e.g. 10–20)
Accuracy monitored on validation + test sets

5️⃣ Output:
Saved .pt model for future inference
Label encoder mapping for inverse transform

🎯 Results
Task	Accuracy	Notes
Stress detection	92%	Binary classification (stress / no_stress)
Emotion detection	77%	Multi-class (e.g. happy, sad, angry, love)

✅ These results were achieved on held-out test sets.

📝 Model Details
Component	Stress / Emotion model
Input	Sentence embeddings (384-d)
Layers	3-5 linear layers
Activation	ReLU, GELU
Regularization	Dropout (0.2-0.3)
Epochs	Tuned for stability (30 typical)

💡 Usage
👉 The models can be loaded and used for prediction:

📌 Future Extensions
Fine-tuning on custom mental health datasets
Add attention or RNN layers for sequential patterns
Integrate into chatbot or web app

🤖 Credits
Stress dataset: StressScan on Kaggle
Emotion dataset: dair-ai/emotion
(details in reference txt file)

Libraries: PyTorch, sentence-transformers, scikit-learn
