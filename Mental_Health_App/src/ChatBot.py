import sqlite3
from pathlib import Path
from datetime import datetime
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LocalChatBot:
    def __init__(self, db_filename="chat_memory.db"):
        # Detect device
        # self.device = 0 if torch.cuda.is_available() else -1
        torch.device("cpu")
        # print(f"✅ Using device: {'cuda' if self.device == 0 else 'cpu'}")

        # # Load pipeline (defaults to GPT-2)
        # self.generator = pipeline("text-generation", model="microsoft/DialoGPT-small", device=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

        # Setup SQLite DB
        db_path = Path(__file__).parent / db_filename
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        self.chat_history_ids = None

    def _create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_query TEXT,
                    bot_response TEXT
                )
            """)

    def call_model(self, user_input):
        # Encode user input + add eos_token
        new_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,
            return_tensors='pt'
        )

        # Build bot input by appending to chat history
        if hasattr(self, 'chat_history_ids') and self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Generate bot response
        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode the generated text (only the new bot part)
        bot_reply = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        # Save to DB
        with self.conn:
            self.conn.execute(
                "INSERT INTO conversations (user_query, bot_response) VALUES (?, ?)",
                (user_input, bot_reply)
            )

        return bot_reply

    def chat_loop(self):
        print("ChatBot: Hello! Let's chat (type 'bye' to quit).")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() == "bye":
                print("ChatBot: Goodbye!")
                break

            bot_reply = self.call_model(user_input)
            print(f"ChatBot: {bot_reply}")

    def show_memory(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT timestamp, user_query, bot_response FROM conversations ORDER BY id")
        for row in cursor.fetchall():
            print(f"{row[0]} | You: {row[1]} | Bot: {row[2]}")

    def close(self):
        self.conn.close()
