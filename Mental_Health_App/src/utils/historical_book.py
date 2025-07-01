import sqlite3
from pathlib import Path
from datetime import datetime

class ChatBotMemory:
    def __init__(self, db_filename="chat_memory.db"):
        # Ensure relative path in current directory
        db_path = Path(__file__).parent / db_filename
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    name TEXT
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_query TEXT,
                    bot_response TEXT,
                    emotion_index INTEGER,
                    mental_index INTEGER,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
            """)

    def start_conversation(self, name):
        conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
        with self.conn:
            self.conn.execute(
                "INSERT INTO conversations (conversation_id, name) VALUES (?, ?)",
                (conversation_id, name)
            )
        return conversation_id

    def save_exchange(self, conversation_id, user_query, bot_response, emotion_index, mental_index):
        with self.conn:
            self.conn.execute(
                """INSERT INTO messages 
                   (conversation_id, user_query, bot_response, emotion_index, mental_index)
                   VALUES (?, ?, ?, ?, ?)""",
                (conversation_id, user_query, bot_response, emotion_index, mental_index)
            )

    def fetch_conversation(self, conversation_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT timestamp, user_query, bot_response, emotion_index, mental_index
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
        """, (conversation_id,))
        return cursor.fetchall()

    def fetch_all_conversations(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM conversations")
        return cursor.fetchall()

    def close(self):
        self.conn.close()
