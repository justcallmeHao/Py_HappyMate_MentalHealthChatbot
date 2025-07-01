from historical_book import ChatBotMemory

# Initialize DB
memory = ChatBotMemory()

# Start a new conversation
conv_id = memory.start_conversation("JohnChat")

# Save exchanges
memory.save_exchange(conv_id, "Hi", "Hello!", 0, 1)
memory.save_exchange(conv_id, "I feel sad", "I'm here for you.", 2, 3)

# Fetch & print conversation
for row in memory.fetch_conversation(conv_id):
    print(row)

memory.close()