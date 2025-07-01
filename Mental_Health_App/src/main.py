import os
from dotenv import load_dotenv
from chatbot import LocalChatBot

def main():
    bot = LocalChatBot()
    bot.chat_loop()
    #bot.show_memory()
    bot.close()

if __name__ == "__main__":
    main()
