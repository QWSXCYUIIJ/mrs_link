import telebot
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Задайте ваш токен, полученный у BotFather
API_TOKEN = '7916627787:AAHgUpjoIupZfs9jwTkUzak79AX0mC4-jwk'
bot = telebot.TeleBot(API_TOKEN)

# Загрузка модели
model = load_model('my_model.h5')

# Загрузка токенизатора
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Загрузка max_length
with open('max_length.json', 'r') as f:
    max_length = json.load(f)['max_length']

# Соответствие меток
label_mapping = {0: 'Деньги', 1: 'Развитие', 2: 'Личное'}

# Обработка входящих сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_input = message.text
    print(f'User Input: {user_input}')  # Отладочный вывод
    
    # Токенизация и паддинг
    test_sequence = tokenizer.texts_to_sequences([user_input])
    print(f'Tokenized Sequence: {test_sequence}')  # Отладочный вывод
    test_sequence = pad_sequences(test_sequence, maxlen=max_length)
    print(f'Padded Sequence: {test_sequence}')  # Отладочный вывод
    
    # Получение предсказаний
    predictions = model.predict(test_sequence)
    print(f'Predictions: {predictions}')  # Отладочный вывод
    predicted_label = np.argmax(predictions, axis=1)[0]
    
    # Отправляем результат пользователю
    response = f'Общее между этими словами -> "{label_mapping[predicted_label]}"'
    bot.reply_to(message, response)

# Запуск бота
if __name__ == '__main__':
    bot.polling(none_stop=True)
