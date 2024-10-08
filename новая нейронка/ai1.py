# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FrrdEWIujckwabTCJyrCiMX-FpN_prZL
"""
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

data = {
    "words": [
        # Слова, относящиеся к деньгам
        "деньги наличные",
        "зарплата премия",
        "кредит ссуда",
        "монета банкнота",
        "сбережения капитал",
        "платеж перевод",
        "бюджет доход",
        "инвестиции депозит",
        "ипотека займ",
        "акции облигации",
        "дивиденды прибыль",
        "кошелек банк",
        "комиссия процент",
        "налог пошлина",
        "баланс аванс",
        "аренда ставка",
        "счет карта",
        "бонус премия",
        "долг задолженность",
        "контракт сделка",
        "кредиты ипотеки",
        "вклад депозит",
        "бизнес доходы",
        "обмен валюта",
        "фонд акции",
        "финансы активы",
        "контроль траты",
        "покупка продажа",
        "транзакция перевод",
        "деньги ликвидация",
        "спекуляция трейдинг",
        "наличные активы",
        "платеж чек",
        "касса банкомат",
        "облигация заём",
        "дебет кредит",
        "ипотека процент",
        "курс инфляция",
        "девальвация дефолт",
        "рефинансирование займ",
        "микрокредитование кредитование",
        "хранение капитал",
        "наличность оборот",
        "ликвидность активы",
        "оценка инвестиций",
        "долгосрочный займ",
        "банковский перевод",
        "валютный рынок",
        "финансовый кризис",
        "торговля активами",
        "страховка полис",

        # Слова, относящиеся к развитию
        "личностный рост",
        "обучение навыки",
        "образование тренинг",
        "саморазвитие мотивация",
        "успех карьера",
        "прогресс улучшение",
        "цели достижения",
        "новые знания",
        "развитие лидерства",
        "планирование стратегии",
        "командная работа",
        "рост профессионализма",
        "поддержка наставничество",
        "обратная связь",
        "развитие проектов",
        "вовлеченность сотрудников",
        "постановка целей",
        "коучинг тренинг",
        "развитие коммуникации",
        "управление временем",
        "совершенствование навыков",
        "инновации прогресс",
        "обучение сотрудников",
        "постоянное улучшение",
        "стратегическое мышление",
        "развитие интеллекта",
        "развитие бизнеса",
        "технологические изменения",
        "эффективность работы",
        "мотивация команда",
        "коммуникативные навыки",
        "критическое мышление",
        "развитие предпринимательства",
        "менторство поддержка",
        "внедрение инноваций",
        "трансформация процесса",
        "творческое мышление",
        "обучение программированию",
        "изучение языков",
        "возможности развития",
        "управление проектами",
        "лидерство наставничество",
        "совершенствование процессов",
        "рост компании",
        "личная эффективность",
        "развитие гибкости",
        "развитие карьерных возможностей",
        "освоение новых технологий",
        "разработка стратегии",
        "развитие командных навыков",

        # Слова, относящиеся к личному
        "семья дети",
        "дружба отношения",
        "здоровье спорт",
        "хобби увлечения",
        "отдых отпуск",
        "личная жизнь",
        "свадьба любовь",
        "семейные ценности",
        "личное пространство",
        "дом уют",
        "домашние дела",
        "воспитание детей",
        "родительские обязанности",
        "личное время",
        "эмоции чувства",
        "личные цели",
        "психологическое здоровье",
        "собственные интересы",
        "самоосознание",
        "социальные связи",
        "отношения с близкими",
        "личные границы",
        "отношения с партнером",
        "забота о себе",
        "внутренний мир",
        "самовыражение",
        "самореализация",
        "размышления",
        "взаимоотношения",
        "личные достижения",
        "привычки распорядок",
        "праздники традиции",
        "личные переживания",
        "саморазмышления",
        "воспоминания эмоции",
        "цели в жизни",
        "свободное время",
        "личная мотивация",
        "внутренняя гармония",
        "вдохновение творчество",
        "родственные связи",
        "личные приоритеты",
        "чувства эмоции",
        "личные взгляды",
        "социальное взаимодействие",
        "личные решения",
        "воспитание традиции",
        "личные интересы",
        "отношения в семье",
        "личные впечатления",
        "эмоциональное состояние"
    ],
    "label": [
        # Метки для слов, относящихся к деньгам (0)
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0,

        # Метки для слов, относящихся к развитию (1)
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1,

        # Метки для слов, относящихся к личному (2)
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2
    ]
}



print(len(data["words"]), len(data["label"]))

# Преобразование данных в DataFrame
df = pd.DataFrame(data)

# Шаг 1: Токенизация и преобразование слов в последовательности
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['words'])
X = tokenizer.texts_to_sequences(df['words'])

# Шаг 2: Паддинг последовательностей
max_length = max(len(seq) for seq in X)
X = pad_sequences(X, maxlen=max_length)

# Шаг 3: Преобразование меток в числовые значения
labels = pd.get_dummies(df['label']).values

# Шаг 4: Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Шаг 5: Создание модели нейронной сети
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=max_length))
#model.add(Dropout(0.5))
model.add(LSTM(units=32, return_sequences=True))
model.add(Dropout(0.5))
model.add(GlobalMaxPooling1D())
model.add(Dense(units=16, activation='relu'))

model.add(Dense(units=3, activation='softmax'))

# Шаг 6: Компиляция модели
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',  # Для меток в формате one-hot
              metrics=['accuracy'])

# Шаг 7: Обучение модели
model.fit(X_train, y_train, epochs=100, batch_size=2, validation_data=(X_test, y_test))

# Шаг 8: Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Пример предсказания
test_data = ["залупа бляди", "деньги"]
test_sequences = tokenizer.texts_to_sequences(test_data)
test_sequences = pad_sequences(test_sequences, maxlen=max_length)

predictions = model.predict(test_sequences)
predicted_labels = np.argmax(predictions, axis=1)

# Перевод предсказанных индексов в метки
label_mapping = {i: label for i, label in enumerate(df['label'].unique())}
for i, pred in enumerate(predicted_labels):
    print(f'Words: "{test_data[i]}" -> Predicted label: "{label_mapping[pred]}"')

# Сохранение модели
model.save('my_model.h5')

# Сохранение токенизатора
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('max_length.json', 'w') as f:
    json.dump({'max_length': max_length}, f)

