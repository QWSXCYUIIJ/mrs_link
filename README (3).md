

# **Задача:** 
Система анализа списка пользовательских ответов
# **Команда:** 
***МИР*** <br>
Участники команды:<br>
Малафеев Максим<br> Исмаилова Карина<br> Ищенко Тимофей<br>
________

## **Структура**
Файл ai1.py - код обучения нейронки <br>
файл bot.py - код бота, которому мы запускаем файлы обучения и через личный API-токен запускаем бота в Telegram

## **Обучение Нейронки и Заметки**

###Преобразуем все слова в последовательность
```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['words'])
X = tokenizer.texts_to_sequences(df['words'])
```
###Паддинг последовательностей
```python
max_length = max(len(seq) for seq in X)
X = pad_sequences(X, maxlen=max_length)
```
###Преобразование меток в числовые значения
```python
labels = pd.get_dummies(df['label']).values
```
###Разделение данных на обучающую и тестовую выборки
где X_train и X_test - обучающие и тестовые выборки на входе, а с Y - выборки на выходе 
```python
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
```
###Создание модели нейронной сети
```python
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=max_length))
#model.add(Dropout(0.5))
model.add(LSTM(units=32, return_sequences=True))
model.add(Dropout(0.5))
model.add(GlobalMaxPooling1D())
model.add(Dense(units=16, activation='relu'))

model.add(Dense(units=3, activation='softmax'))
```
###Компиляция модели
```python
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',  # Для меток в формате one-hot
              metrics=['accuracy'])

```
###Обучение и оценка

```python
model.fit(X_train, y_train, epochs=100, batch_size=2, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

```
###Загрузка модели в бота
необходимо получить токен, полученный у BotFather и загрузить токенизатор с моделью, после пользователю выводится результат [X] - обобщающий синоним наиболее частого типа ответа.
[MTS_link_bot](https://t.me/mts_hackathon_bot)

Наш бот выдает верные решения с точностью около 70%, что для нейронной системы достаточно стабильный и средний результат [X]
