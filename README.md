# Решение задачи бинарной классификации

**Структура проекта**

- папка `solution` - весь код, используемый для решения задачи

   - `main.py` - запуск обучения лучшей модели
   - Папка `best_model` содержит .py код для обучения лучшей модели. Запуск осуществляется через main.py.
   - Папка `notebooks` содержит два подробных исследования и применение моделей машинного и глубокого обучения
      - `DL_based_approach` - обучена полносвязная нейронная сеть с подбором оптимизатороов и learning rate
      - `ML_based_approach` - исследование различных методов машинного обучения

**Результаты**

Наилучший результат показала полносвязная нейронная сеть с оптимизатором `RMSProp` и `lr=3e-2`. При помощи этой модели был сформирован файл `predictions.csv`

Пример запуска:

```
python main.py train.csv test.csv target_dir
```
Сводная таблица:

| Оптимизатор | lr   | f1-macro | ROC curve area |
|-------------|------|----------|----------------|
| RMSProp     | 3e-2 | 0.993    | 0.99           |
| Adam        | 4e-2 | 0.99     | 0.99           |
| SGD         | 4e-2 | 0.488    | 0.51           |

Планировалось провести больше экспериментов и посмотреть на другие архитектуры, но оказалось, что полносвязная нейронная сеть, взятая в качестве бейзлайна, справляется с задачей на хорошем уровне.

- `ML_based_approach` - обучены следующие алгоритмы:
  - Decision Trees
  - Random Forest
  - XGBoost
  - Support Vector Machine
  
В рамках исследования проведен анализ обучающей выборки, настройка гиперпараметров каждой из модели, оценка качества с использованием f1-macro и Roc-Auc, а также применены эмбеддинги word2vec и Fastext с PCA.

Среди методов машинного обучения наилучший результат показал классификатор SVM с векторизацией Tf-Idf. Более подробный анализ предоставлен в ноутбуке.

Сводная таблица:

| Модель                          | f1-macro | ROC AUC  | Лучшие гиперпараметры                                                                                                    |
|---------------------------------|----------|----------|--------------------------------------------------------------------------------------------------------------------------|
| Decision Trees (DT)             | 0.51     | 0.59     | 'dt__criterion': 'gini', 'dt__max_depth': 20, 'dt__min_samples_leaf': 1, 'dt__min_samples_split': 5                      |
| Random Forest (RF)              | 0.65     | 0.79     | 'rf__max_depth': 10, 'rf__min_samples_leaf': 5, 'rf__n_estimators': 300                                                  |
| XGBoost                         | 0.68     | 0.75     | 'xgb__gamma': 0.1, 'xgb__learning_rate': 0.1, 'xgb__max_depth': 10, 'xgb__min_child_weight': 3, 'xgb__n_estimators': 300 |
| **Support Vector Machine**      | **0.82** | **0.89** | 'svc__C': 1, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'                                                                 |
| SVM + Word2Vec                  | 0.52     | 0.54     | 'svc__C': 10, 'svc__gamma': 'scale', 'svc__kernel': 'linear'                                                             |
| SVM + pretrained FastText + PCA | 0.63     | 0.67     | 'svc__C': 10, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'                                                                |
| RF + pretrained FastText + PCA  | 0.63     | 0.69     | 'rf__max_depth': 10, 'rf__min_samples_leaf': 10, 'rf__n_estimators': 100                                                 |
