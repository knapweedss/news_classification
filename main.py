import argparse
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from best_model.RMSProp import load_and_vectorize_data, FC_NeuralNetwork, train_FC


def main(train_data_path, test_data_path, model_save_path):
    model = FC_NeuralNetwork()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.RMSprop
    n_epochs = 10
    batch_size = 32

    # Загрузка и векторизация данных
    X_train, X_val, y_train, y_val = load_and_vectorize_data(train_data_path)

    # Обучение модели
    model_RMSProp, loss_RMSProp = train_FC(model, loss_fn, optimizer, n_epochs, batch_size, 3e-2, X_train, y_train, X_val, y_val)

    # Сохранение модели
    torch.save(model_RMSProp.state_dict(), model_save_path)

    # Предсказание
    df_test = pd.read_csv(test_data_path, sep='\t')
    tfidf = TfidfVectorizer(max_features=5000)
    X_test_tfidf = tfidf.fit_transform(df_test['title']).toarray()

    X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)

    model_RMSProp.eval()
    with torch.no_grad():
        predictions = model_RMSProp(X_test_tensor)

    predicted_probs = predictions
    predicted_labels = (predicted_probs > 0.5).int()

    # Формирование predictions.csv
    df_test['prob_fake'] = predicted_probs.numpy()
    df_test['is_fake'] = predicted_labels.numpy()
    df_test.to_csv('predictions.csv', sep='\t', encoding='utf-8', index=False, columns=['title', 'prob_fake', 'is_fake'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Обучение лучшей модели")
    parser.add_argument('train_data_path', type=str, help="Путь к CSV файлу с обучающими данными")
    parser.add_argument('test_data_path', type=str, help="Путь к CSV файлу с данными для предсказаний")
    parser.add_argument('model_save_path', type=str, help="Путь, куда сохранить обученную модель")

    args = parser.parse_args()
    main(args.train_data_path, args.test_data_path, args.model_save_path)