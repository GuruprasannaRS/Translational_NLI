from preprocessing import datapreprocessing
from dataloader import Dataloader
from gensim.models import Word2Vec
import torch
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from model import CNN
from itertools import product
import csv
import warnings
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore', message='Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.', category=UndefinedMetricWarning)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix

filepath  = r'D:\UWaterloo\MSCI_641_Text_Analytics\Project\Data\fr_fever-00000-of-00001-b7ec330d6224f90b.parquet' 

data_preprocessing = datapreprocessing(filepath)

check_df = data_preprocessing.parquet_to_pd()
test_df = pd.read_csv(r'D:\UWaterloo\MSCI_641_Text_Analytics\Project\Output\pred_df.csv')
check_dfe, check_dff = data_preprocessing.eng_fre_dataset(check_df)
# check_dfe, xtre, xve, xtee, ytre, yve, ytee = data_preprocessing.english_pipeline(check_dfe)
check_dff, xtrf, xvf, xtef, ytrf, yvf, ytef = data_preprocessing.french_pipeline(check_dff,test_df)
print(len(xtef))
print(len(ytef))

# embedding_model_e = Word2Vec.load("w2ve.model")
# special_tokens = [["<SOS>","<EOS>","<PAD>","<UNK>"] * 15 ] 
# embedding_model_e.build_vocab(special_tokens, update=True)
# embedding_vectors_e = embedding_model_e.wv
# embedding_matrix_e = embedding_model_e.wv.vectors
# vocab_size_e, embedded_dim_e = embedding_matrix_e.shape

embedding_model_f = Word2Vec.load("w2vf.model")
special_tokens = [["<SOS>","<EOS>","<PAD>","<UNK>"] * 15 ] 
embedding_model_f.build_vocab(special_tokens, update=True)
embedding_vectors_f = embedding_model_f.wv
embedding_matrix_f = embedding_model_f.wv.vectors
vocab_size_f, embedded_dim_f = embedding_matrix_f.shape

max_sent_length = 50

# premise_name_e = 'premise_original'
# hypo_name_e = 'hypothesis_original'
premise_name_f = 'premise'
hypo_name_f = 'hypothesis'



# dataloader_e = Dataloader(embedding_vectors_e, max_sent_length, xtre, ytre, xve, yve, xtee, ytee, premise_name_e,hypo_name_e)

# train_loader_e = dataloader_e.get_train_loader()
# val_loader_e = dataloader_e.get_val_loader()
# test_loader_e = dataloader_e.get_test_loader()
# features_train_e = []
# labels_train_e = []
# for batch_features, batch_labels in train_loader_e:
#     features_train_e.append(batch_features)
#     labels_train_e.append(batch_labels)
# features_val_e = []
# labels_val_e = []
# for batch_features, batch_labels in val_loader_e:
#     features_val_e.append(batch_features)
#     labels_val_e.append(batch_labels)
# features_test_e = []
# labels_test_e = []
# for batch_features, batch_labels in test_loader_e:
#     features_test_e.append(batch_features)
#     labels_test_e.append(batch_labels)
# features_tr_e = torch.cat(features_train_e, dim=0).numpy()
# labels_tr_e = torch.cat(labels_train_e, dim=0).numpy()
# features_val_e = torch.cat(features_val_e, dim=0).numpy()
# labels_val_e = torch.cat(labels_val_e, dim=0).numpy()
# features_te_e = torch.cat(features_test_e, dim=0).numpy()
# labels_te_e = torch.cat(labels_test_e, dim=0).numpy()



# # Model
# nb_classifier = MultinomialNB()
# features_tr_e, labels_tr_e = shuffle(features_tr_e, labels_tr_e)
# nb_classifier.fit(features_tr_e, labels_tr_e)
# #Calculate training accuracy
# train_pred = nb_classifier.predict(features_tr_e)
# train_accuracy_mnb_e = accuracy_score(labels_tr_e, train_pred)

# print('Training Accuracy:', np.round(train_accuracy_mnb_e*100, 3))

# param_grid = {
#     'alpha': [0.1, 0.5]
# }
# grid_search = GridSearchCV(nb_classifier, param_grid, scoring='accuracy', cv=5)
# grid_search.fit(features_tr_e, labels_tr_e)
# # Get the best SVM model with the optimal hyperparameters
# best_svm_model = grid_search.best_estimator_
# print("Best hyperparameters:", best_svm_model)
# best_svm_model.fit(features_tr_e, labels_tr_e)
# # Calculate validation accuracy
# val_pred = best_svm_model.predict(features_val_e)
# val_accuracy_mnb_e = accuracy_score(labels_val_e, val_pred)
# print('Validation Accuracy:', val_accuracy_mnb_e)

# test_pred = nb_classifier.predict(features_te_e)
# test_accuracy_mnb_e = accuracy_score(labels_te_e, test_pred)
# print('Test Accuracy:', test_accuracy_mnb_e)

# precision_mnb_e = precision_score(labels_te_e, test_pred, average='macro')
# print('Precision English',precision_mnb_e)
# recall_mnb_e = recall_score(labels_te_e, test_pred, average='macro')
# print('Recall English',recall_mnb_e)
# f1_mnb_e = f1_score(labels_te_e, test_pred, average='macro')
# print('F1 score English',f1_mnb_e)
# kappa_mnb_e = cohen_kappa_score(labels_te_e, test_pred)
# print('Cohen kappa score English',kappa_mnb_e)

# report = classification_report(labels_te_e, test_pred)
# print("Classification Report:")
# print(report)
# # Compute the confusion matrix
# confusion_mat = confusion_matrix(labels_te_e, test_pred)
# confusion_matrix_mnb_e = confusion_mat.tolist()
# print("Confusion Matrix:")
# print(confusion_mat)


dataloader_f = Dataloader(embedding_vectors_f, max_sent_length, xtrf, ytrf, xvf, yvf, xtef, ytef, premise_name_f,hypo_name_f)
train_loader_f = dataloader_f.get_train_loader()
val_loader_f = dataloader_f.get_val_loader()
test_loader_f = dataloader_f.get_test_loader()
features_train_f = []
labels_train_f = []
for batch_features, batch_labels in train_loader_f:
    features_train_f.append(batch_features)
    labels_train_f.append(batch_labels)
features_val_f = []
labels_val_f = []
for batch_features, batch_labels in val_loader_f:
    features_val_f.append(batch_features)
    labels_val_f.append(batch_labels)
features_test_f = []
labels_test_f = []
for batch_features, batch_labels in test_loader_f:
    features_test_f.append(batch_features)
    labels_test_f.append(batch_labels)
print(len(features_test_f))
print(len(labels_test_f))
features_tr_f = torch.cat(features_train_f, dim=0).numpy()
labels_tr_f = torch.cat(labels_train_f, dim=0).numpy()
features_val_f = torch.cat(features_val_f, dim=0).numpy()
labels_val_f = torch.cat(labels_val_f, dim=0).numpy()
features_te_f = torch.cat(features_test_f, dim=0).numpy()
labels_te_f = torch.cat(labels_test_f, dim=0).numpy()



# #Model
nb_classifier = MultinomialNB()
features_tr_f, labels_tr_f = shuffle(features_tr_f, labels_tr_f)
nb_classifier.fit(features_tr_f, labels_tr_f)

train_pred = nb_classifier.predict(features_tr_f)
train_accuracy_mnb_f = accuracy_score(labels_tr_f, train_pred)

print('Training Accuracy:', np.round(train_accuracy_mnb_f*100, 3))
param_grid = {
    'alpha': [0.1, 0.5]
}
grid_search = GridSearchCV(nb_classifier, param_grid, scoring='accuracy', cv=5)
grid_search.fit(features_tr_f, labels_tr_f)
# # Get the best SVM model with the optimal hyperparameters
best_svm_model = grid_search.best_estimator_
print("Best hyperparameters:", best_svm_model)
best_svm_model.fit(features_tr_f, labels_tr_f)
# # Calculate validation accuracy
val_pred = best_svm_model.predict(features_val_f)
val_accuracy_mnb_f = accuracy_score(labels_val_f, val_pred)
print('Validation Accuracy:', val_accuracy_mnb_f)
test_pred = nb_classifier.predict(features_te_f)

test_accuracy_mnb_f = accuracy_score(labels_te_f, test_pred)
print('Test Accuracy:', test_accuracy_mnb_f)

label_original = labels_te_f.tolist()

output_file = r"D:\UWaterloo\MSCI_641_Text_Analytics\Project\Output\pred_labels_original.csv"

with open(output_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Original Label'])  # Write the header
    writer.writerows([[label] for label in label_original])  # Write each label as a list

print('Labels Exported Successfully')

test_pred = test_pred.tolist()  # Convert numpy array to a list of lists

output_file = r"D:\UWaterloo\MSCI_641_Text_Analytics\Project\Output\pred_labels_mnb.csv"

with open(output_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Predicted Label'])  # Write the header
    writer.writerows([[label] for label in test_pred])  # Write each label as a list

print('Labels Exported Successfully')

precision_mnb_f = precision_score(labels_te_f, test_pred, average='macro')
print('Precision French',precision_mnb_f)
recall_mnb_f = recall_score(labels_te_f, test_pred, average='macro')
print('Recall French',recall_mnb_f)
f1_mnb_f = f1_score(labels_te_f, test_pred, average='macro')
print('F1 score French',f1_mnb_f)
kappa_mnb_f = cohen_kappa_score(labels_te_f, test_pred)
print('Cohen kappa score French',kappa_mnb_f)


report = classification_report(labels_te_f, test_pred)
print("Classification Report:")
print(report)
# Compute the confusion matrix
confusion_mat = confusion_matrix(labels_te_f, test_pred)
confusion_matrix_mnb_f = confusion_mat.tolist()

print("Confusion Matrix:")
print(confusion_mat)



# CNN


def train_model(model, train_loader, criterion, optimizer, activation, dropout, name, lang):
    model.train()
    if name == 'Test Set': 
        total = 0
        correct = 0
        predicted_labels_list = []
        for data in train_loader:
            inputs, labels = data
            # print(len(inputs))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels_list.extend(predicted.tolist())
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
        # print(len(labels))
        # print(len(predicted))
        print(len(predicted_labels_list))
        # Name of the CSV file to store the predicted labels
        output_file = r"D:\UWaterloo\MSCI_641_Text_Analytics\Project\Output\pred_lables_cnn.csv"

        # Write the predicted labels to the CSV file
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Predicted Label'])
            for label in predicted_labels_list:
                writer.writerow([label])
        print('Labels Exported Successfully')

        accuracy = correct / total
        precision = precision_score(labels, predicted, average='macro')
        print('Precision French',precision)
        recall = recall_score(labels, predicted, average='macro')
        print('Recall French',recall)
        f1 = f1_score(labels, predicted, average='macro')
        print('F1 score French',f1)
        kappa = cohen_kappa_score(labels, predicted)
        print('Cohen kappa score French',kappa)
        report = classification_report(labels, predicted)
        print("Classification Report:")
        print(report)  

        print(f'Set: {name},Classification_lang: {lang}, Accuracy: {accuracy}, Activation Function: {activation}')
    else:
        for epoch in range(num_epochs):
            total = 0
            correct = 0
            for data in train_loader:
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.long()).sum().item()
            accuracy = correct / total
            precision = precision_score(labels, predicted, average='macro')
            # print('Precision French',precision)
            recall = recall_score(labels, predicted, average='macro')
            # print('Recall French',recall)
            f1 = f1_score(labels, predicted, average='macro')
            # print('F1 score French',f1)
            kappa = cohen_kappa_score(labels, predicted)
            # print('Cohen kappa score French',kappa)
            print(f'Set: {name}, Classification_lang: {lang}, Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy}, Activation Function: {activation}, Dropout: {dropout}')   
            

    return accuracy, precision, recall, f1, kappa

# english = 'English_classification'
french = 'French_classification'

train_name = 'Train Set'
validation_name = 'Validation Set'
test_name = 'Test Set'
output_size = 3
# hidden_size = 128
activation_function = [nn.ReLU(), nn.Tanh()]
dropout_rate = [0.2,0.2]
num_epochs = 5

hyperparameter_combo = product(activation_function, dropout_rate)
# best_results_e = {}
best_results_f = {}

for activation_function, dropout_rate in hyperparameter_combo:
    # model_e = CNN(vocab_size_e, embedded_dim_e, output_size, activation_function,dropout_rate)
    model_f = CNN(vocab_size_f, embedded_dim_f, output_size, activation_function,dropout_rate)
    loss_func = nn.CrossEntropyLoss()
    # optimizer_e = optim.Adam(model_e.parameters())
    optimizer_f = optim.Adam(model_f.parameters())
    # train_accuracy_e, _, _, _, _ = train_model(model_e, train_loader_e, loss_func, optimizer_e, activation_function, dropout_rate,train_name, english)
    # validation_accuracy_e, _, _, _, _  = train_model(model_e, val_loader_e, loss_func, optimizer_e, activation_function, dropout_rate,validation_name, english)

    train_accuracy_f, _, _, _, _  = train_model(model_f, train_loader_f, loss_func, optimizer_f, activation_function, dropout_rate,train_name, french)
    validation_accuracy_f, _, _, _, _  = train_model(model_f, val_loader_f, loss_func, optimizer_f, activation_function, dropout_rate,validation_name, french)

    activation_name = activation_function.__class__.__name__
    # if activation_name not in best_results_e:
    #     best_results_e[activation_name] = {'accuracy': validation_accuracy_e,'dropout_rate': dropout_rate}
    # else:
    #     if validation_accuracy_e > best_results_e[activation_name]['accuracy']:
    #         best_results_e[activation_name]['accuracy'] = validation_accuracy_e
    #         best_results_e[activation_name]['dropout_rate'] = dropout_rate

    if activation_name not in best_results_f:
        best_results_f[activation_name] = {'accuracy': validation_accuracy_f,'dropout_rate': dropout_rate}
    else:
        if validation_accuracy_f > best_results_f[activation_name]['accuracy']:
            best_results_f[activation_name]['accuracy'] = validation_accuracy_f
            best_results_f[activation_name]['dropout_rate'] = dropout_rate

# for activation_name, results in best_results_e.items():
#     print(f'Best Accuracy for English (Activation: {activation_name}): {results["accuracy"]}')
#     print(f'Best Dropout Rate for English (Activation: {activation_name}): {results["dropout_rate"]}')   

# best_activation_name_e = None
# best_dropout_rate_e = None
# highest_accuracy_e = 0.0

# for activation_name, results in best_results_e.items():
#     accuracy = results["accuracy"]
#     dropout_rate = results["dropout_rate"]

#     # Check if the current accuracy is better than the current highest_accuracy_e
#     if accuracy > highest_accuracy_e:
#         highest_accuracy_e = accuracy
#         best_activation_name_e = activation_name
#         best_dropout_rate_e = dropout_rate

# print(f'Best Accuracy for English (Activation: {best_activation_name_e}): {highest_accuracy_e}')
# print(f'Best Dropout Rate for English (Activation: {best_activation_name_e}): {best_dropout_rate_e}')

best_activation_name_f = None
best_dropout_rate_f = None
highest_accuracy_f = 0.0

for activation_name, results in best_results_f.items():
    accuracy = results["accuracy"]
    dropout_rate = results["dropout_rate"]

    # Check if the current accuracy is better than the current highest_accuracy_e
    if accuracy > highest_accuracy_f:
        highest_accuracy_f = accuracy
        best_activation_name_f = activation_name
        best_dropout_rate_f = dropout_rate

print(f'Best Accuracy for French (Activation: {best_activation_name_f}): {highest_accuracy_f}')
print(f'Best Dropout Rate for French (Activation: {best_activation_name_f}): {best_dropout_rate_f}')




# for activation_name in best_results_f.keys():
#     model_f = CNN(vocab_size_f, embedded_dim_f, output_size, activation_function,best_results_f[activation_name]['dropout_rate'])
#     loss_func = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model_f.parameters())
#     test_loader = dataloader_f.get_test_loader()
#     test_accuracy = train_model(model_f, test_loader_f, loss_func, optimizer, activation_function, dropout_rate,test_name, french)
#     test_results_f[activation_name] = test_accuracy

# # # Print the test accuracies for each activation function
# # for activation_name, accuracy in test_results_e.items():
# #     print(f'Test Accuracy for english (Activation: {activation_name}): {accuracy}')

# for activation_name, accuracy in test_results_f.items():
#     print(f'Test Accuracy for french (Activation: {activation_name}): {accuracy}')


# best_activation_name = None
# best_dropout_rate = None
# best_accuracy = 0.0

# for activation_name, results in best_results_e.items():
#     accuracy = results["accuracy"]
#     dropout_rate = results["dropout_rate"]

#     # Check if the current accuracy is better than the current best_accuracy
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_activation_name = activation_name
#         best_dropout_rate = dropout_rate

# activation_function = getattr(nn, best_activation_name)()
# # Step 2: Create a new instance of the CNN model using the best dropout rate and activation function.
# model_e = CNN(vocab_size_e, embedded_dim_e, output_size, activation_function, best_dropout_rate)

# # Step 3: Evaluate the model's accuracy on the test data using the test dataloader.
# loss_func = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model_e.parameters())
# test_loader = dataloader_e.get_test_loader()
# test_accuracy_cnn_e, precision_cnn_e, recall_cnn_e, f1_cnn_e, kappa_cnn_e = train_model(model_e, test_loader, loss_func, optimizer, activation_function, best_dropout_rate, test_name, english)

# print(f'Maximum accuracy for english (Activation: {best_activation_name}, Dropout Rate: {best_dropout_rate}): {test_accuracy_cnn_e}')


best_activation_name = None
best_dropout_rate = None
best_accuracy = 0.0

for activation_name, results in best_results_f.items():
    accuracy = results["accuracy"]
    dropout_rate = results["dropout_rate"]

    # Check if the current accuracy is better than the current best_accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_activation_name = activation_name
        best_dropout_rate = dropout_rate

activation_function = getattr(nn, best_activation_name)()
# Step 2: Create a new instance of the CNN model using the best dropout rate and activation function.
model_f = CNN(vocab_size_f, embedded_dim_f, output_size, activation_function, best_dropout_rate)

# Step 3: Evaluate the model's accuracy on the test data using the test dataloader.
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_f.parameters())
test_loader = dataloader_f.get_test_loader()
test_accuracy_cnn_f, precision_cnn_f, recall_cnn_f, f1_cnn_f, kappa_cnn_f = train_model(model_f, test_loader, loss_func, optimizer, activation_function, best_dropout_rate, test_name, french)

print(f'Maximum accuracy for english (Activation: {best_activation_name}, Dropout Rate: {best_dropout_rate}): {test_accuracy_cnn_f}')




   
import csv

# print(train_accuracy_mnb_e)
# print(val_accuracy_mnb_e)
# print(test_accuracy_mnb_e)
# print(precision_mnb_e)
# print(recall_mnb_e)
# print(f1_mnb_e)
# print(kappa_mnb_e)
# print(confusion_matrix_mnb_e)


# data_mnb_e = [
#     ["train_accuracy_mnb_e", train_accuracy_mnb_e],
#     ["val_accuracy_mnb_e", val_accuracy_mnb_e],
#     ["test_accuracy_mnb_e", test_accuracy_mnb_e],
#     ["precision_mnb_e", precision_mnb_e],
#     ["recall_mnb_e", recall_mnb_e],
#     ["f1_mnb_e", f1_mnb_e],
#     ["kappa_mnb_e", kappa_mnb_e],
#     ["confusion_matrix_mnb_e", confusion_matrix_mnb_e]
# ]

data_mnb_f = [
    ["train_accuracy_mnb_f", train_accuracy_mnb_f],
    ["val_accuracy_mnb_f", val_accuracy_mnb_f],
    ["test_accuracy_mnb_f", test_accuracy_mnb_f],
    ["precision_mnb_f", precision_mnb_f],
    ["recall_mnb_f", recall_mnb_f],
    ["f1_mnb_f", f1_mnb_f],
    ["kappa_mnb_f", kappa_mnb_f],
    ["confusion_matrix_mnb_f", confusion_matrix_mnb_f]
]

# data_cnn_e = [
#     ["train_accuracy_cnn_e", train_accuracy_e],
#     ["val_accuracy_cnn_e", highest_accuracy_e],
#     ["test_accuracy_cnn_e", test_accuracy_cnn_e],
#     ["precision_cnn_e", precision_cnn_e],
#     ["recall_cnn_e", recall_cnn_e],
#     ["f1_cnn_e", f1_cnn_e],
#     ["kappa_cnn_e", kappa_cnn_e]
# ]

data_cnn_f = [
    ["train_accuracy_cnn_f", train_accuracy_f],
    ["val_accuracy_cnn_f", highest_accuracy_f],
    ["test_accuracy_cnn_f", test_accuracy_cnn_f],
    ["precision_cnn_f", precision_cnn_f],
    ["recall_cnn_f", recall_cnn_f],
    ["f1_cnn_f", f1_cnn_f],
    ["kappa_cnn_f", kappa_cnn_f]
]

# Export to a CSV file
# with open("metrics_mnb_e.csv", "w", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(data_mnb_e)

# Export to a CSV file
with open(r"D:\UWaterloo\MSCI_641_Text_Analytics\Project\Output\metrics_mnb_f.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data_mnb_f)

# Export to a CSV file
# with open("metrics_cnn_e.csv", "w", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(data_cnn_e)

# Export to a CSV file
with open(r"D:\UWaterloo\MSCI_641_Text_Analytics\Project\Output\metrics_cnn_f.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data_cnn_f)

# data_english = {
#     "train_accuracy for MNB": train_accuracy_mnb_e,
#     "val_accuracy for MNB": val_accuracy_mnb_e,
#     "test_accuracy for MNB": test_accuracy_mnb_e,
#     "Precision for MNB": precision_mnb_e,
#     "Recall for MNB": recall_mnb_e,
#     "F1 Score for MNB": f1_mnb_e,
#     "Cohen Kappa score for MNB":kappa_mnb_e,
#     "Confusion Matrix for MNB":confusion_matrix_mnb_e
# }

# # Export to a CSV file
# with open("mnb_eng_model_metrics.csv", "w", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(data_english)

print('Exported Successfully')








