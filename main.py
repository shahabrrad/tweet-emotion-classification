from os import name
import numpy as np
import pandas as pd
import scipy.special as sc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# get tokens from the text
def tokenize(text):
    tokens = text.split()
    return tokens

# create count dictionary of the words
def count_dict(sentences, unique_tokens):
    count_dict = {}
    for word in unique_tokens:
        count_dict[word] = 0
    for sent in sentences:
        for word in sent:
            count_dict[word] += 1
    return count_dict

##### tfidf encoding code #####
def term_frequency(document, word):
    N = len(document)
    occurance = len([token for token in document if token == word])
    return occurance / N

def inverse_document_frequency(word, doc_count):
    try:
        word_occurance = word_count[word] + 1
    except:
        word_occurance = 1
    return np.log(doc_count / word_occurance)

def tf_idf(sentence, doc_count, unique_tokens, word_index):
    vec = np.zeros((len(unique_tokens),))
    for word in sentence:
        tf = term_frequency(sentence, word)
        idf = inverse_document_frequency(word, doc_count)
        if word in word_index:
          vec[word_index[word]] = tf * idf
    return vec

def create_tfidf_data(train_data, unique_tokens, word_index):
  data = []
  for terms in train_data.tokens.values:
    data.append(tf_idf(terms, len(train_data), unique_tokens, word_index))
  return data

###############################

###### one hot encoding #####
def one_hot_encode(labels):
    # Convert labels to a pandas Series if not already
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels)

    # Get the unique labels
    unique_labels = sorted(labels.unique())

    # Create an empty DataFrame with columns for each unique label
    one_hot_encoded = pd.DataFrame(np.zeros((len(labels), len(unique_labels))), columns=unique_labels)

    # Fill the DataFrame with ones for the corresponding labels
    for label in unique_labels:
        one_hot_encoded[label] = (labels == label).astype(int)

    return one_hot_encoded, unique_labels
##############

# negative log likelihood for LR
def loss(X, Y, W):
    Z = - X @ W
    n = X.shape[0]
    loss = (1/n) * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
    return loss

# gradient function for LR
def gradient(X, Y, W, mu):
    Z = - X @ W
    Y_pred = sc.softmax(Z, axis=1)
    n = X.shape[0]
    regularization = ((mu * np.sign(W)) / n)
    gd = 1/n * (X.T @ (Y - Y_pred)) +  regularization
    return gd

# gradient descent for LR with validation data
def gradient_descent(X, Y, x_val = None, y_val = None, max_iter=1000, eta=0.1, mu=0.01):
    Y_onehot = Y
    y_val_onehot = y_val
    # y_val_onehot = one_hot_encode(y_val)
    # print(X)
    W = np.zeros((X.shape[1], np.array(Y_onehot).shape[1]))
    step = 0
    step_lst = []
    loss_lst = []
    W_lst = []

    val_loss_lst = []

    while step < max_iter:
        step += 1
        W -= eta * gradient(X, Y_onehot, W, mu)
        step_lst.append(step)
        W_lst.append(W)
        # get training loss
        train_loss = loss(X, Y_onehot, W)
        loss_lst.append(train_loss)
        valid_loss = "undefined"
        if x_val != None:
        # get validation loss
            valid_loss = loss(x_val, y_val_onehot, W)
            val_loss_lst.append(valid_loss)
        if step % 100 == 0:
          print(f"gradient descent step {step} train loss {train_loss} validation loss {valid_loss}")
    # training loss
    df = pd.DataFrame({
        'step': step_lst,
        'loss': loss_lst
    })
    # validation loss
    if x_val != None:
        df_val = pd.DataFrame({
            'step': step_lst,
            'loss': val_loss_lst
        })
    else:
        df_val = None
    return df, df_val, W

# predict function for LR
def predict(W, X):
        Z = - X @ W
        y_pred = sc.softmax(Z, axis=1)
        return np.argmax(y_pred, axis=1)

def LR():
    df = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    # create tfidf of train and test data
    df['tokens'] = df['text'].apply(tokenize)
    df_test['tokens'] = df_test['text'].apply(tokenize)
    all_tokens = np.concatenate(df['tokens'].to_list())
    unique_tokens = np.unique(all_tokens, axis=0)
    word_index = {}
    for i, word in enumerate(unique_tokens):
        word_index[word] = i
    word_count = count_dict(df.tokens, unique_tokens)
    tfidf_vectors = create_tfidf_data(df, unique_tokens, word_index)
    tfidf_vectors_test = create_tfidf_data(df_test, unique_tokens, word_index)
    X_train = tfidf_vectors
    y_train = df.emotions
    # resampling
    smote = SMOTE(sampling_strategy='minority')
    X_train, y_train = smote.fit_resample(X_train, y_train)
    # one hot encoding
    y_train, labels = one_hot_encode(y_train.values)
    loss, val_loss, W = gradient_descent(pd.DataFrame(X_train), y_train, x_val=None, y_val=None, max_iter=1000, eta=0.8, mu=0.1)
    predicted_y = predict(W, pd.DataFrame(tfidf_vectors_test))
    predicted_emotion = [labels[i] for i in predicted_y]
    df_test["emotions"] = predicted_emotion
    df_test = df_test.drop(columns=['tokens'])
    df_test = df_test.set_index('id')
    # print(df_test)
    df_test.to_csv("test_lr.csv")


    # your logistic regression 


#################################
#################################
##### functions used in NN ######
#################################
#################################

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Define activation functions
def relu(x):
    return np.maximum(0, x)

# Weight initialization
def init_weights(rows, cols):
    return np.random.randn(rows, cols) * np.sqrt(2 / (rows + cols))

# Forward propagation
def forward_prop(X, W1, W2):
    Z1 = np.dot(X, W1)
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2)
    A2 = softmax(Z2)
    return A1, A2

# L1 regularization
def l1_regularization(W1, W2, lamb):
    return lamb * (np.sum(np.abs(W1)) + np.sum(np.abs(W2)))

# L2 regularization
def l2_regularization(W1, W2, lamb):
    return lamb * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

# Training function with Adam optimization
def train(X, y, X_val= None, y_val = None, hidden_nodes=50, lamb=0.1, reg_type="l1", epochs=1000, learning_rate=0.001, batch_size=256, beta1=0.9, beta2=0.999, epsilon=1e-8):
    train_losss = []
    valid_losss = []
    m = X.shape[0]
    input_nodes = X.shape[1]
    output_nodes = y.shape[1]

    W1 = init_weights(input_nodes, hidden_nodes)
    W2 = init_weights(hidden_nodes, output_nodes)

    W1_m, W1_v = np.zeros_like(W1), np.zeros_like(W1)
    W2_m, W2_v = np.zeros_like(W2), np.zeros_like(W2)

    for epoch in range(epochs):
        # if epoch == 1000:
        #   learning_rate = 0.0001
        # Shuffle the data
        permutation = np.random.permutation(m)
        X = X.iloc[permutation]
        y = y.iloc[permutation]
        # print(y)
        #get train loss
        A1, A2 = forward_prop(X, W1, W2)
        cost = -np.mean(y * np.log(A2), axis = 0)
        # if reg_type == 'l1':
        #         cost += l1_regularization(W1, W2, lamb)
        # elif reg_type == 'l2':
        #         cost += l2_regularization(W1, W2, lamb)
        train_losss.append(cost)

        # Mini-batch Adam
        for i in range(0, m, batch_size):
            X_batch = X.iloc[i:i + batch_size]
            y_batch = y.iloc[i:i + batch_size]

            A1, A2 = forward_prop(X_batch, W1, W2)

            cost = -np.mean(y_batch * np.log(A2), axis = 0)
            if reg_type == 'l1':
                cost += l1_regularization(W1, W2, lamb)
            elif reg_type == 'l2':
                cost += l2_regularization(W1, W2, lamb)

            dZ2 = A2 - y_batch
            dW2 = (1 / batch_size) * np.dot(A1.T, dZ2)
            dZ1 = np.dot(dZ2, W2.T) * (A1 > 0)
            dW1 = (1 / batch_size) * np.dot(X_batch.T, dZ1)

            if reg_type == 'l1':
                dW2 += (lamb / batch_size) * np.sign(W2)
                dW1 += (lamb / batch_size) * np.sign(W1)
            elif reg_type == 'l2':
                dW2 += (2 * lamb / batch_size) * W2
                dW1 += (2 * lamb / batch_size) * W1

            # Adam updates
            W1_m = beta1 * W1_m + (1 - beta1) * dW1
            W2_m = beta1 * W2_m + (1 - beta1) * dW2

            W1_v = beta2 * W1_v + (1 - beta2) * (dW1 ** 2)
            W2_v = beta2 * W2_v + (1 - beta2) * (dW2 ** 2)

            W1_m_hat = W1_m / (1 - beta1 ** (epoch + 1))
            W2_m_hat = W2_m / (1 - beta1 ** (epoch + 1))

            W1_v_hat = W1_v / (1 - beta2 ** (epoch + 1))
            W2_v_hat = W2_v / (1 - beta2 ** (epoch + 1))

            W1 -= learning_rate * W1_m_hat / (np.sqrt(W1_v_hat) + epsilon)
            W2 -= learning_rate * W2_m_hat / (np.sqrt(W2_v_hat) + epsilon)
          
        #get valid loss
        # if X_val.any():
        A1, A2 = forward_prop(X_val, W1, W2)
        cost = -np.mean(y_val * np.log(A2), axis = 0)
        valid_losss.append(cost)
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}: Cost = {np.mean(cost)}")

    return W1, W2, train_losss, valid_losss



def NN():
    df = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    # create tfidf of train and test data
    df['tokens'] = df['text'].apply(tokenize)
    df_test['tokens'] = df_test['text'].apply(tokenize)
    all_tokens = np.concatenate(df['tokens'].to_list())
    unique_tokens = np.unique(all_tokens, axis=0)
    word_index = {}
    for i, word in enumerate(unique_tokens):
        word_index[word] = i
    word_count = count_dict(df.tokens, unique_tokens)
    tfidf_vectors = create_tfidf_data(df, unique_tokens, word_index)
    tfidf_vectors_test = create_tfidf_data(df_test, unique_tokens, word_index)
    X_train = tfidf_vectors
    y_train = df.emotions
    # resampling
    smote = SMOTE(sampling_strategy='minority')
    X_train, y_train = smote.fit_resample(X_train, y_train)
    # one hot encoding
    y_train, labels = one_hot_encode(y_train.values)
    W1, W2, train_loss, validation_loss = train(pd.DataFrame(X_train), y_train, X_val=pd.DataFrame(X_train), y_val=y_train, hidden_nodes=50, lamb=0.1, reg_type="l1", epochs=1000, learning_rate=0.001, batch_size=256)
    _, A2_val = forward_prop(pd.DataFrame(tfidf_vectors_test), W1, W2)
    prediction = np.argmax(A2_val, axis=1)
    predicted_emotion = [labels[i] for i in prediction]
    df_test["emotions"] = predicted_emotion
    df_test = df_test.drop(columns=['tokens'])
    df_test = df_test.set_index('id')
    # print(df_test)
    df_test.to_csv("test_nn.csv")
    return
    # your Multi-layer Neural Network

if __name__ == '__main__':
    print ("..................Beginning of Logistic Regression................")
    LR()
    print ("..................End of Logistic Regression................")

    print("------------------------------------------------")

    print ("..................Beginning of Neural Network................")
    NN()
    print ("..................End of Neural Network................")



###### cross validation code

# def split_data_kfold(X, y, n_splits=5, shuffle=True, random_state=None):

#     if shuffle:
#         if random_state is not None:
#             np.random.seed(random_state)
#         indices = np.random.permutation(len(X))
#     else:
#         indices = np.arange(len(X))
    
#     fold_indices = []
#     fold_size = len(X) // n_splits
    
#     for i in range(n_splits):
#         start = i * fold_size
#         end = start + fold_size
#         val_indices = indices[start:end]
#         train_indices = np.concatenate([indices[:start], indices[end:]])
#         fold_indices.append((train_indices, val_indices))
    
#     return fold_indices


# # Cross-validation
# def cross_val(X, y, epochs, learning_rates, reg_rates):
#     kf = split_data_kfold(X, y, n_splits=5, shuffle=True, random_state=42)
#     best_hidden_nodes = None
#     best_lamb = None
#     best_reg_type = None
#     best_score = -np.inf
#     loss_graph = []
#     smote = SMOTE(sampling_strategy='minority')
#     kf = split_data_kfold(X, y, n_splits=5, shuffle=True, random_state=42)
#     for i,(train_idx, val_idx) in enumerate(kf):
#           # print(X.iloc[np.array([138,  543,  673])])
#           # print(val_idx[0])
#           X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#           y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
#     # print(len(y_train))
#     # print(len(y_val))
#     X_train, y_train = smote.fit_resample(X_train, y_train)
#     # print(y_train.values.T[0])
#     # print(y_val)
#     y_train, labels = one_hot_encode(y_train.values)
#     y_val, labels = one_hot_encode(y_val.values)


#     for lr in learning_rates:
#         for reg in reg_rates:
#             # for batch_size in batch_sizes:
#                 fold_scores = []
#                 # print(y_train)
                
#                 loss, val_loss, W = gradient_descent(X_train, y_train, x_val=X_val, y_val=y_val, max_iter=1000, eta=lr, mu=reg)
#                 val_predictions = predict(W, pd.DataFrame(X_val))
#                 # print(val_predictions)
#                 # print(y_val)
#                 y_valed = [np.argmax(y_val.iloc[i]) for i in range(240)]
#                 val_accuracy = np.mean(val_predictions == y_valed)
#                 # loss_graph.append({"train": train_loss, "validation": validation_loss})
#                 print(loss)
#                 print(val_loss)
#                 plt.plot(loss)
#                 plt.plot(val_loss)
#                 plt.legend(["train", "valid"])
#                 plt.title(f"learning rate {lr} regularization {reg}")
#                 plt.show()

#                 print(f"learning rate {lr} regularization rate {reg}, fold_score {val_accuracy}")


#     return best_hidden_nodes, best_lamb, best_reg_type, best_score, loss_graph

# df = pd.read_csv('train.csv')
# df_test = pd.read_csv('test.csv')

#     # create tfidf of train and test data
# df['tokens'] = df['text'].apply(tokenize)
# df_test['tokens'] = df_test['text'].apply(tokenize)
# all_tokens = np.concatenate(df['tokens'].to_list())
# unique_tokens = np.unique(all_tokens, axis=0)
# word_index = {}
# for i, word in enumerate(unique_tokens):
#     word_index[word] = i
# word_count = count_dict(df.tokens, unique_tokens)
# tfidf_vectors = create_tfidf_data(df, unique_tokens, word_index)
# tfidf_vectors_test = create_tfidf_data(df_test, unique_tokens, word_index)
    

# hidden_nodes_list = [50, 100, 200] #, 100]

# learning_rates = [0.8, 0.5, 0.1, 0.01, 0.001] #[0.0001, 0.001, 0.01] #, 0.1]
# reg_rates=[0.1] #,0.01,0.001]

# # y_one_hot = one_hot_encode(y_train)
# # x_tfidf = pd.DataFrame(create_tfidf_data(df))
#     # Cross-validation and grid search
# best_hidden_nodes_og, best_lamb_og, best_reg_type_og, best_score_og, loss_graphog = cross_val(pd.DataFrame(tfidf_vectors), df.emotions, epochs=1000, learning_rates=learning_rates, reg_rates=reg_rates)

# print(f"Best Hyperparameters: Hidden Nodes = {best_hidden_nodes_og}, Lambda = {best_lamb_og}, Regularization = {best_reg_type_og}")
# print(f"Best Validation Accuracy: {best_score_og}")