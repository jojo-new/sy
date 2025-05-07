Assignment – 1

import pandas as pd
df = pd.read_csv(r"C:\Users\dell\Downloads\boston_housing.csv")
print(df.head())

df.isnull().sum()

from sklearn.model_selection import train_test_split

X = df.loc[:, df.columns != 'MEDV']
y = df.loc[:, df.columns == 'MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
mms.fit(X_train)
X_train = mms.transform(X_train)
X_test = mms.transform(X_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(128, input_shape=(13, ), activation='relu', name='dense_1'))
model.add(Dense(64, activation='relu', name='dense_2'))
model.add(Dense(1, activation='linear', name='dense_output'))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()


history = model.fit(X_train, y_train, epochs=100, validation_split=0.05, verbose = 1)

mse, mae = model.evaluate(X_test, y_test)
print('MSE: ', mse)
print('MAE: ', mae)

y1 = model.predict(X_test[:])

y_test

ps=[]
for i in y1:
    ps.append(list(i)[0])

d = pd.DataFrame({'actual':y_test['MEDV'],'predicted':ps})

d

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y1)
print('R² Score:', r2)







Assignment_no =02

from tensorflow.keras.datasets import imdb
(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words = 10000)

import numpy as np
def vectorize_sequences(sequences, dimensions = 10000):
  results = np.zeros((len(sequences), dimensions))
  for i,sequences in enumerate(sequences):
    results[i, sequences] = 1
  return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_label).astype('float32')
y_test = np.asarray(test_label).astype('float32')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(16, input_shape=(10000, ), activation = "relu"))
model.add(Dense(16, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer='adam', loss = 'mse', metrics = ['accuracy'])
model.summary()

history = model.fit(x_train, y_train, validation_split = 0.2, epochs = 20, verbose = 1, batch_size = 512)

mse,mae = model.evaluate(x_test,y_test)
print('MSE ',mse)
print('MAE ',mae)

y_preds = model.predict(x_test)


preds=[]
for i in y_preds:
    if i[0]>0.5:
        preds.append(1)
    else:
        preds.append(0)


from sklearn.metrics import accuracy_score,precision_score,recall_score
print(accuracy_score(y_test,preds))
print(precision_score(y_test,preds))
print(recall_score(y_test,preds))

word_index = imdb.get_word_index()
def return_token(tid):
    for k,v in word_index.items():
        if v == tid-3:
            return k
    return '?'     


def print_review(id_):
    sentence = ' '.join(return_token(i) for i in train_data[id_])
    return sentence

print_review(0)

train_label[0]
print_review(1)

train_label[1]








Assignment_no=3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv(r"C:\Users\dell\Downloads\GOOG.csv")  
df.isnull().sum()
df = df[['date', 'close']]
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['close']])



plt.figure(figsize=(12, 6))
plt.plot(df['close'], label='Google Stock Price')
plt.title('Google Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.legend()
plt.show()



def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)



split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(sequence_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


y_pred = model.predict(X_test)


y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100

print(f"\n📈 Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")



plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Actual Price')
plt.plot(y_pred_rescaled, label='Predicted Price')
plt.title('Google Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Price USD')
plt.legend()
plt.show()
















#include <iostream> 
#include <vector> 
#include <queue> 
#include <omp.h> 
using namespace std; 

class Graph { 
    int V; 
    vector<vector<int>> adj; 

public: 
    Graph(int V) { 
        this->V = V; 
        adj.resize(V); 
    } 

    void addEdge(int u, int v) { 
        adj[u].push_back(v); 
        adj[v].push_back(u); // For undirected graph 
    } 
    void parallelBFS(int start) { 
        vector<bool> visited(V, false); 
        queue<int> q; 

        visited[start] = true; 
        q.push(start); 

        cout << "\nParallel BFS starting from node " << start << ":\n"; 

        while (!q.empty()) { 
            int size = q.size(); 
            vector<int> levelNodes; 

            #pragma omp parallel 
            { 
                vector<int> localNodes; 

                #pragma omp for 
                for (int i = 0; i < size; i++) { 
                    int node = -1; 
                    bool valid = false; 
                    #pragma omp critical 
                    { 
                        if (!q.empty()) { 
                            node = q.front(); 
                            q.pop(); 
                            valid = true; 
                        } 
                    } 

                    if (!valid) continue; 

                    localNodes.push_back(node); 

                    for (int neighbor : adj[node]) { 
                        bool needVisit = false; 

                        #pragma omp critical 
                        { 
                            if (!visited[neighbor]) { 
                                visited[neighbor] = true; 
                                q.push(neighbor); 
                                needVisit = true; 
                            } 
                        } 
                    } 
                } 

                #pragma omp critical 
                levelNodes.insert(levelNodes.end(), localNodes.begin(), localNodes.end()); 
            } 

            for (int node : levelNodes) 
                cout << node << " "; 
        } 

        cout << endl; 
    } 

    void parallelDFSUtil(int node, vector<bool>& visited) { 
        bool alreadyVisited; 

        #pragma omp critical 
        { 
            alreadyVisited = visited[node]; 
            if (!alreadyVisited) { 
                visited[node] = true; 
                cout << node << " "; 
            } 
        } 

        if (alreadyVisited) return; 

        #pragma omp parallel for 
        for (int i = 0; i < adj[node].size(); i++) { 
            int neighbor = adj[node][i]; 

            #pragma omp task 
            parallelDFSUtil(neighbor, visited); 
        } 
    } 

    void parallelDFS(int start) { 
        vector<bool> visited(V, false); 
        cout << "\nParallel DFS starting from node " << start << ":\n"; 

        #pragma omp parallel 
        { 
            #pragma omp single 
            parallelDFSUtil(start, visited); 
        } 

        cout << endl; 
    } 
}; 

int main() { 
    int V, E; 
    cout << "Enter number of vertices: "; 
    cin >> V; 
    Graph g(V); 
    cout << "Enter number of edges: "; 
    cin >> E; 
    cout << "Enter each edge as two space-separated vertices (u v):\n"; 
    for (int i = 0; i < E; i++) { 
        int u, v; 
        cin >> u >> v; 
        g.addEdge(u, v); 
    } 
    int start; 
    cout << "Enter starting node for traversal: "; 
    cin >> start; 

    g.parallelBFS(start); 
    g.parallelDFS(start); 

    return 0; 
}


