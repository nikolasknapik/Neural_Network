import numpy as np
import matplotlib.pyplot as plt

# XOR dáta
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
Y = np.array([[0],[1],[1],[1]], dtype=float)

n_in_W1, n_out_W1 = 2, 4  
n_in_W2, n_out_W2 = 4, 4  
n_in_W3, n_out_W3 = 4, 1

# Inicializácia parametrov
np.random.seed(0)
W1 = np.random.randn(n_in_W1, n_out_W1) * np.sqrt(2 / n_in_W1)
b1 = np.zeros((1, n_out_W1))
W2 = np.random.randn(n_in_W2, n_out_W2) * np.sqrt(2 / n_in_W2)
b2 = np.zeros((1, n_out_W2))
W3 = np.random.randn(n_in_W3, n_out_W3) * 0.1
b3 = np.zeros((1, n_out_W3))

# Aktivačné funkcie a ich derivácie
def forward_tanh(x):
    return np.tanh(x)

def backward_tanh(y):
    # Ak y = tanh(x), derivácia podľa x je (1 - y^2)
    return 1 - y**2

def forward_sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def backward_sigmoid(y):
    # derivácia sigmoid je y*(1-y)
    return y*(1.0-y)

def forward_relu(x):
    return np.maximum(0, x)

def backward_relu(x):
    grad = np.where(x > 0, 1, 0)
    return grad



# MSE a jeho derivácia
def mse(pred, target):
    return np.mean((pred - target)**2)

def mse_grad(pred, target):
    return 2*(pred - target)/pred.shape[0]


def forward_pass(X):
    # Prvá skrytá vrstva
    z1 = X.dot(W1) + b1
    a1 = forward_tanh(z1)
    
    # Druhá skrytá vrstva
    z2 = a1.dot(W2) + b2
    a2 = forward_tanh(z2)
    
    # Výstupná vrstva
    z3 = a2.dot(W3) + b3
    a3 = forward_sigmoid(z3)
    
    return z1, a1, z2, a2, z3, a3

# Backward Pass
def backward_pass(X, Y, z1, a1, z2, a2, z3, a3):
    # Gradient chyby podľa výstupu
    dz3 = mse_grad(a3, Y) * backward_sigmoid(a3)  # dL/dz3
    dW3 = a2.T.dot(dz3)
    db3 = np.sum(dz3, axis=0, keepdims=True)
    
    # Spätný prejazd druhou skrytou vrstvou
    dz2 = dz3.dot(W3.T) * backward_tanh(a2)  # dL/dz2
    dW2 = a1.T.dot(dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    
    # Spätný prejazd prvou skrytou vrstvou
    dz1 = dz2.dot(W2.T) * backward_tanh(a1)  # dL/dz1
    dW1 = X.T.dot(dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3

# Update parametrov
def update_params(lr, dW1, db1, dW2, db2, dW3, db3):
    global W1, b1, W2, b2, W3, b3
    
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3
# Tréning
epochs = 500
lr = 0.1
train_errors = []

for epoch in range(epochs):
    z1, a1, z2, a2, z3, a3 = forward_pass(X)
    loss = mse(a3, Y)
    train_errors.append(loss)
    
    dW1, db1, dW2, db2, dW3, db3 = backward_pass(X, Y, z1, a1, z2, a2, z3, a3)
    update_params(lr, dW1, db1, dW2, db2, dW3, db3)

# Graf priebehu chyby
plt.plot(train_errors)
plt.title("Tréningová chyba (MSE) - OR")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# Výstup po natrénovaní
_, _, _, _, _, pred = forward_pass(X) 
rounded_pred = np.round(pred)
print("X:\n", X)
print("Predikcia:\n", pred)
print("Predikcia zaokruhlene: \n", rounded_pred)
print("Cieľ:\n", Y)
