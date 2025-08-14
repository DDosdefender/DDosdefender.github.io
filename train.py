import os
import random
import numpy as np

# -------------------------
# Target function (10 inputs)
# -------------------------
def target_function(inputs):
    ip0, ip1, ip2, past_sameValue, req_at60, req_at50, req_at40, req_at30, req_at20, req_at10 = inputs

    ip_score = (ip0 << 16) + (ip1 << 8) + ip2
    total_requests = req_at60 + req_at50 + req_at40 + req_at30 + req_at20 + req_at10
    weighted_requests = (
        req_at60 * 1 +
        req_at50 * 2 +
        req_at40 * 3 +
        req_at30 * 4 +
        req_at20 * 5 +
        req_at10 * 6
    )

    if total_requests > 100:
        return 1.0
    elif weighted_requests > 300:
        return 0.75
    elif total_requests > 50:
        return 0.5
    elif past_sameValue > 0.7:
        return 0.25
    else:
        return 0.0

# -------------------------
# Activation
# -------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# -------------------------
# Neural Network
# -------------------------
class SimpleNeuralNet:
    def __init__(self):
        self.input_size = 10  # updated to 10
        self.hidden_size = 16
        self.output_size = 1

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))

        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.npz")
        if os.path.exists(self.model_path):
            self.load()
            print("[INFO] Loaded existing model from model.npz")
        else:
            print("[INFO] No saved model found. Initialized new model.")

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, X, y, y_pred, lr):
        m = len(X)
        loss_grad = 2 * (y_pred - y) / m

        dW2 = np.dot(self.a1.T, loss_grad)
        db2 = np.sum(loss_grad, axis=0, keepdims=True)

        dhidden = np.dot(loss_grad, self.W2.T) * relu_derivative(self.z1)
        dW1 = np.dot(X.T, dhidden)
        db1 = np.sum(dhidden, axis=0, keepdims=True)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, epochs=1000, batch_size=32, lr=0.01):
        # Create a pool of persistent IP addresses
        ip_pool = [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(50)
        ]

        for epoch in range(epochs):
            X_batch = []
            y_batch = []

            for _ in range(batch_size):
                # Choose a repeating IP from the pool
                ip0, ip1, ip2 = random.choice(ip_pool)

                # Keep behavior random but bounded
                past_sameValue = random.uniform(0.0, 1.0)
                req_at60 = random.randint(0, 50)
                req_at50 = random.randint(0, 50)
                req_at40 = random.randint(0, 50)
                req_at30 = random.randint(0, 50)
                req_at20 = random.randint(0, 50)
                req_at10 = random.randint(0, 50)

                inputs = [
                    ip0, ip1, ip2,
                    past_sameValue,
                    req_at60, req_at50, req_at40,
                    req_at30, req_at20, req_at10
                ]

                output = target_function(inputs)

                inputs_norm = [
                    ip0 / 255,
                    ip1 / 255,
                    ip2 / 255,
                    past_sameValue,
                    req_at60 / 100,
                    req_at50 / 100,
                    req_at40 / 100,
                    req_at30 / 100,
                    req_at20 / 100,
                    req_at10 / 100,
                ]

                X_batch.append(inputs_norm)
                y_batch.append([output])

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)

            y_pred = self.forward(X_batch)
            loss = np.mean((y_batch - y_pred) ** 2)
            self.backward(X_batch, y_batch, y_pred, lr)

            if epoch % 100 == 0:
                print(f"Epoch {epoch} - Loss: {loss:.6f}")

        self.save()
        print("[INFO] Model saved to model.npz")

    def predict(self, raw_inputs):
        # Normalize before prediction
        inputs_norm = [
            raw_inputs[0] / 255,
            raw_inputs[1] / 255,
            raw_inputs[2] / 255,
            raw_inputs[3],
            raw_inputs[4] / 100,
            raw_inputs[5] / 100,
            raw_inputs[6] / 100,
            raw_inputs[7] / 100,
            raw_inputs[8] / 100,
            raw_inputs[9] / 100,
        ]
        inputs = np.array(inputs_norm).reshape(1, -1)
        return self.forward(inputs)[0, 0]

    def save(self):
        np.savez(self.model_path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self):
        data = np.load(self.model_path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']

model = SimpleNeuralNet()

# Memory to store last predictions per IP (used as past_sameValue)
_ip_memory = {}

def isThreat(ip0, ip1, ip2, req_at60, req_at50, req_at40, req_at30, req_at20, req_at10):
    # Compose IP tuple
    ip_key = (ip0, ip1, ip2)

    # Get last prediction for this IP or default to 0.0
    past_sameValue = _ip_memory.get(ip_key, 0.0)

    # Compose full input vector
    full_input = [
        ip0, ip1, ip2,
        past_sameValue,
        req_at60, req_at50, req_at40,
        req_at30, req_at20, req_at10
    ]

    # Predict using the model
    prediction = model.predict(full_input)

    # Get the true label using the target function (supervised feedback)
    target = target_function(full_input)

    # Train on this one example
    normalized_input = [
        ip0 / 255, ip1 / 255, ip2 / 255,
        past_sameValue,
        req_at60 / 100, req_at50 / 100, req_at40 / 100,
        req_at30 / 100, req_at20 / 100, req_at10 / 100,
    ]
    model.backward(np.array([normalized_input]), np.array([[target]]), np.array([[prediction]]), lr=0.01)

    # Save updated prediction for next time
    _ip_memory[ip_key] = prediction

    return prediction
