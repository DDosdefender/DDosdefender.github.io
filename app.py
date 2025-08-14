from flask import Flask, make_response, request, abort, render_template, g, jsonify
import os
import numpy as np
import time
from collections import deque

app = Flask(__file__)

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
        self.input_size = 10
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

    def save(self):
        np.savez(self.model_path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self):
        data = np.load(self.model_path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']

    def predict(self, raw_inputs):
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

model = SimpleNeuralNet()

_ip_memory = {}

def isThreat(ip0, ip1, ip2, req_at60, req_at50, req_at40, req_at30, req_at20, req_at10):
    ip_key = (ip0, ip1, ip2)
    past_sameValue = _ip_memory.get(ip_key, 1.0)

    full_input = [
        ip0, ip1, ip2,
        past_sameValue,
        req_at60, req_at50, req_at40,
        req_at30, req_at20, req_at10
    ]

    prediction = model.predict(full_input)

    target = target_function(full_input)

    normalized_input = [
        ip0 / 255, ip1 / 255, ip2 / 255,
        past_sameValue,
        req_at60 / 100, req_at50 / 100, req_at40 / 100,
        req_at30 / 100, req_at20 / 100, req_at10 / 100,
    ]
    model.backward(np.array([normalized_input]), np.array([[target]]), np.array([[prediction]]), lr=0.01)

    _ip_memory[ip_key] = prediction

    return prediction

# --- Request tracking for protection ---
_request_log = {}

def cleanup_old_requests(ip, current_time):
    if ip not in _request_log:
        return
    while _request_log[ip] and _request_log[ip][0] < current_time - 60:
        _request_log[ip].popleft()
    if not _request_log[ip]:
        del _request_log[ip]

def count_requests_in_range(ip, current_time, start_sec_ago, end_sec_ago):
    if ip not in _request_log:
        return 0
    lower_bound = current_time - start_sec_ago
    upper_bound = current_time - end_sec_ago
    count = 0
    for t in _request_log[ip]:
        if lower_bound <= t < upper_bound:
            count += 1
    return count

@app.before_request
def protect_with_isThreat():
    client_ip = request.remote_addr
    if client_ip is None:
        return

    now = time.time()
    if client_ip not in _request_log:
        _request_log[client_ip] = deque()
    _request_log[client_ip].append(now)

    cleanup_old_requests(client_ip, now)

    # Parse first 3 parts of IPv4 or default zeros
    try:
        ip_parts = list(map(int, client_ip.split('.')[:3]))
        if len(ip_parts) != 3:
            ip_parts = (ip_parts + [0,0,0])[:3]
    except:
        ip_parts = [0, 0, 0]

    req_at60 = len(_request_log[client_ip])  # last 60 seconds total
    req_at50 = count_requests_in_range(client_ip, now, 60, 50)
    req_at40 = count_requests_in_range(client_ip, now, 50, 40)
    req_at30 = count_requests_in_range(client_ip, now, 40, 30)
    req_at20 = count_requests_in_range(client_ip, now, 30, 20)
    req_at10 = count_requests_in_range(client_ip, now, 20, 10)

    threat_score = isThreat(ip_parts[0], ip_parts[1], ip_parts[2],
                           req_at60, req_at50, req_at40,
                           req_at30, req_at20, req_at10)

    g.score = float(str(threat_score)[0:4]) * 100

    if threat_score > 0.7:
        abort(403, description=f"Blocked by threat detection (score: {threat_score:.2f})")

@app.route('/')
def home():
    html = render_template('index.html')
    res = make_response(html)
    res.headers['AI-score'] = g.score 
    return res

@app.route('/request-stats')
def request_stats():
    client_ip = request.remote_addr
    now = time.time()

    data = {
        'req_at60': count_requests_in_range(client_ip, now, 70, 60),
        'req_at50': count_requests_in_range(client_ip, now, 60, 50),
        'req_at40': count_requests_in_range(client_ip, now, 50, 40),
        'req_at30': count_requests_in_range(client_ip, now, 40, 30),
        'req_at20': count_requests_in_range(client_ip, now, 30, 20),
        'req_at10': count_requests_in_range(client_ip, now, 20, 10)
    }
    return jsonify(data)

if __name__ == '__main__':
    if os.getenv("INRENDER") == "true":
        app.run(host="0.0.0.0", port=os.getenv("PORT"))
        print(F"[INFO] Running in render mode on port {os.getenv('PORT')}")
    else:
        app.run(port=5000)
