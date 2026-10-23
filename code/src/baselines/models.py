"""비교 기준 모델들. 전부 numpy만 사용 (저사양 PC에서도 실행 가능하게).

- majority     : 학습 데이터의 최빈 클래스만 예측 (찍기 기준선)
- ngram_table  : 학습 데이터에서 "이 문맥 다음에 가장 많이 나온 클래스"를 외운 표.
                 입력이 이산 품사 조합이므로 이 과제에서 사실상의 천장(Bayes 근사)이다.
- softmax MLP  : 역전파 + Adam. hidden=0이면 로지스틱 회귀.
"""
import numpy as np


def accuracy(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    assert pred.shape == y.shape, f"shape mismatch {pred.shape} vs {y.shape}"
    return float(np.mean(pred == y))


def majority_predict(y_train, n):
    return np.full(n, np.bincount(y_train).argmax(), dtype=np.int64)


def ngram_table_predict(ctx_train, y_train, ctx_test, n_classes):
    """보지 못한 문맥은 최빈 클래스로 대체."""
    fallback = np.bincount(y_train, minlength=n_classes).argmax()
    table = {}
    for c, y in zip(map(tuple, ctx_train), y_train):
        table.setdefault(c, np.zeros(n_classes, dtype=np.int64))[y] += 1
    best = {c: v.argmax() for c, v in table.items()}
    return np.array([best.get(tuple(c), fallback) for c in ctx_test], dtype=np.int64)


class MLP:
    """입력 -> (tanh 은닉층) -> softmax. 바이어스 포함, Adam, 미니배치, 검증 세트 조기 종료."""

    def __init__(self, n_in, n_hidden, n_classes, seed=0):
        rng = np.random.RandomState(seed)
        self.params = {}
        if n_hidden > 0:
            self.params["W1"] = rng.randn(n_in, n_hidden) * np.sqrt(1.0 / n_in)
            self.params["b1"] = np.zeros(n_hidden)
            self.params["W2"] = rng.randn(n_hidden, n_classes) * np.sqrt(1.0 / n_hidden)
        else:
            self.params["W2"] = rng.randn(n_in, n_classes) * 0.01
        self.params["b2"] = np.zeros(n_classes)
        self.n_hidden = n_hidden
        self.rng = rng

    def n_params(self):
        return int(sum(p.size for p in self.params.values()))

    def _forward(self, X):
        p = self.params
        h = np.tanh(X @ p["W1"] + p["b1"]) if self.n_hidden > 0 else X
        z = h @ p["W2"] + p["b2"]
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return h, e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self._forward(X)[1].argmax(1)

    def fit(self, X, y, X_val, y_val, epochs=60, batch=128, lr=3e-3, patience=8):
        p = self.params
        m = {k: np.zeros_like(v) for k, v in p.items()}
        v = {k: np.zeros_like(w) for k, w in p.items()}
        b1, b2, t = 0.9, 0.999, 0
        best_acc, best, wait, history = -1.0, None, 0, []
        for ep in range(epochs):
            order = self.rng.permutation(len(X))
            for s in range(0, len(X), batch):
                idx = order[s:s + batch]
                xb, yb = X[idx], y[idx]
                h, prob = self._forward(xb)
                d = prob
                d[np.arange(len(yb)), yb] -= 1.0
                d /= len(yb)
                g = {"W2": h.T @ d, "b2": d.sum(0)}
                if self.n_hidden > 0:
                    dh = (d @ p["W2"].T) * (1 - h ** 2)
                    g["W1"] = xb.T @ dh
                    g["b1"] = dh.sum(0)
                t += 1
                for k in p:
                    m[k] = b1 * m[k] + (1 - b1) * g[k]
                    v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                    p[k] -= lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + 1e-8)
            val_acc = accuracy(self.predict(X_val), y_val)
            history.append(val_acc)
            if val_acc > best_acc + 1e-4:
                best_acc, best, wait = val_acc, {k: w.copy() for k, w in p.items()}, 0
            else:
                wait += 1
                if wait >= patience:
                    break
        self.params = best
        return history
