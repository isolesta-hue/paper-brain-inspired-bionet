"""기억 트랙 4-1b: 언어용 해마형 기억 (REFERENCE U1·U3).

실험 06 결과("기억 영역은 넓은 연결이 필요하다, 국소 연결로는 안 된다")를 반영해 해마 구조를 따른다.

  치상회(DG) — 패턴 분리
    입력 x를 큰 뉴런 집단(N_dg)으로 고정 무작위 투영한 뒤, 가장 강한 k개만 발화 (k-WTA 억제 = S8 용도 전환).
    비슷한 문맥 → 겹치는 희소 부호, 다른 문맥 → 거의 겹치지 않는 부호.
  CA3→CA1 — 연합 저장 (원샷 헤비안)
    경험 하나를 한 번 보고: A[정답 클래스] += 부호(x).   역전파·반복 없음.
  회상
    새 문맥의 부호와 겹치는 기억들이 클래스별로 투표: 점수 = Â @ 부호(x).
    readout="scale" (기본): 각 DG 뉴런이 자기 출력 연결을 합 1로 정규화 (시냅스 스케일링, 항상성).
      Â[y, j] = A[y, j] / Σ_y A[y, j]  = "뉴런 j가 켜졌을 때 클래스 y였던 비율".
      raw(정규화 없음)는 경험이 쌓일수록 흔한 클래스의 표가 지나치게 커져 데이터가 많을 때 성능이 떨어졌다 (4-1b 사전 점검).
"""
import numpy as np


class HippocampalMemory:
    def __init__(self, n_in, n_classes, n_dg=8000, k=20, seed=0, readout="scale"):
        rng = np.random.RandomState(seed)
        self.P = rng.randn(n_dg, n_in)
        self.k = k
        self.A = np.zeros((n_classes, n_dg))
        self.n_dg = n_dg
        self.readout = readout

    def code(self, X):
        """X: (B, n_in) → 희소 이진 부호 (B, n_dg)."""
        h = X @ self.P.T
        idx = np.argpartition(h, -self.k, axis=1)[:, -self.k:]
        c = np.zeros_like(h)
        np.put_along_axis(c, idx, 1.0, axis=1)
        return c

    def store(self, X, y, chunk=2000):
        """원샷 저장: 각 경험을 한 번씩만 본다. (메모리 절약을 위해 묶음 단위로 계산)"""
        for i in range(0, len(X), chunk):
            np.add.at(self.A, y[i:i + chunk], self.code(X[i:i + chunk]))

    def predict(self, X, chunk=2000):
        A = self.A / np.maximum(1.0, self.A.sum(0, keepdims=True)) if self.readout == "scale" else self.A
        scores = np.vstack([self.code(X[i:i + chunk]) @ A.T for i in range(0, len(X), chunk)])
        return (scores + np.random.RandomState(0).rand(*scores.shape) * 1e-9).argmax(1)

    def n_params(self):
        return self.A.size   # 저장되는 가중치 (투영 P는 고정 무작위라 학습 파라미터 아님)
