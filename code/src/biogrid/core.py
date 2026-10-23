"""BioGrid 1단계 최소 코어.

설계서(archive 17번) 중 이번 단계에 넣은 것만:
  - 2차원 격자 H행 × W열, 신호는 좌→우로 흐른다 (a-1, a-2)
  - 1열: 입력 벡터 전체를 받는 감각 열 (입력 뉴런층에 해당)
  - 2열~: 왼쪽 열의 [위, 옆, 아래] 3개 이웃에서만 입력을 받는다 (b-1, V1.4의 3방향 압축 텐서)
  - 뉴런은 문턱(V_base)과 비교해 발화/침묵. 발화는 확률적이다:
      p = sigmoid(Σ w·입력 − V_base),  발화 s ~ Bernoulli(p)
  - 판독: 마지막 열을 클래스 수만큼 행 구역으로 나누고 발화 수가 가장 많은 구역이 예측 (V1.4 방식)
  - 학습: 3-요소 규칙(도파민 × 자격 흔적) 하나만.
      자격 흔적 e = (s − p) · 입력      ← 뉴런 자신이 아는 값만 사용
      도파민    δ = r − r̄               ← 전역 보상(정답 1/오답 0)에서 최근 평균을 뺀 값
      Δw = lr · δ · e,   ΔV_base = −lr · δ · (s − p)
    역전파는 없다. 모든 뉴런이 같은 δ 하나를 받는다(전역 방송) → credit assignment는 2단계 과제.
    이 규칙은 확률적 이진 뉴런의 REINFORCE(Williams 1992)와 같아서, 기대 보상을 올리는 방향으로 평균적으로 움직인다.

일부러 넣지 않은 것: 화학 신호 공유, 생명게임, 체력, 수면, 허브, GA.
"""
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class BioGrid:
    def __init__(self, n_in, n_classes, height=40, width=8, w_scale=1.0, seed=0):
        assert height >= n_classes
        rng = np.random.RandomState(seed)
        self.n_in, self.n_classes, self.H, self.W = n_in, n_classes, height, width
        self.rng = rng
        # 1열(감각 열): 입력 전체 → H개 뉴런
        self.W_in = rng.randn(height, n_in) * w_scale / np.sqrt(max(1.0, n_in / 3))
        # 2열 ~ W열: 3방향 지역 연결. W_loc[k, r, c] = 열 c 뉴런 r이 왼쪽 열의 (r-1, r, r+1) 중 k번째에서 받는 가중치
        self.W_loc = rng.randn(3, height, width - 1) * w_scale
        self.V = np.zeros((height, width))  # 발화 문턱(기저값)
        # 판독 구역: 마지막 열의 행을 클래스 수만큼 나눔 (남는 행은 판독에 쓰지 않음)
        zone = height // n_classes
        self.zone_of_row = np.full(height, -1)
        for k in range(n_classes):
            self.zone_of_row[k * zone:(k + 1) * zone] = k
        self.reward_avg = 0.0
        self.last_silent = 0.0

    def n_params(self):
        return self.W_in.size + self.W_loc.size + self.V.size

    @staticmethod
    def _three_neighbors(col):
        """col: (B, H) → (3, B, H). 각 행 r에 대해 왼쪽 열의 r-1, r, r+1 값 (가장자리는 0)."""
        up = np.zeros_like(col); up[:, 1:] = col[:, :-1]
        down = np.zeros_like(col); down[:, :-1] = col[:, 1:]
        return np.stack([up, col, down])

    def forward(self, X, stochastic=True):
        """X: (B, n_in). 반환: 마지막 열 발화 (B, H)와 학습용 기록."""
        B = len(X)
        rec = []
        p = _sigmoid(X @ self.W_in.T - self.V[:, 0])
        s = (self.rng.rand(B, self.H) < p).astype(float) if stochastic else (p > 0.5).astype(float)
        rec.append((X, p, s))
        for c in range(1, self.W):
            nb = self._three_neighbors(s)
            p = _sigmoid(np.einsum("kbh,kh->bh", nb, self.W_loc[:, :, c - 1]) - self.V[:, c])
            s = (self.rng.rand(B, self.H) < p).astype(float) if stochastic else (p > 0.5).astype(float)
            rec.append((nb, p, s))
        return s, rec

    def readout(self, s_last):
        """구역별 발화 수 → 예측. 동점(무발화 포함)은 무작위로 깨고, 무발화 비율을 기록한다."""
        B = len(s_last)
        votes = np.zeros((B, self.n_classes))
        for k in range(self.n_classes):
            votes[:, k] = s_last[:, self.zone_of_row == k].sum(1)
        self.last_silent = float(np.mean(votes.sum(1) == 0))
        votes = votes + self.rng.rand(B, self.n_classes) * 1e-3  # 동점 무작위 처리
        return votes.argmax(1)

    def predict(self, X, n_votes=9):
        """확률적 발화를 n_votes번 돌려 다수결."""
        tally = np.zeros((len(X), self.n_classes))
        silent = 0.0
        for _ in range(n_votes):
            s, _ = self.forward(X)
            pred = self.readout(s)
            silent += self.last_silent
            tally[np.arange(len(X)), pred] += 1
        self.last_silent = silent / n_votes
        return (tally + self.rng.rand(*tally.shape) * 1e-3).argmax(1)

    def train_step(self, X, y, lr, reward_decay=0.99, output_signal="global", hidden_credit="global", gate="fired"):
        """output_signal / hidden_credit 조합으로 학습 신호를 고른다 (2단계 비교용).

        output_signal
          "global"      : 출력 열도 전역 도파민 δ × 자격 흔적 (1단계와 동일)
          "zone_target" : 출력 뉴런마다 방향 있는 오차 e = 목표 − p.
                          목표 = 정답 클래스 구역이면 1, 아니면 0  (설계서 c-7, GeoBrain V3.1의 클래스 구역 태그)
        hidden_credit
          "global"   : 은닉 열은 전역 도파민 δ × 자격 흔적 (1단계와 동일)
          "relay_w"  : 오차를 오른쪽 열에서 왼쪽 열로 이웃끼리 바톤 터치한다. 실제로 발화한 뉴런만 받는다
                       (= "데이터가 흘러간 라인에만" 보상, 좌표 ID 역추적의 목적). 전달 세기는 순방향 가중치 사용 (설계서 c-9)
          "relay_fa" : relay_w와 같으나 전달 세기에 고정 무작위 가중치 사용 (뒤 뉴런이 앞 연결 세기를 몰라도 되는 버전)
        gate (relay일 때 누가 바톤을 받는가)
          "fired"       : 실제로 발화한 뉴런만 ("흐른 라인에만"). 한 번 꺼진 뉴런은 신호를 못 받아 계속 죽어 있을 수 있다
          "sensitivity" : 문턱 근처라 발화가 바뀔 수 있는 뉴런일수록 많이 받는다, p(1-p)
        """
        if output_signal == "global" and hidden_credit == "global":
            return self._train_global(X, y, lr, reward_decay)
        return self._train_directed(X, y, lr, reward_decay, output_signal, hidden_credit, gate)

    def _train_directed(self, X, y, lr, reward_decay, output_signal, hidden_credit, gate):
        s_last, rec = self.forward(X)
        pred = self.readout(s_last)
        r = (pred == y).astype(float)
        delta = r - self.reward_avg
        self.reward_avg = reward_decay * self.reward_avg + (1 - reward_decay) * r.mean()
        B, last = len(X), self.W - 1

        # 열별 "학습 신호" g[c] (B, H): 이 값 × 입력이 가중치 변화량
        g = [None] * self.W
        _, p, s = rec[last]
        if output_signal == "zone_target":
            target = (self.zone_of_row[None, :] == y[:, None]).astype(float)
            e = (target - p) * (self.zone_of_row >= 0)[None, :]
        else:
            e = delta[:, None] * (s - p)
        g[last] = e

        for c in range(last, 0, -1):
            if hidden_credit == "global":
                _, p_prev, s_prev = rec[c - 1]
                g[c - 1] = delta[:, None] * (s_prev - p_prev)
                continue
            F = self.W_loc[:, :, c - 1] if hidden_credit == "relay_w" else self._feedback(c - 1)
            e = g[c]
            back = np.zeros_like(e)
            back[:, :-1] += F[0, 1:] * e[:, 1:]     # 나(r)의 신호를 "위 채널"로 받은 아래 이웃 r+1
            back[:, :] += F[1] * e                    # 나를 "옆 채널"로 받은 r
            back[:, 1:] += F[2, :-1] * e[:, :-1]     # 나를 "아래 채널"로 받은 위 이웃 r-1
            _, p_prev, s_prev = rec[c - 1]
            gate_val = s_prev if gate == "fired" else p_prev * (1 - p_prev) * 4.0
            g[c - 1] = np.clip(back * gate_val, -1.0, 1.0)

        x, _, _ = rec[0]
        self.W_in += lr * g[0].T @ x / B
        self.V[:, 0] -= lr * g[0].mean(0)
        for c in range(1, self.W):
            nb, _, _ = rec[c]
            self.W_loc[:, :, c - 1] += lr * np.einsum("bh,kbh->kh", g[c], nb) / B
            self.V[:, c] -= lr * g[c].mean(0)
        return r.mean()

    def _feedback(self, col):
        if not hasattr(self, "F_fa"):
            self.F_fa = np.random.RandomState(12345).randn(3, self.H, self.W - 1)
        return self.F_fa[:, :, col]

    def _train_global(self, X, y, lr, reward_decay):
        s_last, rec = self.forward(X)
        pred = self.readout(s_last)
        r = (pred == y).astype(float)
        delta = r - self.reward_avg                       # 도파민 = 보상 예측 오차 (B,)
        self.reward_avg = reward_decay * self.reward_avg + (1 - reward_decay) * r.mean()
        B = len(X)
        # 1열
        x, p, s = rec[0]
        g = delta[:, None] * (s - p)                      # (B, H)
        self.W_in += lr * g.T @ x / B
        self.V[:, 0] -= lr * g.mean(0)
        # 2열 ~
        for c in range(1, self.W):
            nb, p, s = rec[c]
            g = delta[:, None] * (s - p)
            self.W_loc[:, :, c - 1] += lr * np.einsum("bh,kbh->kh", g, nb) / B
            self.V[:, c] -= lr * g.mean(0)
        return r.mean()

    def features(self, X, n_samples=5):
        """대조군용: 격자 모든 뉴런의 평균 발화율 (B, H*W)."""
        acc = np.zeros((len(X), self.H * self.W))
        for _ in range(n_samples):
            _, rec = self.forward(X)
            acc += np.concatenate([r[2] for r in rec], axis=1)
        return acc / n_samples
