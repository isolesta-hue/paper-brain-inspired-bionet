"""기억 트랙 4-1: 연상 기억 (REFERENCE U1 — "지형을 기억으로").

두 가지 "지형"을 구분해 구현한다.

1) StateLandscapeMemory — 상태 지형 (홉필드형 에너지 지형)
   - 격자 H×W의 뉴런 N개. 패턴 = 어떤 뉴런들이 켜져 있는가 (희소 이진 벡터).
   - 저장(원샷): 패턴 하나를 한 번 보고 연결 W를 헤비안 공분산으로 조정 → 그 패턴이 에너지 골짜기가 된다.
       W += (ξ − f)(ξ − f)ᵀ ⊙ C      (C = 연결 마스크, 자기 연결 없음)
   - 회상: 단서를 넣고 반복 (L15 공명을 회상 동역학으로 용도 전환)
       h = W s,  s ← 상위 k개만 발화 (S8 GABA 상위 k% 억제를 회상의 억제 장치로 용도 전환), k = 패턴 활성 수
   - 연결 마스크 C:
       "full"        모든 뉴런 쌍 (교과서 홉필드)
       "local8"      격자 8이웃만 (설계서 b-1)
       "smallworld"  8이웃 + 뉴런마다 무작위 장거리 연결 n_long개 (설계서 b-1의 "가끔 먼 연결")
       "random"      같은 연결 수를 위치와 무관하게 무작위로 (smallworld의 대조군)

2) SpatialTerrainMemory — 공간 지형 (GeoBrain F1을 기억에 그대로 사용)
   - 격자 공간의 저항 H. 패턴(길이 W)을 맨 윗줄에 주입하면 에너지가 저항 낮은 곳으로 아래·좌·우로 흐른다 (18:75 TerrainCore 규칙).
   - 저장: 그 패턴의 에너지가 흐른 곳의 저항을 깎는다 (골짜기 파기).
   - 회상: 단서를 주입해 흐르게 한 뒤, 맨 아랫줄 에너지 상위 k칸을 회상 결과로 본다.
"""
import numpy as np


def grid_mask(H, W, kind, n_long=8, seed=0):
    N = H * W
    rng = np.random.RandomState(seed)
    rr, cc = np.divmod(np.arange(N), W)
    if kind == "full":
        C = np.ones((N, N))
    else:
        dr = np.abs(rr[:, None] - rr[None, :])
        dc = np.abs(cc[:, None] - cc[None, :])
        local = ((dr <= 1) & (dc <= 1)).astype(float)
        if kind == "local8":
            C = local
        elif kind == "smallworld":
            C = local.copy()
            for i in range(N):
                C[i, rng.choice(N, n_long, replace=False)] = 1.0
        elif kind == "random":
            per = int(local.sum(1).mean()) + n_long
            C = np.zeros((N, N))
            for i in range(N):
                C[i, rng.choice(N, per, replace=False)] = 1.0
        else:
            raise ValueError(kind)
        C = np.maximum(C, C.T)          # 대칭 연결 (에너지 지형이 존재하려면 대칭이어야 함)
    np.fill_diagonal(C, 0.0)
    return C


class StateLandscapeMemory:
    def __init__(self, H, W, kind="full", n_long=8, f=0.1, seed=0):
        self.N = H * W
        self.C = grid_mask(H, W, kind, n_long, seed)
        self.W = np.zeros((self.N, self.N))
        self.f = f
        self.k = max(1, int(round(f * self.N)))
        self.rng = np.random.RandomState(seed + 1)

    def n_connections(self):
        return int(self.C.sum())

    def store(self, xi):
        """패턴 하나를 원샷으로 저장 (xi: 0/1 벡터)."""
        d = xi - self.f
        self.W += np.outer(d, d) * self.C

    def recall(self, cue, steps=10, normalize=True):
        """normalize: 연결 수가 뉴런마다 다르므로 입력 합을 (연결된 이웃 중 켜진 수의 기대값)으로 나눠 공정하게 비교."""
        s = cue.astype(float).copy()
        deg = self.C.sum(1)
        for _ in range(steps):
            h = self.W @ s
            if normalize:
                h = h / np.maximum(1.0, self.C @ s)        # 켜진 이웃 하나당 평균 지지
            h = h + self.rng.rand(self.N) * 1e-9            # 동점 무작위 처리
            new = np.zeros(self.N)
            new[np.argpartition(h, -self.k)[-self.k:]] = 1.0
            if np.array_equal(new, s):
                break
            s = new
        return s


class SpatialTerrainMemory:
    """GeoBrain TerrainCore(18:75)의 흐름 규칙을 기억 저장에 사용."""

    def __init__(self, H, W, lr=0.3, steps=20, decay=0.9, seed=0):
        self.H, self.W, self.lr, self.steps, self.decay = H, W, lr, steps, decay
        self.Hd = np.full((H, W), 0.3)   # 아래로 가는 저항
        self.Hs = np.full((H, W), 1.2)   # 옆으로 가는 저항

    def _flow(self, top):
        E = np.zeros((self.H, self.W)); E[0] = top; trace = np.zeros_like(E)
        for _ in range(self.steps):
            down = np.maximum(0, E - self.Hd) * 0.4
            left = np.maximum(0, E - self.Hs) * 0.2
            right = np.maximum(0, E - self.Hs) * 0.2
            left[:, 0] = 0; right[:, -1] = 0; down[-1] = 0
            tot = down + left + right
            scale = np.where(tot > E, E / np.where(tot == 0, 1, tot), 1.0)
            down *= scale; left *= scale; right *= scale
            nE = (E - down - left - right) * self.decay
            nE[0] = E[0]
            nE[1:] += down[:-1]; nE[:, :-1] += left[:, 1:]; nE[:, 1:] += right[:, :-1]
            E = np.clip(nE, 0, 1)
            trace += E
        return E, trace

    def store(self, xi):
        _, trace = self._flow(xi)
        carve = self.lr * trace / (trace.max() + 1e-9)
        self.Hd = np.clip(self.Hd - carve, 0.01, 3.0)
        self.Hs = np.clip(self.Hs - 0.5 * carve, 0.01, 3.0)

    def recall(self, cue, k):
        E, _ = self._flow(cue)
        out = np.zeros(self.W)
        out[np.argpartition(E[-1] + np.random.rand(self.W) * 1e-9, -k)[-k:]] = 1.0
        return out


def make_patterns(P, N, f, rng):
    k = max(1, int(round(f * N)))
    X = np.zeros((P, N))
    for i in range(P):
        X[i, rng.choice(N, k, replace=False)] = 1.0
    return X


def make_cue(xi, keep, noise_on, rng):
    """활성 비트 중 keep 비율만 남기고, 꺼진 자리 중 noise_on개를 잘못 켠다."""
    on = np.flatnonzero(xi); off = np.flatnonzero(xi == 0)
    cue = np.zeros_like(xi)
    cue[rng.choice(on, max(1, int(round(keep * len(on)))), replace=False)] = 1.0
    if noise_on > 0:
        cue[rng.choice(off, noise_on, replace=False)] = 1.0
    return cue


def overlap(a, b):
    """회상 결과 a가 원래 패턴 b의 활성 비트를 얼마나 되찾았나 (0~1)."""
    return float((a * b).sum() / max(1.0, b.sum()))
