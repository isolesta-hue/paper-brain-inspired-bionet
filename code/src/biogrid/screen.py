"""기능 검증(스크리닝)용 BioGrid 확장판.

core.BioGrid를 그대로 두고(실험 01·02 재현성), 참고 자료(docs/REFERENCE.md)의 메커니즘을
`features` 딕셔너리로 하나씩 켤 수 있게 만든 버전. features={} 이면 core의 A0~A4와 같은 계산을 한다
(같은 시드에서 같은 숫자가 나오는지 tests로 확인).

각 기능 키와 REFERENCE ID:
  열 계산(순전파) 단계
    resonance   L15  각 열에서 T번 발화 시도 → 평균 발화율을 다음 열로 (미세 스텝 / 율 부호)
    lif         N7   미세 스텝 동안 막전위 누적(누수 0.8) + 문턱 넘으면 발화·리셋, 발화율 전달
    mean_inhib  S9   같은 열 평균 입력을 빼는 억제 (gaba 세기)
    wave40      S10  40Hz 사인파를 입력에 더함
    roles       N8   뉴런마다 역할 5종 무작위 배정, 입력에 ×1.1 / ×0.9 / +이전열 같은 행 0.3 / ×0.5 / ×1
    receptor    N9   이웃 발화와 수용체 패턴 내적 ≥ 문턱일 때만 입력 수용, 오답 시 막힌 채널 탐색 (L17 포함)
    gol_thresh  S4   이웃 발화 수(같은 열 위아래 + 왼쪽 열 3)로 문턱 조정: 많으면 ↓, 적으면 ↑
    chem        S5   같은 열 이웃이 강하게 발화하면 도파민(문턱↓), 침묵하면 억제(입력 감소)
    topk        S8   열마다 발화 확률 상위 k%만 발화 허용 (GABA 희소화). cols = 적용할 열 (None = 전부), gate = 이긴 뉴런만 학습
    gol_filter  S3   고립 발화 제거, 과밀 발화 약화 (뉴런은 죽이지 않음)
    hub         S15  가운데 열을 "허브 열"로: 왼쪽 열 전체와 연결 (장거리 연결)
    residual    –    열 c가 두 칸 왼쪽(c-2) 같은 행 발화도 받음 (Transformer residual, REFERENCE 10장)
    prob_readout N11 판독 때 발화 수 대신 발화 확률 합으로 구역 비교
  학습 후(느린 상태) 단계
    fatigue     N4   자주 발화한 뉴런의 문턱이 올라가고 서서히 회복
    homeostasis N5   뉴런별 발화율을 목표(0.2)로 문턱 조정
    homeo_hidden N5  숨은 열만 발화율을 목표(0.5, 가장 민감한 곳)로 — 임계 근처 유지 (NH 33)
    astrocyte   S13  전역 발화율 <5%면 모든 문턱↓, >60%면 ↑
    v_share     S7   문턱을 같은 열 위아래 이웃과 조금씩 평균 (화학 가중치 공유)
    long_short  N6   가중치 = 장기 + 단기, 단기는 빨리 배우고 빨리 잊음
    sleep       L14  변화량을 태그에 쌓았다가 K배치마다 한 번에 반영 (수면기)
    vitality    M6+M10 계속 침묵·포화하는 뉴런의 체력↓, 0이면 가장 건강한 위아래 이웃 가중치 복제+변이로 교체
    gol_death   S2   위아래 이웃이 모두 침묵/포화한 뉴런을 영구 사멸 (형태학적 생명게임)
    dyn_conn    M9   3방향 연결에 존재 마스크. 주기적으로 약한 연결 제거·무작위 새 연결 생성
    terrain     F6   가중치 고정, 문턱(V)만 학습 = "지형을 깎는" 모델
  전역 도파민(A0) 학습 규칙 변형 — hidden_credit="global"일 때만 의미
    hebb_pm1    L2   v5.2 규칙: Δw = lr·(맞으면 +1, 틀리면 −1)·발화·입력 (기준선 차감·탐색항 없음)
    dopa_asym   L4   도파민 샤워: 오답 도파민을 절반 크기로 (맞음 +, 틀림 −0.5배)
    conn_tag    L13  연결 단위 도파민 태그: 자격 흔적을 배치 간 감쇠 누적(0.9)해서 사용
"""
import numpy as np

from .core import BioGrid, _sigmoid

DEFAULTS = {
    "resonance": {"T": 5},
    "lif": {"T": 5, "leak": 0.8},
    "mean_inhib": {"gaba": 0.5},
    "wave40": {"amp": 0.05},
    "roles": {},
    "receptor": {"lr": 0.02, "thresh": 0.2},
    "gol_thresh": {"scale": 0.1},
    "chem": {"dopa": 0.08, "inhib": 0.15},
    "topk": {"k": 0.15},
    "gol_filter": {"gaba": 0.5},
    "hub": {},
    "residual": {},
    "prob_readout": {},
    "fatigue": {"decay": 0.9, "gain": 0.1},
    "homeostasis": {"lr": 0.01, "target": 0.2},
    "astrocyte": {"step": 0.05},
    "v_share": {"mix": 0.1},
    "long_short": {"short_lr": 1.0, "long_lr": 0.1, "short_decay": 0.95},
    "sleep": {"K": 10},
    "vitality": {"loss": 5.0, "gain": 1.0},
    "gol_death": {"min_alive": 0.6},
    "dyn_conn": {"density": 0.7, "every": 50, "frac": 0.05},
    "terrain": {},
    "hebb_pm1": {},
    "dopa_asym": {},
    "conn_tag": {"decay": 0.9},
    "fan": {"k": 3},
    "homeo_hidden": {"lr": 0.01, "target": 0.5},
}


class BioGridX(BioGrid):
    def __init__(self, n_in, n_classes, height=40, width=8, seed=0, features=None):
        super().__init__(n_in, n_classes, height=height, width=width, seed=seed)
        self.f = {}
        for k, v in (features or {}).items():
            cfg = dict(DEFAULTS[k]); cfg.update(v or {})
            self.f[k] = cfg
        # 기능 초기화는 core가 rng를 다 쓴 뒤 별도 rng로 (features={}일 때 core와 같은 난수열 유지)
        frng = np.random.RandomState(seed + 777)
        H, W = height, width
        self.hub_col = W // 2 if ("hub" in self.f and W >= 3) else -1
        self.hub_cols = list(self.f["hub"].get("cols") or [self.hub_col]) if self.hub_col > 0 else []   # NH 40: 허브 여러 개
        if self.hub_cols:
            self.hub_col = self.hub_cols[0]
        if len(self.hub_cols) == 1:
            self.W_hub = frng.randn(H, H) / np.sqrt(H / 3)
        elif len(self.hub_cols) > 1:
            self.W_hub = frng.randn(len(self.hub_cols), H, H) / np.sqrt(H / 3)
        self.M_hub = None
        conn = self.f["hub"].get("conn") if self.hub_cols else None
        if conn:                                   # NH 41: 허브 뉴런마다 앞 열과의 연결 수 m, 길이 = band(가까운 행 m개) / rand(아무 행 m개)
            kind, m = conn
            crng = np.random.RandomState(seed + 999)
            M = np.zeros((H, H))
            frac = None
            if kind.startswith("mix"):                # NH 61: 행(뉴런)마다 섞기 — 비율 frac의 뉴런만 무작위 긴 연결 m개, 나머지는 가까운 7행 (흩어진 허브 뉴런)
                frac = float(kind[3:]) / 100
            hub_rows = None
            if frac is not None:
                n_hub = len(self.hub_cols) if self.W_hub.ndim == 3 else 1
                M = np.zeros((n_hub, H, H))
                hub_rows = crng.rand(n_hub, H) < frac
                for j in range(n_hub):
                    for r in range(H):
                        if hub_rows[j, r]:
                            M[j, r, crng.choice(H, m, replace=False)] = 1.0
                        else:
                            lo = min(max(0, r - 3), H - 7)
                            M[j, r, lo:lo + 7] = 1.0
                scale = np.where(hub_rows, np.sqrt(H / m), np.sqrt(H / 7))[..., None]
                self.M_hub = M if self.W_hub.ndim == 3 else M[0]
                self.W_hub = self.W_hub * self.M_hub * (scale if self.W_hub.ndim == 3 else scale[0])
            else:
                for r in range(H):
                    if kind == "band":
                        lo = min(max(0, r - m // 2), H - m)
                        M[r, lo:lo + m] = 1.0
                    else:
                        M[r, crng.choice(H, m, replace=False)] = 1.0
                self.M_hub = M if self.W_hub.ndim == 2 else np.broadcast_to(M, self.W_hub.shape).copy()
                self.W_hub = self.W_hub * self.M_hub * np.sqrt(H / m)   # 처음 입력 크기를 전체 연결과 같게
        if "residual" in self.f:
            self.W_res = frng.randn(H, W) * 0.5
        if "roles" in self.f:
            self.role = frng.randint(0, 5, size=(H, W))
        if "fan" in self.f and self.f["fan"]["k"] != 3:
            k = self.f["fan"]["k"]
            assert k % 2 == 1
            # 지역 연결을 위아래 k//2칸까지 (k개 이웃). 처음 입력 크기를 3이웃과 같게: 표준편차 × sqrt(3/k)
            self.W_loc = frng.randn(k, H, W - 1) * np.sqrt(3.0 / k)
        if "receptor" in self.f:
            self.R = frng.uniform(-1, 1, (3, H, W))
            self.R_th = np.full((H, W), self.f["receptor"]["thresh"])
        if "long_short" in self.f:
            self.W_in_s = np.zeros_like(self.W_in); self.W_loc_s = np.zeros_like(self.W_loc)
        if "dyn_conn" in self.f:
            self.M_conn = (frng.rand(*self.W_loc.shape) < self.f["dyn_conn"]["density"]).astype(float)
            self.conn_usage = np.zeros_like(self.W_loc)
        self.fatigue = np.zeros((H, W))
        self.vit = np.full((H, W), 100.0)
        self.alive = np.ones((H, W))
        self.rate_ema = np.full((H, W), 0.5)
        self.tags = None
        self.etag = None
        self.n_batches = 0
        self.t = 0
        self.frng = frng
        self.replaced = 0

    def _three_neighbors(self, col):
        """3이웃 (기본) 또는 fan k이웃: 결과[j]의 행 r = 왼쪽 열의 행 r + (j − k//2) (가장자리는 0)."""
        k = self.f["fan"]["k"] if hasattr(self, "f") and "fan" in self.f else 3
        if k == 3:
            return BioGrid._three_neighbors(col)
        out = np.zeros((k,) + col.shape)
        for j in range(k):
            d = j - k // 2
            if d < 0:
                out[j][:, -d:] = col[:, :d]
            elif d > 0:
                out[j][:, :-d] = col[:, d:]
            else:
                out[j] = col
        return out

    def n_params(self):
        n = super().n_params()
        if self.hub_col > 0:
            n += self.W_hub.size - len(self.hub_cols) * 3 * self.H   # 허브 열은 3방향 대신 전체 연결
        if "residual" in self.f:
            n += self.W_res.size
        if "receptor" in self.f:
            n += self.R.size + self.R_th.size
        return n

    # ------------------------------------------------------------------ 순전파
    def _w_in(self):
        return self.W_in + (self.W_in_s if "long_short" in self.f else 0)

    def _w_loc(self, c):
        w = self.W_loc[:, :, c - 1] + (self.W_loc_s[:, :, c - 1] if "long_short" in self.f else 0)
        if "dyn_conn" in self.f:
            w = w * self.M_conn[:, :, c - 1]
        return w

    def _fire(self, I, Vc, B, stochastic):
        """확률/발화 계산. 공명·LIF면 미세 스텝 평균 발화율을 반환."""
        p = _sigmoid(getattr(self, "gain", 1.0) * (I - Vc))   # gain: 노르아드레날린 (NH 38, 기본 1 = 예전과 같음)
        if "resonance" in self.f and stochastic:
            T = self.f["resonance"]["T"]
            s = (self.rng.rand(T, B, self.H) < p[None]).mean(0)
        elif "lif" in self.f:
            T, leak = self.f["lif"]["T"], self.f["lif"]["leak"]
            u = np.zeros_like(I); cnt = np.zeros_like(I)
            drive = I - Vc
            for _ in range(T):
                u = leak * u + drive + (self.rng.randn(*I.shape) * 0.5 if stochastic else 0)
                spk = (u > 1.0).astype(float); cnt += spk; u = u * (1 - spk)
            s = cnt / T
        elif stochastic:
            s = (self.rng.rand(B, self.H) < p).astype(float)
        else:
            s = (p > 0.5).astype(float)
        return p, s

    def forward(self, X, stochastic=True):
        B, H = len(X), self.H
        f = self.f
        rec, hist = [], []
        tvec = self.t + np.arange(B)
        for c in range(self.W):
            if c == 0:
                inputs = {"x": X}
                I = X @ self._w_in().T
            elif c in self.hub_cols:
                j = self.hub_cols.index(c)
                inputs = {"dense": hist[c - 1], "hub_j": j}
                I = hist[c - 1] @ self._whub(j).T
            else:
                nb = self._three_neighbors(hist[c - 1])
                inputs = {"nb": nb}
                I = np.einsum("kbh,kh->bh", nb, self._w_loc(c))
            if "residual" in f and c >= 2:
                inputs["res"] = hist[c - 2]
                I = I + hist[c - 2] * self.W_res[:, c]
            if "receptor" in f and "nb" in inputs:
                aff = np.einsum("kbh,kh->bh", inputs["nb"], self.R[:, :, c])
                acc = (aff >= self.R_th[:, c]).astype(float)
                inputs["acc"] = acc
                I = I * acc
            if "roles" in f and c >= 1:
                r = self.role[:, c]
                I = np.where(r == 0, I * 1.1, I); I = np.where(r == 1, I * 0.9, I)
                I = np.where(r == 2, I + 0.3 * hist[c - 1], I); I = np.where(r == 3, I * 0.5, I)
            if "mean_inhib" in f:
                I = I - f["mean_inhib"]["gaba"] * I.mean(1, keepdims=True)
            if "wave40" in f:
                I = I + f["wave40"]["amp"] * np.sin(2 * np.pi * 40 * (tvec / 5000.0))[:, None]
            Vc = self.V[:, c] + self.fatigue[:, c]
            if "gol_thresh" in f or "chem" in f:
                p0 = _sigmoid(I - Vc)
                if "gol_thresh" in f:
                    cnt = np.zeros_like(p0)
                    cnt[:, 1:] += p0[:, :-1]; cnt[:, :-1] += p0[:, 1:]
                    if c >= 1:
                        cnt += self._three_neighbors(hist[c - 1]).sum(0)
                    g = f["gol_thresh"]["scale"]
                    Vc = Vc + np.where(cnt >= 3, -g, np.where(cnt <= 1, g, 0.0))
                if "chem" in f:
                    nbp = np.zeros_like(p0); nbp[:, 1:] += p0[:, :-1]; nbp[:, :-1] += p0[:, 1:]
                    dopa = (nbp > 1.0).astype(float); inh = (nbp < 0.2).astype(float)
                    Vc = Vc - f["chem"]["dopa"] * dopa
                    I = I * (1 - f["chem"]["inhib"] * inh)
            p, s = self._fire(I, Vc, B, stochastic)
            if "topk" in f and (f["topk"].get("cols") is None or c in f["topk"]["cols"]):
                k = max(1, int(H * f["topk"]["k"]))
                kth = np.partition(p, -k, axis=1)[:, -k][:, None]
                win = (p >= kth).astype(float)
                s = s * win
                if f["topk"].get("gate"):
                    inputs["win"] = win          # 이긴 뉴런만 학습 (뉴호라이즌스 NH 19: 규칙마다 다른 소수 뉴런이 배움)
            if "gol_filter" in f:
                act = (s > 0).astype(float)
                nbc = np.zeros_like(act); nbc[:, 1:] += act[:, :-1]; nbc[:, :-1] += act[:, 1:]
                if c >= 1:
                    nbc += (self._three_neighbors(hist[c - 1]) > 0).sum(0)
                s = np.where((act > 0) & (nbc == 0), 0.0, s)
                s = np.where((act > 0) & (nbc >= 4), s * (1 - f["gol_filter"]["gaba"]), s)
            s = s * self.alive[:, c]
            rec.append((inputs, p, s))
            hist.append(s)
        return hist[-1], rec

    def readout(self, s_last, p_last=None):
        if "prob_readout" in self.f and p_last is not None:
            s_last = p_last
        return super().readout(s_last)

    def predict(self, X, n_votes=9):
        tally = np.zeros((len(X), self.n_classes)); silent = 0.0
        for _ in range(n_votes):
            s, rec = self.forward(X)
            pred = self.readout(s, rec[-1][1])
            silent += self.last_silent
            tally[np.arange(len(X)), pred] += 1
        self.last_silent = silent / n_votes
        return (tally + self.rng.rand(*tally.shape) * 1e-3).argmax(1)

    # ------------------------------------------------------------------ 학습
    def train_step(self, X, y, lr, reward_decay=0.99, output_signal="zone_target",
                   hidden_credit="relay_w", gate="sensitivity"):
        s_last, rec = self.forward(X)
        pred = self.readout(s_last, rec[-1][1])
        # y가 2차원(B x H)이면 마지막 열이 그려 낼 목표 값 자체 (특징 판독, 뉴호라이즌스): 정답/오답 보상은 없음
        r = np.zeros(len(y)) if np.ndim(y) == 2 else (pred == y).astype(float)
        delta = r - self.reward_avg
        self.reward_avg = reward_decay * self.reward_avg + (1 - reward_decay) * r.mean()
        if "dopa_asym" in self.f:
            delta = np.where(delta < 0, 0.5 * delta, delta)
        if "hebb_pm1" in self.f:
            delta = np.where(r > 0, 1.0, -1.0)
        B, last, f = len(X), self.W - 1, self.f
        g = [None] * self.W
        _, p, s = rec[last]
        if output_signal == "zone_target":
            target = y if np.ndim(y) == 2 else (self.zone_of_row[None, :] == y[:, None]).astype(float)
            g[last] = (target - p) * (self.zone_of_row >= 0)[None, :]
        else:
            g[last] = delta[:, None] * (s if "hebb_pm1" in f else (s - p))

        back = [np.zeros((B, self.H)) for _ in range(self.W)]
        for c in range(last, 0, -1):
            if hidden_credit == "global":
                _, pp, sp = rec[c - 1]
                g[c - 1] = delta[:, None] * (sp if "hebb_pm1" in f else (sp - pp))
                continue
            inputs, _, _ = rec[c]
            e = g[c]
            if "acc" in inputs:
                e = e * inputs["acc"]
            if "dense" in inputs:
                F = self._whub(inputs["hub_j"]) if hidden_credit == "relay_w" else self._fb_dense()
                back[c - 1] += e @ F
            else:
                F = self._w_loc(c) if hidden_credit == "relay_w" else self._feedback(c - 1)
                kk = len(F)
                if kk == 3:
                    back[c - 1][:, :-1] += F[0, 1:] * e[:, 1:]
                    back[c - 1] += F[1] * e
                    back[c - 1][:, 1:] += F[2, :-1] * e[:, :-1]
                else:                                    # fan: 열 c의 행 r이 왼쪽 행 r + d에서 받음 → 그 행으로 오차를 되돌림
                    for j in range(kk):
                        d = j - kk // 2
                        contrib = F[j] * e
                        if d < 0:
                            back[c - 1][:, :d] += contrib[:, -d:]
                        elif d > 0:
                            back[c - 1][:, d:] += contrib[:, :-d]
                        else:
                            back[c - 1] += contrib
            if "res" in inputs:
                back[c - 2] += e * self.W_res[:, c]
            _, pp, sp = rec[c - 1]
            gate_val = sp if gate == "fired" else pp * (1 - pp) * 4.0
            if "win" in rec[c - 1][0]:
                gate_val = gate_val * rec[c - 1][0]["win"]
            g[c - 1] = np.clip(back[c - 1] * gate_val * self.alive[:, c - 1], -1.0, 1.0)

        # 가중치 변화량 계산
        dW_in = g[0].T @ rec[0][0]["x"] / B
        dW_loc = np.zeros_like(self.W_loc)
        dV = np.stack([-g[c].mean(0) for c in range(self.W)], 1)
        extra = {}
        for c in range(1, self.W):
            inputs = rec[c][0]
            ec = g[c] * inputs["acc"] if "acc" in inputs else g[c]
            if "dense" in inputs:
                dh = ec.T @ inputs["dense"] / B
                if self.W_hub.ndim == 2:
                    extra["W_hub"] = dh
                else:
                    extra.setdefault("W_hub", np.zeros_like(self.W_hub))[inputs["hub_j"]] = dh
            else:
                dW_loc[:, :, c - 1] = np.einsum("bh,kbh->kh", ec, inputs["nb"]) / B
            if "res" in inputs:
                extra.setdefault("W_res", np.zeros_like(self.W_res))[:, c] = (ec * inputs["res"]).mean(0)
        if "conn_tag" in f:
            dec = f["conn_tag"]["decay"]
            if self.etag is None:
                self.etag = [np.zeros_like(dW_in), np.zeros_like(dW_loc), np.zeros_like(dV)]
            self.etag = [dec * a + b for a, b in zip(self.etag, [dW_in, dW_loc, dV])]
            dW_in, dW_loc, dV = [t * (1 - dec) for t in self.etag]
        if "dyn_conn" in f:
            dW_loc = dW_loc * self.M_conn
            self.conn_usage = 0.99 * self.conn_usage + np.abs(dW_loc)
        self._apply(lr, dW_in, dW_loc, dV, extra)
        if "receptor" in f:
            self._update_receptor(rec, r, f["receptor"]["lr"])
        self._slow_updates(rec)
        self.t += B
        self.n_batches += 1
        return r.mean()

    def _whub(self, j):
        return self.W_hub if self.W_hub.ndim == 2 else self.W_hub[j]

    def _fb_dense(self):
        if not hasattr(self, "F_hub_fa"):
            self.F_hub_fa = np.random.RandomState(54321).randn(self.H, self.H) / np.sqrt(self.H / 3)
        return self.F_hub_fa

    def _apply(self, lr, dW_in, dW_loc, dV, extra):
        f = self.f
        if "sleep" in f:
            if self.tags is None:
                self.tags = {"W_in": 0, "W_loc": 0, "V": 0, **{k: 0 for k in extra}}
            self.tags["W_in"] = self.tags["W_in"] + dW_in
            self.tags["W_loc"] = self.tags["W_loc"] + dW_loc
            self.tags["V"] = self.tags["V"] + dV
            for k, v in extra.items():
                self.tags[k] = self.tags.get(k, 0) + v
            if (self.n_batches + 1) % f["sleep"]["K"] != 0:
                return
            # 원본 V1.4처럼 쌓인 태그 합계를 그대로 반영 (K로 나누면 학습 속도만 1/K가 되어 불공정)
            dW_in, dW_loc, dV = self.tags["W_in"], self.tags["W_loc"], self.tags["V"]
            extra = {k: self.tags[k] for k in extra}
            self.tags = None
        if "terrain" not in f:
            if "long_short" in f:
                ls = f["long_short"]
                self.W_in_s = ls["short_decay"] * self.W_in_s + lr * ls["short_lr"] * dW_in
                self.W_loc_s = ls["short_decay"] * self.W_loc_s + lr * ls["short_lr"] * dW_loc
                self.W_in += lr * ls["long_lr"] * dW_in
                self.W_loc += lr * ls["long_lr"] * dW_loc
            else:
                self.W_in += lr * dW_in
                self.W_loc += lr * dW_loc
            if "W_hub" in extra:
                self.W_hub += lr * extra["W_hub"]
                if self.M_hub is not None:
                    self.W_hub *= self.M_hub
            if "W_res" in extra:
                self.W_res += lr * extra["W_res"]
        self.V += lr * dV

    def _update_receptor(self, rec, r, rlr):
        for c in range(1, self.W):
            inputs, _, s = rec[c]
            if "acc" not in inputs:
                continue
            nb, acc = inputs["nb"], inputs["acc"]
            active = (s > 0).astype(float)
            ok = r[:, None]
            for k in range(3):
                self.R[k, :, c] += rlr * (nb[k] * active * acc * ok).mean(0)
                self.R[k, :, c] += rlr * (nb[k] * active * (1 - acc) * (1 - ok)).mean(0)
            self.R_th[:, c] = np.clip(self.R_th[:, c] - 0.1 * rlr * (active * (1 - acc) * (1 - ok)).mean(0), 0.01, 1.0)

    def _slow_updates(self, rec):
        f = self.f
        rate = np.stack([r[2].mean(0) for r in rec], 1)          # (H, W) 이번 배치 발화율
        self.rate_ema = 0.95 * self.rate_ema + 0.05 * rate
        if "fatigue" in f:
            self.fatigue = f["fatigue"]["decay"] * self.fatigue + f["fatigue"]["gain"] * rate
        if "homeostasis" in f:
            self.V += f["homeostasis"]["lr"] * (rate - f["homeostasis"]["target"])
        if "homeo_hidden" in f:                     # NH 33: 숨은 열만 발화율을 목표(0.5 = 시그모이드가 가장 민감한 곳)로 — 출력 열(구역 판독)은 그대로
            hh = f["homeo_hidden"]
            self.V[:, :-1] += hh["lr"] * (rate[:, :-1] - hh["target"])
        if "astrocyte" in f:
            gr = rate.mean()
            self.V += f["astrocyte"]["step"] * (float(gr > 0.6) - float(gr < 0.05))
        if "v_share" in f:
            nbm = np.zeros_like(self.V); cnt = np.zeros_like(self.V)
            nbm[1:] += self.V[:-1]; cnt[1:] += 1; nbm[:-1] += self.V[1:]; cnt[:-1] += 1
            mix = f["v_share"]["mix"]
            self.V = (1 - mix) * self.V + mix * nbm / cnt
        if "vitality" in f:
            bad = (self.rate_ema < 0.02) | (self.rate_ema > 0.98)
            self.vit += np.where(bad, -f["vitality"]["loss"], f["vitality"]["gain"])
            self.vit = np.clip(self.vit, 0, 100)
            dead = np.argwhere(self.vit <= 0)
            for r_, c_ in dead:
                cand = [(rr, c_) for rr in (r_ - 1, r_ + 1) if 0 <= rr < self.H]
                pr, pc = max(cand, key=lambda t: self.vit[t])
                if c_ == 0:
                    self.W_in[r_] = self.W_in[pr] + self.frng.randn(self.n_in) * 0.05
                else:
                    self.W_loc[:, r_, c_ - 1] = self.W_loc[:, pr, c_ - 1] + self.frng.randn(len(self.W_loc)) * 0.05
                self.V[r_, c_] = self.V[pr, c_]
                self.vit[r_, c_] = 50.0
                self.rate_ema[r_, c_] = 0.5
                self.replaced += 1
        if "gol_death" in f and self.n_batches % 50 == 49:
            ra = self.rate_ema
            up = np.vstack([np.full((1, self.W), 0.5), ra[:-1]]); dn = np.vstack([ra[1:], np.full((1, self.W), 0.5)])
            lonely = (up < 0.05) & (dn < 0.05); crowded = (up > 0.95) & (dn > 0.95)
            die = (lonely | crowded) & (self.alive > 0)
            die[:, -1] = False                                       # 출력 열 보호
            room = int(self.alive.sum() - f["gol_death"]["min_alive"] * self.alive.size)
            idx = np.argwhere(die)[:max(0, room)]
            for r_, c_ in idx:
                self.alive[r_, c_] = 0.0
        if "dyn_conn" in f and self.n_batches % f["dyn_conn"]["every"] == f["dyn_conn"]["every"] - 1:
            on = np.argwhere(self.M_conn > 0); off = np.argwhere(self.M_conn == 0)
            n = min(int(len(on) * f["dyn_conn"]["frac"]), len(off))
            if n > 0:
                score = self.conn_usage[tuple(on.T)] + np.abs(self.W_loc[tuple(on.T)]) * 1e-3
                weakest = on[np.argsort(score)[:n]]
                self.M_conn[tuple(weakest.T)] = 0
                grow = off[self.frng.choice(len(off), n, replace=False)]
                self.M_conn[tuple(grow.T)] = 1
                self.W_loc[tuple(grow.T)] = self.frng.randn(n) * 0.1
