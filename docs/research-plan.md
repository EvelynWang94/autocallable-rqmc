# Applied Project Plan

## Fast and Stable Monte Carlo Pricing and Delta Estimation for a Three-Asset Autocallable

**Working subtitle:** A comparative study of randomized quasi-Monte Carlo and survival conditioning under correlated geometric Brownian motion

**核心一句话：**

> European option 是验证工具，barrier option 是研究对象，variance reduction 是比较对象，sensitivity estimation 是应用落点，accuracy–stability–runtime trade-off 是最终结论。

---

## 1. 项目定位

### 1.1 研究对象

本项目研究一个以三项资产为标的、包含定期自动赎回、条件票息和下行敲入保护的 worst-of autocallable。基础资产在风险中性测度下服从 correlated geometric Brownian motion（correlated GBM）。

论文的核心不是证明 GBM 是最真实的市场模型，而是将模型复杂度固定在一个透明、可验证的层次，使研究重点落在：

1. 不连续、路径依赖 payoff 对 Monte Carlo 收敛和 Greeks 稳定性的影响；
2. randomized quasi-Monte Carlo（RQMC）能否降低定价误差；
3. survival conditioning 能否通过平滑 barrier indicator 改善价格与 Delta；
4. 两种方法结合后是否存在额外收益；
5. 方法的优势在哪些 volatility、correlation 和 barrier regimes 下成立；
6. 更快的算法是否真的在相同精度下更快，而不只是单次运行时间更短。

### 1.2 建议论文题目

> **Fast and Stable Monte Carlo Pricing and Delta Estimation for Multi-Asset Autocallable Notes**
>
> *A Comparative Study of Randomized Quasi-Monte Carlo and Survival Conditioning under Correlated GBM*

### 1.3 学术定位与可主张的贡献

Survival conditioning、autocallable payoff smoothing、RQMC 和 Monte Carlo Greeks 均已有成熟研究。因此，本项目不应声称“首次提出”这些方法。

更可信的硕士论文贡献是：

- 在统一代码和相同计算预算下，比较 standard MC、RQMC、survival-conditioned MC 和 RQMC + survival conditioning；
- 将比较从价格估计延伸到三项 component Delta 的稳定性；
- 用 payoff decomposition 将 survival conditioning 集中应用于 autocallable 的 downside protection leg；
- 研究 correlation、volatility、barrier proximity 和 monitoring frequency 如何改变不同方法的相对表现；
- 以 RMSE、Delta dispersion、wall-clock time 和 time-to-accuracy 共同评价方法；
- 给出方法有效和失效的条件，而不是只报告某个参数组合下的加速倍数。

如果希望增加一个小而明确的 distinction-level contribution，可以在核心完成后增加：

> **Barrier-aware conditioning order:** 比较固定资产顺序与按标准化 barrier distance 排序的 sequential conditioning，并分析该顺序对方差、Delta 和 RQMC 有效维度的影响。

这比立即加入 Heston 或 GPU 更聚焦，也更容易形成一条独立结论。

---

## 2. 研究问题与假设

### 2.1 主研究问题

> 对具有多个不连续触发条件的三资产 autocallable，RQMC 与 survival conditioning 能否单独或联合降低达到给定价格和 Delta 精度所需的计算时间？

### 2.2 子问题

**RQ1 — Validation**

在 European option 上，MC 和 RQMC engine 是否无系统性实现错误，置信区间和经验收敛是否合理？

**RQ2 — Barrier discontinuity**

当现价接近 barrier 时，payoff discontinuity 对 estimator variance 和 Delta stability 的损害有多大？

**RQ3 — Pricing efficiency**

对三资产 autocallable，RQMC、survival conditioning 及其组合分别能带来多大的 RMSE 和 time-to-accuracy 改善？

**RQ4 — Sensitivity estimation**

在 common random numbers 下，survival conditioning 是否能够降低 central finite-difference Delta 对 bump size 和 random seed 的敏感度？

**RQ5 — Regime dependence**

方法排名是否随 correlation、volatility、knock-in barrier、autocall barrier 和 monitoring frequency 改变？

### 2.3 可检验假设

- **H1:** European option 上，MC error 大致呈 \(N^{-1/2}\) 下降，scrambled Sobol RQMC 的经验 RMSE 下降更快。
- **H2:** 当 barrier 离现价较近时，standard MC 的价格方差和 Delta 不稳定性显著增加。
- **H3:** Survival conditioning 对 barrier-driven downside leg 的改善大于对包含多个 autocall/coupon trigger 的完整 payoff 的改善。
- **H4:** RQMC + survival conditioning 通常拥有最低的 work-normalised variance，但不会在所有 regimes 下占优。
- **H5:** Delta 的相对方法排名比 price 更能暴露 estimator 的实际可用性。
- **H6:** 在高维、强不连续或极端相关性状态下，RQMC 的边际收益可能下降；这是应报告的边界，而不是失败。

---

## 3. 核心范围与扩展边界

### 3.1 必须完成的核心

- 三资产 correlated GBM；
- European call validation；
- 单资产 barrier option bridge experiment；
- 三资产 worst-of autocallable；
- standard MC；
- scrambled Sobol RQMC；
- survival conditioning；
- RQMC + survival conditioning；
- 三项 component Delta；
- accuracy、stability 和 runtime 的统一比较；
- 至少三个有经济解释的 parameter regimes。

### 3.2 核心完成后的扩展顺序

1. **CPU optimisation / JIT**：确认算法结果不变，只改变执行效率；
2. **更多 Greeks**：先 Vega，再 Gamma；Gamma 对不连续 payoff 更困难；
3. **GPU**：单独评价 hardware acceleration；
4. **Heston**：最后加入 stochastic volatility，并重新处理 calibration、discretisation bias 和 RQMC dimension。

### 3.3 明确不放入核心的内容

- local volatility、stochastic local volatility；
- jump diffusion；
- stochastic interest rates；
- issuer credit risk / XVA；
- dynamic hedging P&L backtest；
- machine-learning surrogate；
- full production calibration。

这些内容会稀释论文的 numerical question。Heston、GPU 和更多 Greeks 只能在核心实验和主要图表已经冻结后开始。

---

## 4. 产品定义

### 4.1 状态变量

设三项资产为 \(S_1,S_2,S_3\)，合同初始参考水平为 \(K_i\)。定义：

\[
R_{i,t}=\frac{S_{i,t}}{K_i},\qquad
W_t=\min_{i=1,2,3}R_{i,t}.
\]

\(W_t\) 是 worst-of performance。

### 4.2 Baseline term sheet

以下参数是研究用的 stylised baseline，应全部存入配置文件，而不是写死在代码中。

| 条款 | Baseline |
|---|---:|
| Notional | 100 |
| Maturity | 3 years |
| Autocall observations | Quarterly |
| Autocall barrier | 100% of initial reference |
| Coupon barrier | 70% |
| Coupon | 3% per quarter |
| Coupon memory | No |
| Downside knock-in barrier | 60% |
| Knock-in monitoring | Daily close in final baseline |
| Capital protection | Conditional |
| Underlying rule | Worst of three assets |

为控制开发成本，可以先用 monthly 或 weekly knock-in monitoring 调试；最终主结果应使用 daily close monitoring。如果 CPU 时间不允许，则将 weekly monitoring 设为主规格，并把 daily monitoring 作为明确的 robustness experiment。

### 4.3 Cash-flow rule

在第 \(k\) 个季度 observation date：

- 若 \(W_{t_k}\geq B_{\mathrm{AC},k}\)，支付 notional，加上当期满足条件的 coupon，并终止；
- 若未 autocall 且 \(W_{t_k}\geq B_C\)，支付当期 coupon；
- 若未达到 coupon barrier，则不支付当期 coupon，且不累积。

若存续至到期：

- 若未发生 knock-in，归还 notional；
- 若发生 knock-in，但 \(W_T\geq 1\)，归还 notional；
- 若发生 knock-in 且 \(W_T<1\)，归还 \(\text{Notional}\times W_T\)；
- 到期 coupon 仍按 coupon barrier 判断。

### 4.4 必须测试的 payoff unit tests

- 所有资产始终高于 autocall barrier；
- 从不 autocall、从不 knock-in；
- 发生 knock-in 后恢复至 100% 以上；
- 发生 knock-in 且到期 worst-of 低于 100%；
- 恰好等于 coupon、autocall 和 knock-in barrier；
- 三项资产中不同资产分别成为 worst-of；
- autocall 后不得继续支付 coupon 或 maturity payoff。

---

## 5. 市场模型

### 5.1 Correlated GBM

在风险中性测度 \(\mathbb{Q}\) 下：

\[
\frac{dS_i(t)}{S_i(t)}
=
\left(r(t)-q_i(t)\right)dt+\sigma_i\,dW_i(t),
\]

且

\[
dW_i(t)dW_j(t)=\rho_{ij}dt.
\]

在 piecewise-constant parameters 下，离散步长 \(\Delta t\) 的 exact transition 为：

\[
S_{i,t+\Delta t}
=
S_{i,t}\exp\left[
\left(r-q_i-\frac{1}{2}\sigma_i^2\right)\Delta t
+\sigma_i\sqrt{\Delta t}(LZ)_i
\right],
\]

其中 \(LL^\top=\Sigma_\rho\)，\(Z\sim N(0,I_3)\)。

GBM 的 exact transition 消除了 Euler time-discretisation bias。核心实验中的 monitoring discretisation 是合同定义的一部分，不应与 SDE discretisation error 混为一谈。

### 5.2 Calibration policy

- Spot：估值日收盘价；
- Rate：与 maturity 对应的 risk-free/OIS zero curve；
- Dividend：index dividend yield、forecast yield 或 forward-implied yield；
- Volatility：优先使用与期限接近的 ATM implied volatility；
- Correlation：用对齐后的历史 log-return 估计，并进行 shrinkage 或 positive-definite check；
- Historical volatility：仅作为 robustness，不作为唯一的 risk-neutral volatility input。

必须在论文中承认：使用 ATM implied volatility 加 historical correlation 不是完整的 multi-asset risk-neutral calibration，而是为了隔离 numerical-method question 的可解释近似。

### 5.3 Bloomberg 数据文件

当前数据模板：

`outputs/bloomberg_autocallable_rqmc_20260724/Bloomberg_Autocallable_Data_Template_v2.xlsx`

数据处理规则：

- 使用已修正的 Bloomberg historical periodicity 参数；
- 对不同交易日历先取共同有效日期，再计算相关性；
- 不用向前填充价格制造零收益；
- 若标的是跨时区指数，说明 non-synchronous close bias，必要时改用 weekly returns 做 correlation robustness；
- 保存 raw、cleaned 和 calibrated parameters 三个层次，避免直接覆盖原始下载数据；
- 在每次实验结果中记录 calibration date 和 parameter snapshot。

---

## 6. 四个核心 estimators

| 编号 | Sampling | Barrier treatment | 用途 |
|---|---|---|---|
| M0 | Pseudorandom MC | Direct indicator | Baseline |
| M1 | Scrambled Sobol RQMC | Direct indicator | 单独检验 RQMC |
| M2 | Pseudorandom MC | Survival-conditioned downside leg | 单独检验 conditioning |
| M3 | Scrambled Sobol RQMC | Survival-conditioned downside leg | Hybrid |

所有方法必须使用同一产品、同一市场参数、同一 monitoring grid 和相同 payoff conventions。

### 6.1 Standard MC

- 使用独立 pseudo-random normals；
- 每个 sample size 做 \(R\) 次独立 replications；
- 输出 replication-level estimate，而不是只保留总体平均；
- 记录 seed、sample size、runtime 和 payoff count。

### 6.2 RQMC

- 使用 scrambled Sobol sequence；
- sample size 固定为 \(N=2^m\)；
- 每个 scramble 产生一个独立 estimator；
- RQMC standard error 必须从 independent scrambles 之间估计，不能把 Sobol 点当作 i.i.d. observations；
- baseline coordinate order 使用 time-major、asset-minor；
- Brownian bridge、PCA 或 linear transformation 作为 RQMC robustness / extension，而不是悄悄加入 M1 使比较不公平；
- inverse-normal 前将 \(u\) 限制在 \((\varepsilon,1-\varepsilon)\)，防止数值无穷。

### 6.3 Survival conditioning

#### 6.3.1 为什么不直接“把整个 autocallable 条件化”

三资产 autocallable 同时包含：

- downside knock-in barrier；
- upside autocall trigger；
- coupon trigger；
- worst-of operator；
- early termination。

若对完整 payoff 一次性条件化，会遇到多维截断区域和多个互补事件，既难实现，也不容易说明 estimator 是否无偏。

核心版本采用更清晰的 **payoff decomposition**。

#### 6.3.2 Downside protection decomposition

令 \(P_{\mathrm{base}}\) 为假设到期本金始终按 par 偿付时的 cash flows；令

\[
L
=
\mathbf{1}_{\{\tau_{\mathrm{AC}}>T\}}
\text{Notional}\,(1-W_T)^+
\]

为只有存续至到期时才可能发生的 downside loss。完整 payoff 可以写为：

\[
P=P_{\mathrm{base}}-\mathbf{1}_{\{\mathrm{KI}\}}L.
\]

利用

\[
\mathbf{1}_{\{\mathrm{KI}\}}
=
1-\mathbf{1}_{\{\mathrm{no\ KI}\}},
\]

得到：

\[
\mathbb{E}[P]
=
\mathbb{E}[P_{\mathrm{base}}]
-\mathbb{E}[L]
+\mathbb{E}[L\mathbf{1}_{\{\mathrm{no\ KI}\}}].
\]

前两项可用普通路径估计；最后一项用 one-step survival conditioning。这样 survival conditioning 直接针对最主要的 downside barrier discontinuity，同时保留 autocallable 的实际结构。

#### 6.3.3 多资产 sequential survival transform

在每个 monitoring step，no-knock-in 要求三项资产均高于各自的 absolute knock-in level。对 Cholesky 表示 \(LZ\)，按资产顺序逐个生成截断 normal：

\[
Z_i>\ell_i(Z_1,\ldots,Z_{i-1};S_{t-\Delta t}),
\]

\[
p_i=1-\Phi(\ell_i),
\]

\[
Z_i
=
\Phi^{-1}\left(
\Phi(\ell_i)+U_i p_i
\right).
\]

单步 weight 为 \(\prod_i p_i\)，整条未敲入路径的 likelihood weight 为各 monitoring steps 的 weight 乘积。实际公式必须根据 drift、volatility、Cholesky coefficients 和 absolute barrier levels 明确推导并写入附录。

实现检查：

- 单资产结果先复现 Glasserman–Staum one-step survival；
- 独立资产情形与三个单变量 survival probabilities 的乘积交叉检查；
- correlation matrix 必须 positive definite；
- 使用 log-weights 防止长到期下概率乘积 underflow；
- early autocall 后停止继续累积无用的 survival weight；
- 比较固定资产顺序；若做 barrier-aware ordering，单独标记为 M2a/M3a。

### 6.4 RQMC + survival conditioning

M3 使用 scrambled Sobol uniforms 驱动 sequential truncated-normal transform。

这两个方法可能互补：

- survival conditioning 降低 payoff discontinuity；
- RQMC 改善 integration point coverage；
- 更平滑的 integrand 理论上更适合 RQMC。

但 hybrid 不保证处处最优。多次 observation、worst-of switching 和 autocall trigger 仍会保留不连续性，这正是实验需要回答的问题。

---

## 7. Delta estimation

### 7.1 定义

对第 \(i\) 项资产：

\[
\Delta_i=\frac{\partial V}{\partial S_{i,0}}.
\]

使用 central finite difference：

\[
\widehat{\Delta}_i(h)
=
\frac{\widehat V(S_{i,0}+h)-\widehat V(S_{i,0}-h)}{2h}.
\]

### 7.2 必须固定的合同量

进行 spot bump 时，合同的 initial reference、autocall levels、coupon barriers 和 knock-in barriers 都保持为原来的 absolute levels。

不能随着 bumped spot 重新设置 barrier，否则估计的不是合同 Delta，甚至会因 payoff homogeneity 得到误导性结果。

### 7.3 Common random numbers

- \(S+h\) 和 \(S-h\) 使用完全相同的 pseudo-random normals；
- 对 RQMC，使用相同的 Sobol points 和相同 scramble；
- 对 survival-conditioned estimator，使用相同 uniforms，但重新计算 bumped state 下的 truncation bounds 和 weights；
- 三项 Delta 分开报告，同时可增加 parallel spot shock 的 directional Delta。

### 7.4 Bump-size experiment

建议：

\[
\frac{h}{S_0}\in
\{0.05\%,0.10\%,0.25\%,0.50\%,1.00\%\}.
\]

评价：

- replication standard deviation；
- RMSE against high-precision reference Delta；
- 相邻 bump sizes 的 estimate dispersion；
- Delta curve 随 spot shock 的平滑程度；
- runtime；
- barrier 附近是否出现不稳定 spikes。

核心落点不是找到一个“最好看的 bump”，而是说明每种 estimator 在 bias–variance–runtime 之间如何权衡。

---

## 8. Validation ladder

项目必须遵循从可解析到不可解析的三层验证。

### 8.1 Layer 1 — European option

目的：验证 path generation、discounting、dividend yield、random sampling、RQMC replications 和 timing harness。

测试：

- 与 Black–Scholes closed form 比较 price；
- 与 analytic Delta 比较 finite-difference Delta；
- \(N=2^{10},\ldots,2^{18}\)；
- MC 与 RQMC 各做相同数量 independent replications / scrambles；
- 绘制 log(RMSE)–log(\(N\)) 和 runtime–error。

European option 只承担 engine validation，不作为论文最终研究对象。

### 8.2 Layer 2 — Single-asset barrier option

目的：隔离 barrier discontinuity，验证 survival conditioning。

建议使用 discretely monitored down-and-out 或 up-and-out call，并设置三种 barrier distance：

- far from spot；
- intermediate；
- near spot。

检查：

- standard MC 与 one-step survival price 是否一致；
- 与 analytic continuous-barrier value 或高精度 discrete reference 比较；
- barrier proximity 对 variance 和 Delta 的影响；
- survival conditioning 是否改善 finite-difference Delta；
- 监控频率变化是否影响结论。

### 8.3 Layer 3 — Three-asset autocallable

目的：在真正研究对象上比较四种方法。

由于不存在方便的 closed form，reference value 采用：

1. 大样本 hybrid estimator；
2. 大样本 independent standard MC 作为无偏 cross-check；
3. 两种 reference confidence intervals 应重叠；
4. Delta reference 使用高预算 CRN central difference，并做 bump-size plateau / Richardson consistency check。

不能只把某一次最大的 M3 结果定义为“真值”而不交叉验证。

---

## 9. 实验设计

### 9.1 Replication structure

- 初始建议 \(R=24\) 或 \(32\)；
- MC 使用 \(R\) 个 independent seeds；
- RQMC 使用 \(R\) 个 independent scrambles；
- sample sizes 使用 \(2^m\)；
- 所有方法在相同 \(N\) 下比较，并同时报告 actual runtime；
- pilot experiment 后再冻结最大 \(N\)，避免先设计一个无法运行的 full factorial。

### 9.2 Experiment A — European validation

输出：

- price bias / RMSE；
- Delta bias / RMSE；
- empirical convergence slope；
- confidence-interval coverage；
- runtime。

### 9.3 Experiment B — Barrier option bridge

输出：

- price RMSE by barrier distance；
- Delta dispersion by bump size；
- MC 与 survival conditioning 的 variance reduction factor；
- RQMC 是否在 payoff smoothing 后获得更大收益。

### 9.4 Experiment C — Baseline autocallable

比较 M0–M3：

- price；
- standard error；
- RMSE；
- three component Deltas；
- Delta stability；
- runtime；
- work-normalised variance；
- time-to-tolerance。

### 9.5 Experiment D — Regime analysis

每次只改变一个主要参数，避免 full factorial 失控。

| Dimension | Suggested levels |
|---|---|
| Equicorrelation | 0.2, 0.5, 0.8 |
| Volatility scale | 75%, 100%, 125% of baseline |
| Knock-in barrier | 50%, 60%, 70% |
| Autocall barrier | 90%, 100%, 110% |
| Monitoring | Weekly, daily |

每个 regime 不必重复全部 sample-size grid。可以先用固定中等 \(N\) 做横截面比较，只对最有解释力的 regimes 做完整 convergence analysis。

### 9.6 Experiment E — Optional distinction-level ablation

仅在 C 和 D 完成后：

- fixed Cholesky asset order；
- weakest-to-barrier first；
- strongest-to-barrier first；
- chronological Sobol dimensions；
- Brownian bridge / PCA ordering。

观察 conditioning order 和 effective dimension 是否改变 hybrid estimator 的表现。

---

## 10. 评价指标

### 10.1 Accuracy

\[
\text{Bias}=\mathbb{E}[\widehat V]-V_{\mathrm{ref}},
\]

\[
\text{RMSE}
=
\sqrt{
\frac{1}{R}\sum_{r=1}^{R}
(\widehat V_r-V_{\mathrm{ref}})^2
}.
\]

同时报告 absolute error、relative error 和 confidence-interval coverage。

### 10.2 Stability

价格：

- replication standard deviation；
- interquartile range；
- seed / scramble sensitivity。

Delta：

- replication standard deviation；
- median absolute deviation；
- bump-size dispersion；
- spot-grid roughness；
- barrier-neighbourhood spikes。

不要在 Delta 接近零时只使用 coefficient of variation，因为分母会使结果失真。

### 10.3 Runtime

- wall-clock time；
- payoff evaluations；
- path steps；
- peak memory（可选）；
- setup time 与 pricing time 分开；
- JIT warm-up 不混入稳态 runtime，另行报告；
- 每项 timing 重复多次并取 median。

### 10.4 综合效率

建议至少使用：

\[
\text{Work-normalised error}
=
\text{RMSE}^2\times\text{runtime},
\]

以及：

> **Time-to-accuracy:** 达到预先定义的 price 或 Delta tolerance 所需的最短时间。

最终用 Pareto frontier 表达 accuracy–stability–runtime trade-off，而不是只给出一个“speed-up”数字。

---

## 11. 最终应产出的图表

### 11.1 必要图

1. European price RMSE vs \(N\)；
2. European Delta RMSE vs \(N\)；
3. Barrier price RMSE vs barrier distance；
4. Barrier Delta estimate vs bump size；
5. Autocallable price RMSE vs runtime；
6. Autocallable Delta RMSE / dispersion vs runtime；
7. 四方法的 time-to-accuracy；
8. 方法排名随 correlation 或 volatility 的变化；
9. Delta 随某一 underlying spot shock 的曲线；
10. accuracy–stability–runtime Pareto plot。

### 11.2 必要表

1. Baseline term sheet；
2. Market parameters and data sources；
3. M0–M3 method definition；
4. European validation errors；
5. Baseline autocallable price and Delta results；
6. Regime comparison；
7. Runtime and efficiency；
8. Limitations and extension results。

---

## 12. 实施顺序

### Phase 0 — Scope freeze

- 冻结 baseline term sheet；
- 冻结 monitoring convention；
- 冻结 market data date；
- 写出 exact payoff pseudocode；
- 创建 literature matrix。

**Gate 0:** payoff 可由手工路径测试。

### Phase 1 — Pricing engine

- correlated GBM exact simulation；
- discount curve and dividend inputs；
- European call；
- MC replication framework；
- scrambled Sobol interface；
- deterministic tests。

**Gate 1:** European price 与 Delta 达到预设 tolerance。

### Phase 2 — Barrier bridge

- direct barrier indicator；
- one-step survival；
- single-asset price comparison；
- barrier Delta experiment。

**Gate 2:** direct MC 与 conditioned estimator 在误差范围内一致。

### Phase 3 — Autocallable

- observation schedule；
- coupon logic；
- early redemption；
- worst-of logic；
- knock-in state；
- payoff unit tests。

**Gate 3:** 所有 hand-crafted path tests 通过。

### Phase 4 — Multi-asset conditioning

- downside loss decomposition；
- multivariate sequential truncation；
- log-weight implementation；
- fixed ordering；
- M2 和 M3。

**Gate 4:** 四方法在大样本下 price intervals 一致。

### Phase 5 — Delta

- common-random-number repricing；
- three component Deltas；
- bump grid；
- analytic European Delta validation；
- barrier and autocallable Delta stability。

**Gate 5:** Delta 结果存在可解释 plateau，且 reference construction 通过 cross-check。

### Phase 6 — Main experiments

- baseline sample-size grid；
- regime analysis；
- runtime protocol；
- figures and tables；
- robustness checks。

**Gate 6:** 核心图表冻结后才能开始扩展。

### Phase 7 — Extensions

按 CPU/JIT、Vega/Gamma、GPU、Heston 的顺序推进。

---

## 13. 建议时间表

| Week | Main output |
|---|---|
| 1 | Literature matrix、产品规格、Bloomberg 数据清理 |
| 2 | GBM engine、European validation、timing harness |
| 3 | Single-asset barrier、one-step survival、Delta validation |
| 4 | Three-asset autocallable payoff 与 unit tests |
| 5 | Multi-asset survival conditioning、RQMC hybrid |
| 6 | Baseline price and Delta experiments |
| 7 | Regime analysis、Pareto comparison、核心图表 |
| 8 | 写作、robustness、复现检查 |
| 9+ | GPU、更多 Greeks 或 Heston，若核心已完成 |

若时间明显不足，优先保留：

1. European validation；
2. barrier bridge；
3. baseline autocallable；
4. M0–M3 price；
5. one component Delta；
6. correlation 和 volatility 两组 regime；
7. accuracy–stability–runtime conclusion。

---

## 14. 代码与结果组织

建议目录：

```text
project/
├─ config/
│  ├─ product.yaml
│  └─ market.yaml
├─ data/
│  ├─ raw/
│  ├─ cleaned/
│  └─ calibrated/
├─ src/
│  ├─ models/
│  │  └─ gbm.py
│  ├─ products/
│  │  ├─ european.py
│  │  ├─ barrier.py
│  │  └─ autocallable.py
│  ├─ estimators/
│  │  ├─ mc.py
│  │  ├─ rqmc.py
│  │  └─ survival.py
│  ├─ greeks/
│  │  └─ finite_difference.py
│  └─ experiments/
├─ tests/
├─ results/
│  ├─ raw/
│  ├─ tables/
│  └─ figures/
├─ notebooks/
└─ README.md
```

每个结果文件至少记录：

- git commit 或 code version；
- valuation date；
- model parameters；
- product parameters；
- method；
- sample size；
- replication / scramble id；
- seed；
- price；
- Greeks；
- runtime；
- hardware；
- timestamp。

---

## 15. 论文结构

### 15.1 约 3,000 words 的版本

| Section | Target words | 内容 |
|---|---:|---|
| Abstract | 120 | 问题、方法、主要结果 |
| Introduction | 300 | 动机、research question、contribution |
| Literature and Product | 450 | MC/RQMC、barrier conditioning、autocallable |
| Methodology | 650 | GBM、payoff、M0–M3、Delta |
| Data and Experiment Design | 300 | Bloomberg、calibration、replications、metrics |
| Results and Discussion | 950 | validation、price、Delta、runtime、regimes |
| Conclusion | 230 | trade-off、limitations、extensions |
| **Total** | **3,000** |  |

### 15.2 Results section 的叙事顺序

1. Engine 是正确的；
2. Barrier discontinuity 确实造成问题；
3. Conditioning 在简单 barrier 上解决了什么；
4. 进入 autocallable 后还剩下哪些不连续性；
5. RQMC 与 conditioning 是否互补；
6. 价格上的胜者是否也是 Delta 上的胜者；
7. 结论在什么 regimes 下改变；
8. 最终给出 accuracy–stability–runtime frontier。

---

## 16. 主要风险与应对

### Risk 1 — Survival conditioning 实现过度复杂

先完成单资产版本，再做三资产 downside leg decomposition。不要从完整 autocallable 的全部 trigger 同时开始。

### Risk 2 — RQMC 在高维 daily monitoring 下改善有限

这可能是有效结果。检查 coordinate order、monitoring frequency 和 payoff smoothing 后，再把“effective dimension 与 discontinuity 限制 RQMC”写成结论。

### Risk 3 — Delta 没有稳定参考值

使用 CRN、大预算、多 bump sizes 和 independent reference construction。报告 bump-induced bias，而不是隐藏它。

### Risk 4 — Reference value 循环定义

用大样本 hybrid estimator 与 independent standard MC 交叉验证，并报告两者 confidence intervals。

### Risk 5 — GPU 掩盖 estimator 本身的优劣

先在同一 CPU implementation 上比较 M0–M3。GPU 结果作为独立 extension，以相同算法的 CPU/GPU speed-up 表达。

### Risk 6 — Heston 导致 scope explosion

只有当所有核心图表、Delta 和 regime analysis 已完成才进入 Heston。Heston 需要额外 calibration、variance-process simulation 和 discretisation-bias discussion。

### Risk 7 — 只报告“好看的”参数组合

预先冻结 baseline 和 regime grid，保留 methods underperform 的结果，并解释原因。

---

## 17. 最终结论应回答什么

最终结论不应只是：

> RQMC 比 MC 快，survival conditioning 降低方差。

更有论文价值的结论形式是：

> 对 correlated-GBM 三资产 worst-of autocallable，survival conditioning 主要通过平滑 downside knock-in leg 改善 barrier-near regimes 的价格和 finite-difference Delta；scrambled Sobol RQMC 在有效维度较低、conditioning 后 integrand 较平滑时进一步降低 time-to-accuracy。然而，autocall、coupon 和 worst-of switching 保留的不连续性限制了 hybrid method 的收益，且方法排名会随 correlation、volatility、barrier proximity 和 monitoring frequency 改变。因此，不存在对所有场景占优的单一方法，合理选择取决于 accuracy、Delta stability 与 runtime 的目标权重。

---

# 18. Reference list and reading guide

## 18.1 第一优先级：真正需要精读

### 1. Glasserman — Monte Carlo 总框架

Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.  
[Publisher and DOI](https://doi.org/10.1007/978-0-387-21617-1)

**读什么：** path generation、variance reduction、QMC、discretisation、estimating sensitivities 对应的章节。  
**用于哪里：** Methodology 的总体理论框架、MC error、variance reduction、Greeks。  
**评价：** 本项目最重要的单本参考书，不需要从第一页顺序读完。

### 2. Glasserman and Staum — Survival conditioning 的基础

Glasserman, P. and Staum, J. (2001). “Conditioning on One-Step Survival for Barrier Option Simulations.” *Operations Research*, 49(6), 923–937.  
[DOI and article](https://doi.org/10.1287/opre.49.6.923.10018)

**读什么：** change of measure、one-step conditional survival、estimator construction 和 unbiasedness。  
**用于哪里：** 单资产 barrier bridge 和 multi-step conditioning 的理论来源。  
**评价：** survival conditioning 部分的核心论文。

### 3. Fries and Joshi — Autocallable + smoothing + Greeks

Fries, C. P. and Joshi, M. S. (2011). “Perturbation Stable Conditional Analytic Monte-Carlo Pricing Scheme for Auto-Callable Products.” *International Journal of Theoretical and Applied Finance*, 14(2), 197–219.  
[DOI and abstract](https://doi.org/10.1142/S0219024911006334)

**读什么：** autocallable trigger discontinuity、conditional analytic smoothing、finite-difference sensitivities。  
**用于哪里：** literature gap、autocallable conditioning、Delta stability。  
**评价：** 与本项目最接近的文献之一，必须读；它也提醒你不能把“conditioning autocallable”包装成全新发明。

### 4. L’Ecuyer — RQMC 在金融中的最佳综述入口

L’Ecuyer, P. (2009). “Quasi-Monte Carlo Methods with Applications in Finance.” *Finance and Stochastics*, 13, 307–349.  
[Open-access article](https://doi.org/10.1007/s00780-009-0095-y)

**读什么：** randomisation、error estimation、effective dimension、RQMC 适用条件。  
**用于哪里：** 为什么使用 scrambled Sobol，以及为什么不应宣称 RQMC 对所有高维问题都有效。  
**评价：** 比直接读大量数论论文更适合作为项目入口。

### 5. Owen — Scrambled nets 的理论依据

Owen, A. B. (1997). “Monte Carlo Variance of Scrambled Net Quadrature.” *SIAM Journal on Numerical Analysis*, 34(5), 1884–1910.  
[DOI and article](https://doi.org/10.1137/S0036142994277468)

**读什么：** scrambled net variance 和 randomised error estimation 的理论逻辑。  
**用于哪里：** RQMC estimator 定义，以及为什么用 independent scrambles 估计 uncertainty。  
**评价：** 理论较重；重点理解结论和 assumptions，不必复现全部证明。

### 6. Broadie and Glasserman — Monte Carlo Greeks

Broadie, M. and Glasserman, P. (1996). “Estimating Security Price Derivatives Using Simulation.” *Management Science*, 42(2), 269–285.  
[DOI and article](https://doi.org/10.1287/mnsc.42.2.269)

**读什么：** finite difference、pathwise method、likelihood-ratio method 的差异和 bias/variance 问题。  
**用于哪里：** Delta methodology 和 future extensions。  
**评价：** 解释“为什么 price 看起来稳定但 Delta 不稳定”的基础文献。

### 7. Gerstner, Harrach and Roth — Barrier sensitivities

Gerstner, T., Harrach, B. and Roth, D. (2020). “Monte Carlo Pathwise Sensitivities for Barrier Options.” *Journal of Computational Finance*, 23(5), 75–99.  
[DOI and article](https://doi.org/10.21314/JCF.2020.385)

**读什么：** one-step survival 如何与 barrier Greeks 结合，以及 finite-difference instability。  
**用于哪里：** barrier Delta、更多 Greeks extension。  
**评价：** 直接连接本项目的 barrier、conditioning 和 sensitivity 三条线。

### 8. Deng, Mallett and McCann — Autocallable 产品与估值

Deng, G., Mallett, J. and McCann, C. J. (2011). “Modeling Autocallable Structured Products.” *Journal of Derivatives & Hedge Funds*, 17, 326–340.  
[DOI](https://doi.org/10.1057/jdhf.2011.25) · [Author version / abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1981308)

**读什么：** autocall feature、cash-flow logic、call probabilities 和估值框架。  
**用于哪里：** Product Description 和 industry motivation。  
**评价：** 产品部分的清晰入口，虽然其主要 numerical method 是 PDE。

## 18.2 第二优先级：与论文结果非常相关

### 9. Koster and Rehmet — Multi-asset payoff smoothing

Koster, F. and Rehmet, A. (2018). “Monte Carlo Payoff Smoothing for Pricing Autocallable Instruments.” *Journal of Computational Finance*, 21(4), 59–77.  
[DOI and article](https://doi.org/10.21314/JCF.2018.340)

**用途：** 与 M2/M3 的设计和 stable finite-difference Greeks 直接对话；尤其适合 literature review 的“closest methods”小节。

### 10. Cui, Li and Zhang — 较新的 autocallable pricing and hedging 文献

Cui, Y., Li, L. and Zhang, G. (2024). “Pricing and Hedging Autocallable Products by Markov Chain Approximation.” *Review of Derivatives Research*, 27, 259–303.  
[Open-access article](https://doi.org/10.1007/s11147-024-09206-z)

**用途：** 展示 autocallable 的当代研究仍同时关注 pricing、Delta/hedging、barrier instability 和 runtime；可用于引言和 limitations。

### 11. Broadie, Glasserman and Kou — Discrete barrier monitoring

Broadie, M., Glasserman, P. and Kou, S. G. (1997). “A Continuity Correction for Discrete Barrier Options.” *Mathematical Finance*, 7(4), 325–349.  
[DOI and article](https://doi.org/10.1111/1467-9965.00035)

**用途：** 解释 continuous 与 discrete monitoring 不是同一产品，以及 monitoring frequency 为什么影响 price。

### 12. Lemieux — RQMC 实现型教材

Lemieux, C. (2009). *Monte Carlo and Quasi-Monte Carlo Sampling*. Springer.  
[Publisher and DOI](https://doi.org/10.1007/978-0-387-78165-5)

**用途：** scrambled constructions、effective dimension、practical implementation。  
**评价：** 比 Owen 论文更易读，可与 L’Ecuyer 综述配合。

### 13. Imai and Tan — Dimension reduction

Imai, J. and Tan, K. S. (2006). “A General Dimension Reduction Technique for Derivative Pricing.” *Journal of Computational Finance*, 10(2), 129–155.  
[DOI and article](https://doi.org/10.21314/JCF.2006.143)

**用途：** 若 RQMC 在 daily monitoring 下表现一般，用于解释和设计 linear-transform / PCA extension。

### 14. Joe and Kuo — Sobol direction numbers

Joe, S. and Kuo, F. Y. (2008). “Constructing Sobol Sequences with Better Two-Dimensional Projections.” *SIAM Journal on Scientific Computing*, 30(5), 2635–2654.  
[DOI and article](https://doi.org/10.1137/070709359)

**用途：** Sobol implementation 和高维 direction-number choice；不需要把构造证明写进论文正文。

### 15. Guillaume — Autocallable payoff and risk features

Guillaume, T. (2015). “Autocallable Structured Products.” *The Journal of Derivatives*, 22(3), 73–94.  
[DOI and article](https://doi.org/10.3905/jod.2015.22.3.073)

**用途：** 产品结构、worst/best-of features 和 risk-management discussion。

## 18.3 扩展阶段文献

### 16. Heston — Stochastic volatility model

Heston, S. L. (1993). “A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.” *The Review of Financial Studies*, 6(2), 327–343.  
[DOI](https://doi.org/10.1093/rfs/6.2.327)

**用途：** Heston extension 的模型基础。

### 17. Andersen — Heston simulation

Andersen, L. B. G. (2008). “Simple and Efficient Simulation of the Heston Stochastic Volatility Model.” *Journal of Computational Finance*, 11(3), 1–42.  
[DOI](https://doi.org/10.21314/JCF.2008.189)

**用途：** Heston extension 中的 QE simulation；比直接使用 naive Euler 更适合严谨实验。

### 18. Giles — 更进一步的 complexity reduction

Giles, M. B. (2008). “Multilevel Monte Carlo Path Simulation.” *Operations Research*, 56(3), 607–617.  
[DOI and repository record](https://doi.org/10.1287/opre.1070.0496)

**用途：** future work 中讨论 MLMC；不建议与 RQMC、survival conditioning、GPU 同时放入核心。

---

## 18.4 五天阅读顺序

### Day 1 — 建立共同语言

- Glasserman：path simulation、variance reduction、QMC、sensitivities；
- 输出一页术语表和 estimator comparison table。

### Day 2 — Barrier

- Glasserman and Staum；
- Broadie, Glasserman and Kou；
- 输出 one-step survival 推导和单资产 pseudocode。

### Day 3 — RQMC

- L’Ecuyer；
- Owen 的 abstract、introduction、main variance results；
- Lemieux 的 practical QMC chapters；
- 输出 RQMC replication and error-estimation protocol。

### Day 4 — Autocallable and Greeks

- Fries and Joshi；
- Koster and Rehmet；
- Broadie and Glasserman；
- Gerstner, Harrach and Roth；
- 输出“closest literature vs this project”矩阵。

### Day 5 — 产品与较新文献

- Deng, Mallett and McCann；
- Guillaume；
- Cui, Li and Zhang；
- 输出 baseline term sheet、payoff diagram 和 literature gap paragraph。

---

## 18.5 Literature matrix 模板

阅读每篇文献时记录：

| Field | Question |
|---|---|
| Product | European、barrier、autocallable 还是其他？ |
| Model | GBM、Heston、local vol？ |
| Dimension | 单资产还是多资产？ |
| Monitoring | Continuous 或 discrete？ |
| Method | MC、QMC、conditioning、smoothing、PDE？ |
| Greek | 是否研究 Delta / Vega / Gamma？ |
| Benchmark | 真值如何定义？ |
| Metric | variance、RMSE、runtime 还是 hedging error？ |
| Main finding | 方法何时有效？ |
| Limitation | 何时失效？ |
| Use in this project | 支撑哪一节或哪项实验？ |

这个矩阵会直接转化为 literature review，而不是在写论文时重新翻找文献。

