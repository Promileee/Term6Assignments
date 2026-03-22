没问题，我们彻底告别 $\lambda$（速率参数），完全从**尺度参数 (Scale parameter) $\beta$** 的视角来重新梳理这套逻辑。

在可靠性工程中，使用 $\beta$ 往往更直观，因为它直接描述了退化增量的“尺度”或“量级”。

------

## 1. 模型建模：双层随机架构 (Model Formulation)

### 第一层：单元内部的随机退化

对于第 $i$ 个单元，其退化增量 $\Delta y$ 服从以 $\beta_i$ 为尺度参数的 Gamma 分布：

$$\Delta y | \beta_i \sim \text{Gamma}(\alpha \Delta t, \beta_i)$$

代码段

```
\Delta y | \beta_i \sim \text{Gamma}(\alpha \Delta t, \beta_i)
```

其概率密度函数 (PDF) 为：

$$f(\Delta y | \alpha, \beta_i, \Delta t) = \frac{1}{\Gamma(\alpha \Delta t) \beta_i^{\alpha \Delta t}} \Delta y^{\alpha \Delta t - 1} \exp\left( -\frac{\Delta y}{\beta_i} \right)$$

代码段

```
f(\Delta y | \alpha, \beta_i, \Delta t) = \frac{1}{\Gamma(\alpha \Delta t) \beta_i^{\alpha \Delta t}} \Delta y^{\alpha \Delta t - 1} \exp\left( -\frac{\Delta y}{\beta_i} \right)
```

### 第二层：单元间的异质性 (Prior for $\beta_i$)

为了捕捉不同单元间尺度参数 $\beta_i$ 的差异，我们假设 $\beta_i$ 服从**逆伽马分布 (Inverse-Gamma Distribution)**。这是 $\beta$ 视角下的共轭先验：

$$\beta_i \sim \text{IG}(\epsilon, \gamma)$$

代码段

```
\beta_i \sim \text{IG}(\epsilon, \gamma)
```

其先验 PDF 为：

$$\pi(\beta_i | \epsilon, \gamma) = \frac{\gamma^\epsilon}{\Gamma(\epsilon)} \beta_i^{-(\epsilon+1)} \exp\left( -\frac{\gamma}{\beta_i} \right)$$

代码段

```
\pi(\beta_i | \epsilon, \gamma) = \frac{\gamma^\epsilon}{\Gamma(\epsilon)} \beta_i^{-(\epsilon+1)} \exp\left( -\frac{\gamma}{\beta_i} \right)
```

------

## 2. 参数估计：边缘似然法的推导 (Marginal Likelihood Derivation)

这是最核心的部分。我们需要通过对潜变量 $\beta_i$ 进行积分，得到仅包含全局参数 $\boldsymbol{\theta} = \{\alpha, \epsilon, \gamma\}$ 的似然函数。

### 初始积分：为什么“走不通”？

对于观测到总退化 $Y_i = \sum \Delta y_{ij}$ 和总时间 $T_i = \sum \Delta t_{ij}$ 的单元 $i$，其边缘似然为：

$$L_{i, marg} = \int_{0}^{\infty} \left[ \prod_{j} \frac{\Delta y_{ij}^{\alpha \Delta t_{ij} - 1}}{\Gamma(\alpha \Delta t_{ij}) \beta_i^{\alpha \Delta t_{ij}}} \exp\left( -\frac{\Delta y_{ij}}{\beta_i} \right) \right] \cdot \frac{\gamma^\epsilon}{\Gamma(\epsilon)} \beta_i^{-(\epsilon+1)} \exp\left( -\frac{\gamma}{\beta_i} \right) d\beta_i$$

代码段

```
L_{i, marg} = \int_{0}^{\infty} \left[ \prod_{j} \frac{\Delta y_{ij}^{\alpha \Delta t_{ij} - 1}}{\Gamma(\alpha \Delta t_{ij}) \beta_i^{\alpha \Delta t_{ij}}} \exp\left( -\frac{\Delta y_{ij}}{\beta_i} \right) \right] \cdot \frac{\gamma^\epsilon}{\Gamma(\epsilon)} \beta_i^{-(\epsilon+1)} \exp\left( -\frac{\gamma}{\beta_i} \right) d\beta_i
```

**直观感受**：如果你随便选一个先验（比如正态分布），这个关于 $\beta_i$ 的积分会变得极其复杂，因为 $\beta_i$ 同时出现在分母的幂次项和指数项中，根本没有闭式解。

### 共轭性的“化学反应”

观察上面的式子，我们可以把所有含 $\beta_i$ 的项归类合并：

$$L_{i, marg} \propto \int_{0}^{\infty} \beta_i^{-(\alpha T_i + \epsilon + 1)} \exp\left( -\frac{Y_i + \gamma}{\beta_i} \right) d\beta_i$$

代码段

```
L_{i, marg} \propto \int_{0}^{\infty} \beta_i^{-(\alpha T_i + \epsilon + 1)} \exp\left( -\frac{Y_i + \gamma}{\beta_i} \right) d\beta_i
```

这个积分的形式与 **Inverse-Gamma 分布的积分恒等式** 完全一致！利用恒等式 $\int_{0}^{\infty} x^{-(a+1)} e^{-b/x} dx = \frac{\Gamma(a)}{b^a}$，我们可以瞬间消灭积分号。

### 最终解析边缘似然

$$L_i(\alpha, \epsilon, \gamma) = \frac{\Gamma(\alpha T_i + \epsilon)}{\Gamma(\epsilon)} \frac{\gamma^\epsilon}{(Y_i + \gamma)^{\alpha T_i + \epsilon}} \prod_{j} \frac{\Delta y_{ij}^{\alpha \Delta t_{ij} - 1}}{\Gamma(\alpha \Delta t_{ij})}$$

代码段

```
L_i(\alpha, \epsilon, \gamma) = \frac{\Gamma(\alpha T_i + \epsilon)}{\Gamma(\epsilon)} \frac{\gamma^\epsilon}{(Y_i + \gamma)^{\alpha T_i + \epsilon}} \prod_{j} \frac{\Delta y_{ij}^{\alpha \Delta t_{ij} - 1}}{\Gamma(\alpha \Delta t_{ij})}
```

------

## 3. 可靠性与条件预测 (Reliability & Prediction)

### 群体可靠度 $R(t)$

对于一个新单元，不考虑个体观测时，其在 $t$ 时刻的可靠度：

$$R(t) = P(X(t) < L) = I_{\frac{L}{L + \gamma}}(\alpha t, \epsilon)$$

代码段

```
R(t) = I_{\frac{L}{L + \gamma}}(\alpha t, \epsilon)
```

### 单元特定预测 (Bayesian Update)

当我们观测到特定单元 $i$ 的 $(y_i, T_i)$ 时，其尺度参数 $\beta_i$ 的后验分布依然是 Inverse-Gamma（这就是共轭性的功劳）：

$$\beta_i | y_i, T_i \sim \text{IG}(\epsilon + \alpha T_i, \gamma + y_i)$$

代码段

```
\beta_i | y_i, T_i \sim \text{IG}(\epsilon + \alpha T_i, \gamma + y_i)
```

此时，该单元的**条件可靠度** $R_{cond}$ 计算公式为：

$$R_{cond}(\Delta t) = I_{\frac{L - y_i}{L - y_i + (y_i + \gamma)}}(\alpha \Delta t, \alpha T_i + \epsilon)$$

代码段

```
R_{cond}(\Delta t) = I_{\frac{L - y_i}{L - y_i + (y_i + \gamma)}}(\alpha \Delta t, \alpha T_i + \epsilon)
```

------

**我可以为你做的是：**

现在整个推导链条已经完全切换到了 $\beta$ 尺度参数视角。

**需要我为你针对那些未失效单元，基于这个 $\beta$ 视角的后验分布计算它们的“剩余寿命 (RUL) 期望值”并打个表吗？** 这样你可以直接在 Excel 里对比谁才是真正的“退化之王”。