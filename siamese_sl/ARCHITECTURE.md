# Siamese Network Architectures for Synthetic Lethality Prediction

Three architectures for predicting SL from ESM-2 embeddings $\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^{1280}$.

**LN** (Layer Normalization): $\text{LN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \boldsymbol{\beta}$

**GELU** (Gaussian Error Linear Unit): $\text{GELU}(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF

**Notation**: Equations use standard function composition—$f(g(h(x)))$ means apply $h$ first, then $g$, then $f$. For example, $\text{Dropout}(\text{ReLU}(\text{LN}(\mathbf{W}\mathbf{x})))$ corresponds to the layer sequence: Linear → LN → ReLU → Dropout.

---

## Node Feature Creation (Pool PaRTI)

Gene → protein sequence → ESM-2 → Pool PaRTI → standardize → node feature $\mathbf{x} \in \mathbb{R}^{1280}$

**ESM-2** (`esm2_t33_650M_UR50D`): Transformer encoder (33 layers, 20 heads, 650M params) pretrained via MLM on UniRef50.

$$\mathbf{T}, \mathbf{A} = \text{ESM-2}(\text{sequence}), \quad \mathbf{T} \in \mathbb{R}^{L \times 1280}, \ \mathbf{A} \in \mathbb{R}^{33 \times 20 \times L \times L}$$

**Pool PaRTI**: Aggregate attention, compute PageRank importance weights, pool token embeddings:

$$\mathbf{w} = \text{PageRank}\left(\max_{\ell, h} \mathbf{A}^{(\ell, h)}\right) \in \mathbb{R}^L, \quad \mathbf{e} = \sum_{i=1}^{L} w_i \mathbf{t}_i \in \mathbb{R}^{1280}$$

**Standardize** across corpus (during embedding generation, not training):

$$\mathbf{x} = (\mathbf{e} - \boldsymbol{\mu}) / \boldsymbol{\sigma}$$

where $\boldsymbol{\mu}, \boldsymbol{\sigma}$ are computed over all successfully embedded genes (genes with invalid/short sequences yield zero vectors and are excluded from statistics).

---

## Architecture 1: Basic Siamese (SiameseSL)

$$\hat{y} = g_\phi(\psi(f_\theta(\mathbf{x}_1), f_\theta(\mathbf{x}_2)))$$

### Encoder $f_\theta: \mathbb{R}^{1280} \to \mathbb{R}^{256}$

$$\mathbf{h} = \text{Dropout}_{0.2}(\text{LeakyReLU}_{0.2}(\text{LN}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)))$$
$$\mathbf{z} = \text{LeakyReLU}_{0.2}(\text{LN}(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2))$$

$\mathbf{W}_1 \in \mathbb{R}^{512 \times 1280}$, $\mathbf{W}_2 \in \mathbb{R}^{256 \times 512}$

### Symmetric Aggregation $\psi: \mathbb{R}^{256} \times \mathbb{R}^{256} \to \mathbb{R}^{768}$

$$\psi(\mathbf{z}_1, \mathbf{z}_2) = [\mathbf{z}_1 + \mathbf{z}_2;\ \mathbf{z}_1 \odot \mathbf{z}_2;\ |\mathbf{z}_1 - \mathbf{z}_2|]$$

### Predictor $g_\phi: \mathbb{R}^{768} \to \mathbb{R}$

$$\mathbf{p} = \text{Dropout}_{0.2}(\text{LeakyReLU}_{0.2}(\text{LN}(\mathbf{W}_3 \psi + \mathbf{b}_3)))$$
$$\hat{y} = \mathbf{w}_4^\top \mathbf{p} + b_4$$

$\mathbf{W}_3 \in \mathbb{R}^{128 \times 768}$, $\mathbf{w}_4 \in \mathbb{R}^{128}$

---

## Architecture 2: Attention-Enhanced (SiameseSLWithAttention)

$$\mathbf{h}_i = \pi(\mathbf{x}_i), \quad \tilde{\mathbf{h}}_1, \tilde{\mathbf{h}}_2 = \text{CrossAttn}(\mathbf{h}_1, \mathbf{h}_2), \quad \hat{y} = g_\phi(\psi(\rho(\tilde{\mathbf{h}}_1), \rho(\tilde{\mathbf{h}}_2)))$$

where $\pi$ is projection, $\rho$ is post-attention mapping, $\psi$ is symmetric aggregation, $g_\phi$ is predictor.

### Projection $\pi: \mathbb{R}^{1280} \to \mathbb{R}^{512}$

$$\mathbf{h} = \text{LeakyReLU}_{0.2}(\text{LN}(\mathbf{W}_{\text{proj}} \mathbf{x} + \mathbf{b}_{\text{proj}}))$$

### Cross-Attention ($H=4$, $d_k=128$)

$$\tilde{\mathbf{h}}_1 = \text{MultiheadAttention}(\mathbf{h}_1, \mathbf{h}_2, \mathbf{h}_2)$$
$$\tilde{\mathbf{h}}_2 = \text{MultiheadAttention}(\mathbf{h}_2, \mathbf{h}_1, \mathbf{h}_1)$$

### Post-Attention $\rho: \mathbb{R}^{512} \to \mathbb{R}^{256}$

$$\mathbf{z} = \text{LeakyReLU}_{0.2}(\text{LN}(\mathbf{W}_{\text{post}} \tilde{\mathbf{h}} + \mathbf{b}_{\text{post}}))$$

### Predictor

$$\mathbf{p} = \text{Dropout}_{0.2}(\text{LeakyReLU}_{0.2}(\text{LN}(\mathbf{W}_3 \psi(\mathbf{z}_1, \mathbf{z}_2) + \mathbf{b}_3)))$$
$$\hat{y} = \mathbf{w}_4^\top \mathbf{p} + b_4$$

$\mathbf{W}_3 \in \mathbb{R}^{256 \times 768}$, $\mathbf{w}_4 \in \mathbb{R}^{256}$

---

## Architecture 3: Kernel/RKHS (SiameseSLKernel)

$$\hat{y} = g_\phi(\kappa(\varphi(f_\theta(\mathbf{x}_1)), \varphi(f_\theta(\mathbf{x}_2))))$$

### Encoder $f_\theta: \mathbb{R}^{1280} \to \mathbb{R}^{256}$

Default (`encoder_type='standard'`):

$$\mathbf{h} = \text{Dropout}_{0.2}(\text{GELU}(\text{LN}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)))$$
$$\mathbf{z} = \text{LN}(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)$$

**Optional encoder variants** (for parameter efficiency):
- `lowrank`: Low-rank factorized layers $\mathbf{W} \approx \mathbf{U}\mathbf{V}$ with configurable `encoder_rank`
- `bottleneck`: Two-stage bottleneck MLP with narrow intermediate dimension
- `gated`: Gated Linear Units (GLU) with sigmoid gating

### Hilbert Mapping $\varphi: \mathbb{R}^{256} \to \mathbb{R}^{256}$

$$\varphi(\mathbf{z}) = [\mathbf{P}\mathbf{z};\ \varphi_{\text{RFF}}(\mathbf{z})]$$

**Linear:** $\mathbf{P} \in \mathbb{R}^{128 \times 256}$ (orthogonal init)

**RFF:** $[\varphi_{\text{RFF}}(\mathbf{z})]_j = \sqrt{\frac{2}{D}} \cos\left(\frac{\boldsymbol{\omega}_j^\top \mathbf{z}}{\gamma} + b_j\right)$

where $\boldsymbol{\omega}_j \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $b_j \sim \text{Uniform}(0, 2\pi)$, $D=128$, $\gamma$ learnable.

### Kernel Features $\kappa: \mathbb{R}^{256} \times \mathbb{R}^{256} \to \mathbb{R}^3$

$$\kappa(\boldsymbol{\phi}_1, \boldsymbol{\phi}_2) = [\langle \boldsymbol{\phi}_1, \boldsymbol{\phi}_2 \rangle;\ \|\boldsymbol{\phi}_1\|_2 + \|\boldsymbol{\phi}_2\|_2;\ \|\boldsymbol{\phi}_1\|_2 \cdot \|\boldsymbol{\phi}_2\|_2]$$

### Predictor $g_\phi: \mathbb{R}^3 \to \mathbb{R}$

$$\hat{y} = \mathbf{w}_4^\top \text{GELU}(\mathbf{W}_3 \boldsymbol{\kappa} + \mathbf{b}_3) + b_4$$

$\mathbf{W}_3 \in \mathbb{R}^{32 \times 3}$, $\mathbf{w}_4 \in \mathbb{R}^{32}$

---

## Loss (Binary Cross-Entropy)

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log\sigma(\hat{y}_i) + (1 - y_i) \log(1 - \sigma(\hat{y}_i)) \right]$$

---

## Parameters

| Architecture | Total |
|-------------|-------|
| Basic | 887,553 |
| Attention | 2,036,993 |
| Kernel | 821,666 |
