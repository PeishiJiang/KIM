# Mathematical approach
Consider a vector of inputs $\mathbf{X} = [X_1,...,X_{N_x}]$ and a vector of outputs $\mathbf{Y} = [Y_1,...,Y_{N_y}]$. The objective is to build up a mapping function $f$ from $\mathbf{X}$ to  $\mathbf{Y}$, such that $f: \mathbb{R}^{N_x} \rightarrow \mathbb{R}^{N_y}$, based on $N_e$ pairs/realizations of $\mathbf{X}$ and $\mathbf{Y}$. However, it is oftentimes hard to develop an accurate composite $f$, given the limited training data, partially due to the high computational cost of model ensemble simulation. Instead, it would be desirable to develop a separate inverse mapping $f_i$ for each $Y_j \in \mathbf{Y}$ by using a reduced space $\mathbf{X}^S_j \in \mathbf{X}$ that is most relevant to $Y_j$, such that $f_j: \mathbb{R}^{N_{x_j}} \rightarrow \mathbb{R}$ (see examples in @Jiang:2023 and @Wang:2025). The identification of $\mathbf{X}_j$ for a given $Y_j$ involves a two-step filtering as follows.

**Add a figure illustrating KIM.**

**Step 1: Filtering by global sensitivity analysis.** We first perform a mutual information-based global sensitivity analysis to narrow down a subset $\mathbf{X}^{S_1}_j$, each of which shares zero information with $Y_j$ such that:

$$\mathbf{X}^{S_1}_j = \{X_i: I(X_i;Y_j) \neq 0 \quad \text{with} \: X_i \in \mathbf{X}\},$$

where $I(X_i;Y_j)$ is the mutual information between $X_i$ and $Y_j$ [@Cover:2006]. Based on the $N_e$ realizations, $I$ is calculated on the joint probability of $X_i$ and $Y_j$ using either binning method or k-nearest-neighbor method. Following @Jiang:2023, a statistical significance test is performed to identify the significant $I$ (i.e., $I(X_i;Y_j) \neq 0$) with a significance level of $1-\alpha$ on 100 bootstrap samples.

**Step 2: Filtering by redundancy check.** Then, we conduct a further assessment that filtered out any model output in $\mathbf{X}^{S_1}_j$ whose dynamics are redundant to $Y_j$ given the knowledge of other outputs. This is achieved through a conditional independence test using conditional mutual information [@Cover:2006] given as:

$$\mathbf{X}^{S}_j = \{X_i: I(X_i;Y_j|\mathbf{X}^{S_1}_j \backslash X_i) \neq 0 \quad \text{with} \: X_i \in \mathbf{X}^{S_1}_j \},$$

where $\mathbf{X}^{S_1}_j \backslash X_i$ is the remaining set of $\mathbf{X}^{S_1}_j$ by excluding $X_i$; $I(X_i;Y_j|\mathbf{X}^{S_1}_j \backslash X_i)$ is the conditional mutual information between $X_i$ and $Y_j$ conditioning on $\mathbf{X}^{S_1}_j \backslash X_i$. $I(X_i;Y_j|\mathbf{X}^{S_1}_j \backslash X_i) = 0$ indicates that $X_i$ and $Y_j$ are independent given the knowledge of $\mathbf{X}^{S_1}_j \backslash X_i$. However, calculating $\mathbf{X}^{S_1}_j \backslash X_i$. $I(X_i;Y_j|\mathbf{X}^{S_1}_j \backslash X_i)$ faces the curse of dimensionality due to the potential high dimension in $\mathbf{X}^{S_1}_j$. 

To address this, we leverage the idea of Peter-Clark algorithm for causal inference detection [@Spirtes:2001] to evaluate the zeroness of a high-dimensional conditional mutual information by gradually adding conditioning variables. Specifically, we approximated $I(X_i;Y_j|\mathbf{X}^{S_1}_j \backslash X_i)$ via $I(X_i;Y_j|\mathbf{X}^{S_2}_j)$, where $\mathbf{X}^{S_2}_j$ is a subset of $\mathbf{X}^{S_1}_j \backslash X_i$ with cardinality $\leq 3$. Starting with the cardinality of one, i.e. $|\mathbf{X}^{S_2}_j|=1$,  we conducted statistical significance test on assessing $I(X_i;Y_j|\mathbf{X}^{S_2}_j) = 0$ by exhausting all the combinations out of $\mathbf{X}^{S_1}_j \backslash X_i$ that constitute $\mathbf{X}^{S_2}_j$. We removed $X_i$ from $\mathbf{X}^{S}_j$ when $I(X_i;Y_j|\mathbf{X}^{S_2}_j) = 0$.

**Step 3: Uncertainty aware estimation by training ensemble neural networks.** For each parameter $Y_i$, we train an ensemble of fully-connected neural networks by varying the hyperparameters, including the number of hidden layers, the number of hidden neurons, and the learning rate. We split the $N_e$ model realizations into training, validation, and testing dataset. For each model inference, the ensemble learning enables the predictions through weighted mean $\mu_w$ and weighted standard deviation $\sigma_w$ as:
$$
\begin{align}
    \mu_w &= \sum^{N_e}_{k=1} w_k \cdot \tilde{y_k} \notag\\
    \sigma_w &= \sqrt{\sum^{N_e}_{k=1} w_k \cdot (\tilde{y_k}-\mu_w)^2}, \notag
\end{align}
$$
where $N_e$ is the number of ensemble neural networks; $\tilde{y_k}$ is the estimation by the $k$ th neural network; $w_k$ is the weight to the $k$ th prediction and is calculated through the corresponding loss value in the validation dataset $\mathcal{L}_{k,\text{val}}$, such that $w_k = \frac{1/\mathcal{L}_{k,\text{val}}}{\sum^{N_e}_{k=1} 1/\mathcal{L}_{k,\text{val}}}$.

When evaluating the estimation on the test dataset, we further quantified the bias and uncertainty of the prediction as:
$$
\begin{align}
    \text{Bias} &= E(|\mu_w - y|) \notag\\
    \text{Uncertainty} &= E(\sigma_w / y) \notag,
\end{align}
$$
where $E$ is the expectation operator and $y$ is the true value.