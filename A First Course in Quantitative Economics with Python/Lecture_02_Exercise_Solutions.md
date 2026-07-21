# Linear algebra — exercise solutions

**Course:** [A First Course in Quantitative Economics with Python](https://intro.quantecon.org/intro.html) (Sargent & Stachurski)
**Lecture:** 2 — [Linear Algebra](https://python.quantecon.org/linear_algebra.html)
**Author:** Ghazi Veda · [GitHub](https://github.com/Vghazi-econ)
**Date:** July 2026

Ten exercises covering all of Lecture 2, set inside a single unified scenario: **Saudi Arabia's Vision 2030 economic planning model**. Each exercise uses realistic data from this context and follows the same structure: problem → hand calculation → Python → verification → economic interpretation.

**Libraries used:** `numpy`, `scipy.linalg`

---

## Contents

| #                                             | Topic                        | Key idea                        |
| --------------------------------------------- | ---------------------------- | ------------------------------- |
| [1](#exercise-1--vectors)                      | Vectors                      | $x + y$, $\gamma x$         |
| [2](#exercise-2--inner-product-and-norm)       | Inner product and norm       | $p'x$, $\lVert x \rVert$    |
| [3](#exercise-3--span-and-linear-independence) | Span and linear independence | rank                            |
| [4](#exercise-4--matrix-operations)            | Matrix operations            | $A^\top$, $Aw$, $AA^\top$ |
| [5](#exercise-5--matrices-as-linear-maps)      | Matrices as linear maps      | $f(x) = Ax$                   |
| [6](#exercise-6--solving-linear-systems)       | Solving linear systems       | $\det(A)$, $A^{-1}b$        |
| [7](#exercise-7--least-squares)                | Least squares                | $\hat{x} = (A'A)^{-1}A'y$     |
| [8](#exercise-8--eigenvalues)                  | Eigenvalues                  | $Av = \lambda v$              |
| [9](#exercise-9--neumanns-theorem)             | Neumann's theorem            | $(I-A)^{-1}$, $\rho(A)$     |
| [10](#exercise-10--positive-definite-matrices) | Positive definite matrices   | $x'Ax > 0$, $\nabla C(x)$   |

---

## Exercise 1 — Vectors

### Problem

Saudi Arabia's government allocates spending across four ministries (Defence, Health, Education, Infrastructure) in two consecutive budget years (billion SAR):

$$
y_{2022} = (280,\ 180,\ 200,\ 140), \qquad y_{2023} = (260,\ 210,\ 220,\ 170)
$$

1. Compute the total two-year spending per ministry (vector addition).
2. The government proposes to scale the 2023 budget by a factor of 1.08 (an 8% increase). Compute the new vector.
3. Verify in Python that $1.08\, y_{2023} + y_{2022} = y_{2022} + 1.08\, y_{2023}$. Which property does this demonstrate?

### Hand calculation

**Task 1 — vector addition (element by element):**

$$
y_{2022} + y_{2023} = (280+260,\ 180+210,\ 200+220,\ 140+170) = (540,\ 390,\ 420,\ 310)
$$

**Task 2 — scalar multiplication (every element scaled):**

$$
1.08 \times y_{2023} = (1.08 \times 260,\ 1.08 \times 210,\ 1.08 \times 220,\ 1.08 \times 170) = (280.8,\ 226.8,\ 237.6,\ 183.6)
$$

### Python

```python
import numpy as np

y_2022 = np.array([280, 180, 200, 140])   # billion SAR
y_2023 = np.array([260, 210, 220, 170])

# Task 1: total two-year spending per ministry
total = y_2022 + y_2023
print(total)

# Task 2: 8% increase on the 2023 budget
scaled = 1.08 * y_2023
print(scaled)

# Task 3: commutativity of vector addition
lhs = 1.08 * y_2023 + y_2022
rhs = y_2022 + 1.08 * y_2023
print(np.allclose(lhs, rhs))
```

**Output:**

```
[540 390 420 310]
[280.8 226.8 237.6 183.6]
True
```

### Verification

- Hand result $(540, 390, 420, 310)$ matches the NumPy output ✅
- `np.allclose(lhs, rhs)` returns `True` — vector addition is **commutative**: the order in which budgets are added does not change the total ✅
- Sanity check: every element of the total exceeds both single-year values, as it must for positive spending ✅

### Economic interpretation

Vector addition aggregates budgets across years while preserving the ministry-level breakdown — no information about the composition of spending is lost. Scalar multiplication models a uniform proportional policy: the 8% increase raises every ministry by the same percentage, so relative priorities between ministries are unchanged.

---

## Exercise 2 — Inner product and norm

### Problem

A Saudi household's monthly spending vector is $x = (3, 5, 2, 8)$ (thousand SAR: food, housing, transport, education). The price index vector from the previous year is $p = (1.0, 1.05, 1.08, 1.02)$ (price relatives — 1.08 means 8% more expensive).

1. Compute $p'x$ by hand. What does this number represent?
2. Compute $\lVert x \rVert$ by hand.
3. A second household has spending $z = (4, 4, 3, 6)$. Compute $\lVert x - z \rVert$. Which household is more similar in spending pattern to a reference basket $r = (4, 5, 2, 7)$? Use the norm to decide.

### Hand calculation

**Task 1 — inner product (multiply matching elements, then sum):**

$$
p'x = (1.0)(3) + (1.05)(5) + (1.08)(2) + (1.02)(8) = 3 + 5.25 + 2.16 + 8.16 = 18.57
$$

The basket costs 18 thousand SAR at last year's prices; at current prices the same basket costs 18.57 thousand SAR — the inner product converts quantities into a total cost, and the extra 0.57 (≈ 3.2%) is the effect of inflation on this household.

**Task 2 — norm (square → sum → root):**

Stage 1 — square each element: $3^2 = 9,\ 5^2 = 25,\ 2^2 = 4,\ 8^2 = 64$

Stage 2 — add: $9 + 25 + 4 + 64 = 102$

Stage 3 — square root: $\lVert x \rVert = \sqrt{102} \approx 10.0995$

Sanity check: $\max_i |x_i| = 8 \le 10.0995 \le 18 = \sum_i |x_i|$ ✅

**Task 3 — distances (subtract → square → sum → root):**

$$
x - z = (-1,\ 1,\ -1,\ 2) \;\rightarrow\; 1+1+1+4 = 7 \;\rightarrow\; \lVert x - z \rVert = \sqrt{7} \approx 2.6458
$$

$$
x - r = (-1,\ 0,\ 0,\ 1) \;\rightarrow\; 1+0+0+1 = 2 \;\rightarrow\; \lVert x - r \rVert = \sqrt{2} \approx 1.4142
$$

$$
z - r = (0,\ -1,\ 1,\ -1) \;\rightarrow\; 0+1+1+1 = 3 \;\rightarrow\; \lVert z - r \rVert = \sqrt{3} \approx 1.7321
$$

Since $\sqrt{2} < \sqrt{3}$, household $x$ is more similar to the reference basket $r$.

### Python

```python
import numpy as np

x = np.array([3, 5, 2, 8])       # household x, thousand SAR
p = np.array([1.0, 1.05, 1.08, 1.02])   # price relatives
z = np.array([4, 4, 3, 6])       # household z
r = np.array([4, 5, 2, 7])       # reference basket

# Task 1: inner product — quantities valued at current prices
price_weighted = p @ x
print(f"p'x = {price_weighted:.2f}")

# Task 2: norm of x
norm_x = np.linalg.norm(x)
print(f"||x|| = {norm_x:.4f}")

# Task 3: distances
dist_xz = np.linalg.norm(x - z)
dist_xr = np.linalg.norm(x - r)
dist_zr = np.linalg.norm(z - r)
print(f"||x - z|| = {dist_xz:.4f}")
print(f"||x - r|| = {dist_xr:.4f}")
print(f"||z - r|| = {dist_zr:.4f}")

if dist_xr < dist_zr:
    print("Household x is more similar to the reference basket r")
else:
    print("Household z is more similar to the reference basket r")

# Verification: hand results match NumPy
print(np.allclose([dist_xz, dist_xr, dist_zr],
                  [np.sqrt(7), np.sqrt(2), np.sqrt(3)]))
```

**Output:**

```
p'x = 18.57
||x|| = 10.0995
||x - z|| = 2.6458
||x - r|| = 1.4142
||z - r|| = 1.7321
Household x is more similar to the reference basket r
True
```

### Verification

- Hand results $\sqrt{7},\ \sqrt{2},\ \sqrt{3}$ match NumPy via `np.allclose` ✅
- Sanity check: each distance lies between the largest single difference and the sum of absolute differences — e.g. for $x - z$: $2 \le 2.6458 \le 5$ ✅
- Common mistake to avoid: an inner product is a **scalar**, not a vector — `p * x` gives element-wise products (a vector), `p @ x` sums them into the inner product. Likewise a norm requires the **final square root**; without it you have the squared norm.

### Economic interpretation

The inner product $p'x$ converts a quantity vector into a total cost: the household's basket cost 18 thousand SAR at last year's prices and 18.57 at current prices, so inflation cost this household about 3.2%. This "quantities × prices, summed" operation is exactly how consumer price indices (such as GASTAT's CPI for Saudi Arabia) are constructed. The norm of a difference measures dissimilarity between spending patterns: household $x$ sits $\sqrt{2}$ away from the reference basket while $z$ sits $\sqrt{3}$ away, so $x$'s budget composition more closely resembles the reference household.

---

## Exercise 3 — Span and linear independence

### Problem

Vision 2030 defines three economic "pillars":

$$
p_1 = (1, 0, 2), \qquad p_2 = (0, 1, 3), \qquad p_3 = (2, 1, 7)
$$

($p_1$: oil revenue channel, $p_2$: tourism channel, $p_3$: combined channel.)

1. Is $p_3$ a linear combination of $p_1$ and $p_2$? Find the scalars if so.
2. What is the rank of $[p_1, p_2, p_3]$? What does this tell you about Vision 2030's effective number of independent policy pillars?
3. The ministry adds a fourth pillar $p_4 = (0, 0, 1)$. Does this expand the span? Verify with Python.

### Hand calculation

**Task 1 — is $p_3$ a linear combination of $p_1$ and $p_2$?**

We ask whether scalars $\gamma_1, \gamma_2$ exist such that $\gamma_1 p_1 + \gamma_2 p_2 = p_3$. Matching components:

$$
\text{1st: } \gamma_1 = 2 \qquad \text{2nd: } \gamma_2 = 1 \qquad \text{3rd (test): } 2\gamma_1 + 3\gamma_2 = 2 \cdot 2 + 3 \cdot 1 = 7 \;✓
$$

The test passes, so $p_3 = 2p_1 + 1p_2$ — it **is** a linear combination.

**Task 2 — rank:**

$p_3$ is built from $p_1$ and $p_2$, so it adds nothing new. The independent vectors are $p_1$ and $p_2$ only: $\text{rank} = 2$. Vision 2030 has two truly independent policy pillars, not three.

**Task 3 — does $p_4 = (0,0,1)$ expand the span?**

Try $\gamma_1 p_1 + \gamma_2 p_2 = (0, 0, 1)$. The first two components force $\gamma_1 = 0$ and $\gamma_2 = 0$, but then the third gives $2 \cdot 0 + 3 \cdot 0 = 0 \neq 1$. The only candidate combination fails, so $p_4$ **cannot** be built from $p_1$ and $p_2$ — the span **does expand** (rank rises from 2 to 3).

### Python

```python
import numpy as np

p1 = np.array([1, 0, 2])
p2 = np.array([0, 1, 3])
p3 = np.array([2, 1, 7])
p4 = np.array([0, 0, 1])

# Task 1: does 2*p1 + 1*p2 equal p3?
print(np.allclose(2*p1 + 1*p2, p3))

# Task 2: rank of the three pillars
A = np.column_stack([p1, p2, p3])
print(np.linalg.matrix_rank(A))

# Task 3: does adding p4 raise the rank?
B = np.column_stack([p1, p2, p3, p4])
print(np.linalg.matrix_rank(B))
```

**Output:**

```
True
2
3
```

### Verification

- Predicted outputs from the hand calculation (`True`, `2`, `3`) match the code ✅
- Debugging note: the first run printed `2` for the last line because `p4` had been typed as `(0, 0, 0)` — the zero vector, which can never expand a span since $\gamma \cdot 0 = 0$ adds no new direction. The hand calculation caught the discrepancy; when code and hand results disagree, trust neither until the difference is explained.

### Economic interpretation

Announcing three pillars while the rank is 2 means one "pillar" is only a repackaging of the other two: any outcome achievable with all three channels was already achievable with $p_1$ and $p_2$ alone, so the third adds presentation, not policy capacity. Adding the genuinely independent $p_4$ raises the rank to 3 and expands what the ministry can actually reach — real diversification, in the linear-algebra sense, means adding directions that no combination of existing instruments can replicate.

---

## Exercise 4 — Matrix operations

### Problem

A Gulf Cooperation Council data table gives GDP and inflation for three countries:

$$
A = \begin{bmatrix} 900 & 420 & 140 \\ 2.3 & 3.1 & 2.8 \end{bmatrix} \quad \text{(rows: GDP, inflation; columns: Saudi, UAE, Kuwait)}
$$

A weighting vector $w = (0.5, 0.3, 0.2)$ assigns regional importance weights.

1. Compute $A^\top$. What does a row of $A^\top$ represent?
2. Compute $Aw$. What does this result represent economically?
3. Compute $AA^\top$. What is its shape? What does element $(1,2)$ of the result equal?

### Hand calculation

**Task 1 — transpose.** $A$ is $2 \times 3$, so $A^\top$ is $3 \times 2$: rows become columns. Each row of $A^\top$ is one country's complete profile — e.g. row 1 is $(900,\ 2.3)$, Saudi Arabia's GDP and inflation. This is the standard econometric layout: rows are observations, columns are variables.

**Task 2 — $Aw$.** Shape check: $(2 \times 3)(3 \times 1) = 2 \times 1$ ✓. Each element is an inner product of one row with the weights:

$$
\text{row}_1 \cdot w = 900 \times 0.5 + 420 \times 0.3 + 140 \times 0.2 = 450 + 126 + 28 = 604
$$

$$
\text{row}_2 \cdot w = 2.3 \times 0.5 + 3.1 \times 0.3 + 2.8 \times 0.2 = 1.15 + 0.93 + 0.56 = 2.64
$$

So $Aw = (604,\ 2.64)$: the GCC's weighted regional GDP and weighted regional inflation.

**Task 3 — $AA^\top$.** Shape check: $(2 \times 3)(3 \times 2) = 2 \times 2$ ✓ (inner 3s match and cancel). Element $(1,2)$ is row 1 · row 2:

$$
900 \times 2.3 + 420 \times 3.1 + 140 \times 2.8 = 2070 + 1302 + 392 = 3764
$$

### Python

```python
import numpy as np

A = np.array([[900.0, 420.0, 140.0],
              [2.3,   3.1,   2.8]])
w = np.array([0.5, 0.3, 0.2])

# Task 1: transpose and its shape
print("A.T =", "\n", A.T)
print("shape =", A.T.shape)

# Task 2: weighted regional averages
print("Aw =", A @ w)

# Task 3: A A^T, its shape, and element (1,2)
AAt = A @ A.T
print(AAt.shape)
print(AAt[0, 1])   # maths element (1,2) — Python counts from 0
```

**Output:**

```
A.T = 
 [[900.    2.3]
 [420.    3.1]
 [140.    2.8]]
shape = (3, 2)
Aw = [604.     2.64]
(2, 2)
3764.0
```

### Verification

- All hand results (shape $3 \times 2$, $Aw = (604, 2.64)$, shape $2 \times 2$, element $3764$) match the code ✅
- Shape lines checked before every product — inner dimensions must match and cancel ✅
- Magnitude sanity check: hundreds × small weights must give hundreds ($604$ ✓); hundreds × single digits must give thousands ($3764$ ✓). Writing numbers without thousands separators (2070, not 2,070) avoids comma/decimal confusion.

### Economic interpretation

$Aw$ compresses a country-level data table into regional aggregates: 604 is GCC weighted GDP and 2.64 the GCC weighted inflation rate, with the weights encoding each economy's regional importance — the same construction the IMF uses for regional aggregates in the World Economic Outlook. Element $(1,2)$ of $AA^\top$, by contrast, is a computational ingredient without direct economic meaning: an uncentred cross-product of GDP and inflation becomes interpretable (as a covariance) only after subtracting means. Recognising which numbers carry economic meaning and which are intermediate machinery is part of reading matrix formulas fluently.

---

## Exercise 5 — Matrices as linear maps

### Problem

A policy matrix maps two instruments (spending change $s$, rate change $r$) to three outcomes (GDP, employment, inflation):

$$
A = \begin{bmatrix} 0.8 & -0.3 \\ 0.5 & 0.1 \\ 0.2 & -0.4 \end{bmatrix}
$$

1. State the input and output spaces: $f : \mathbb{R}^? \to \mathbb{R}^?$.
2. Compute $f(x)$ for $x = (2, 1)$ (increase spending by 2%, raise the rate by 1%). Show each inner product.
3. Verify additivity: compute $A(x + y)$ and $Ax + Ay$ for $y = (0, 3)$. Are they equal?

### Hand calculation

**Task 1 — the spaces.** $A$ is $3 \times 2$: input dimension = number of columns, output dimension = number of rows, so $f: \mathbb{R}^2 \to \mathbb{R}^3$ — two policy instruments in, three outcomes out.

**Task 2 — $f(x)$ for $x = (2, 1)$, one inner product per row:**

$$
\text{GDP: } 0.8 \times 2 + (-0.3) \times 1 = 1.6 - 0.3 = 1.3
$$

$$
\text{Employment: } 0.5 \times 2 + 0.1 \times 1 = 1.0 + 0.1 = 1.1
$$

$$
\text{Inflation: } 0.2 \times 2 + (-0.4) \times 1 = 0.4 - 0.4 = 0
$$

**Task 3 — additivity with $y = (0, 3)$.** First $x + y = (2, 4)$, then:

$$
A(x+y) = (1.6 - 1.2,\ \ 1.0 + 0.4,\ \ 0.4 - 1.6) = (0.4,\ 1.4,\ -1.2)
$$

$$
Ay = (-0.9,\ 0.3,\ -1.2), \qquad Ax + Ay = (1.3 - 0.9,\ \ 1.1 + 0.3,\ \ 0 - 1.2) = (0.4,\ 1.4,\ -1.2)
$$

Equal ✓ — additivity holds, as it must for every matrix map.

### Python

```python
import numpy as np

A = np.array([[0.8, -0.3],
              [0.5,  0.1],
              [0.2, -0.4]])
x = np.array([2, 1])      # the policy input from the problem
y = np.array([0, 3])

# Task 2: the policy effect
print("Ax =", A @ x)

# Task 3: additivity
lhs = A @ (x + y)
rhs = A @ x + A @ y
print("A(x+y) =", lhs)
print("Ax + Ay =", rhs)
print(np.allclose(lhs, rhs))
```

**Output:**

```
Ax = [1.3 1.1 0. ]
A(x+y) = [ 0.4  1.4 -1.2]
Ax + Ay = [ 0.4  1.4 -1.2]
True
```

### Verification

- Hand results match the code for all three vectors ✅
- Additivity is a theorem for matrix maps — it cannot fail. A first hand attempt produced "not equal", which correctly signalled arithmetic errors (a misplaced decimal in $(-0.3) \times 4$ and a sign slip in $1.3 + (-0.9)$) rather than a failure of the identity. When an identity appears to fail, hunt for the arithmetic error.
- Sign check: adding a negative number means subtracting — $1.3 + (-0.9) = 0.4$, not $2.2$.

### Economic interpretation

The matrix is a policy machine: feed in instruments (2 points of extra spending, a 1-point rate rise) and it returns outcomes — GDP up 1.3 points, employment up 1.1, and inflation exactly unchanged because the inflationary push of spending ($0.2 \times 2$) is precisely cancelled by the rate rise ($-0.4 \times 1$). A zero in economics is rarely "no effect"; it is usually two forces cancelling. Additivity is the deeper economic claim of linearity: the combined effect of two policy packages equals the sum of their separate effects — true in this linear world, but only an approximation in reality, where large interventions interact.

---

## Exercise 6 — Solving linear systems

### Problem

Two equilibrium conditions in a Gulf economy, where $Y$ = output and $r$ = interest rate:

$$
\text{Goods market: } 3Y - 2r = 100 \qquad \text{Money market: } Y + 4r = 80
$$

1. Write in matrix form $Ax = b$. State $A$ and $b$.
2. Compute $\det(A)$ by hand. Does a unique solution exist?
3. Solve using `scipy.linalg.solve`. What are $Y^*$ and $r^*$?
4. Verify: substitute $Y^*$ and $r^*$ back into both equations by hand.

### Hand calculation

**Task 1 — matrix form.** Each equation becomes one row of $A$; the coefficients of $Y$ and $r$ become the two columns, in that order:

$$
A = \begin{bmatrix} 3 & -2 \\ 1 & 4 \end{bmatrix}, \qquad
x = \begin{bmatrix} Y \\ r \end{bmatrix}, \qquad
b = \begin{bmatrix} 100 \\ 80 \end{bmatrix}
$$

**Shape line:** $A$ is $2 \times 2$; $x$ is $2 \times 1$; therefore $Ax$ is $2 \times 1$, which conforms with $b$ ($2 \times 1$). The inner dimensions cancel; the outer dimensions survive.

**Translation check** — multiplying each row of $A$ by $x$ must reproduce the original equation:

| Row | Row of $A$ times $x$ | Equals | Original equation |
| --- | --- | --- | --- |
| 1 | $3Y + (-2)r = 3Y - 2r$ | 100 | Goods market ✓ |
| 2 | $1Y + 4r = Y + 4r$ | 80 | Money market ✓ |

**Task 2 — the determinant, one stage per line.** For $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, $\det(A) = ad - bc$, with $a = 3$, $b = -2$, $c = 1$, $d = 4$:

1. $ad = 3 \times 4 = 12$
2. $bc = (-2) \times 1 = -2$
3. $ad - bc = 12 - (-2) = 12 + 2 = 14$

$$\det(A) = 14 \neq 0$$

Because the determinant is nonzero, $A$ is invertible and the system has **exactly one** solution. Stage 3 is the trap: subtracting a negative number adds it.

**Tasks 3 and 4 — the solution and its verification.** The solver returns $Y^* = 40$ and $r^* = 10$. Substituting back into the two original equations:

$$
\text{Goods market: } 3(40) - 2(10) = 120 - 20 = 100 \ \checkmark
$$

$$
\text{Money market: } (40) + 4(10) = 40 + 40 = 80 \ \checkmark
$$

Both markets clear at $(Y^*, r^*) = (40, 10)$.

### Python

```python
import numpy as np                      # NumPy: arrays and numerical work
from scipy.linalg import solve          # SciPy's linear-system solver
# If SciPy is unavailable: from numpy.linalg import solve (same interface)

# Coefficient matrix: one row per market, one column per variable
A = np.array([[3.0, -2.0],              # goods market:  3Y - 2r
              [1.0,  4.0]])             # money market:   Y + 4r

# Right-hand side: the target value of each equation
b = np.array([100.0, 80.0])

# Task 2: confirm invertibility before solving
determinant = np.linalg.det(A)
print("det(A) =", determinant)

# Task 3: solve() finds the x satisfying Ax = b, without inverting A
equilibrium = solve(A, b)
print("solution =", equilibrium)

# Task 4: machine verification — does A @ x reproduce b?
print("A @ x =", A @ equilibrium)
print("check:", np.allclose(A @ equilibrium, b))
```

**Output:**

```
det(A) = 14.000000000000004
solution = [40. 10.]
A @ x = [100.  80.]
check: True
```

### Verification

- **Hand check:** both original equations balance exactly at $(40, 10)$ — $120 - 20 = 100$ ✓ and $40 + 40 = 80$ ✓
- **Python check:** `np.allclose(A @ equilibrium, b)` returns `True`, confirming the hand result ✅
- **Dimension check:** $(2 \times 2)(2 \times 1) = (2 \times 1)$, conformable with $b$ ✅
- ⚠️ **Floating-point note:** the determinant printed as `14.000000000000004`, not `14`. Computers store decimals in binary and cannot represent every value exactly. Never write `det(A) == 0`; use `np.isclose(determinant, 0)` instead.
- **Sanity check:** both equilibrium values are positive, as output and a nominal interest rate must be. A negative $Y^*$ would signal an error in the setup, not an economy producing negative output.

### Economic interpretation

Output and the interest rate cannot be determined one at a time, because each market constrains both variables simultaneously — this is the core logic of the IS–LM framework. Solving $Ax = b$ locates the single pair $(Y^*, r^*)$ at which the goods market and the money market clear together.

The determinant carries the economic content. $\det(A) \neq 0$ means the two equations are genuinely different restrictions: output and the interest rate pull the two markets in different directions, so the two lines cross at exactly one point. Had $\det(A) = 0$, the lines would be either parallel (no equilibrium exists) or identical (infinitely many equilibria, and the model would fail to pin down the economy). A determinant is therefore not merely an algebraic quantity; it is a test of whether a model says enough to determine its own unknowns.

**MENA application.** The model as written assumes the interest rate is determined domestically. For Saudi Arabia this assumption fails: the riyal has been pegged to the United States dollar at 3.75 since 1986, and under free capital mobility the peg forces the Saudi Arabian Monetary Authority (SAMA) to track Federal Reserve policy rates closely — the open-economy trilemma, which holds that a country cannot simultaneously maintain a fixed exchange rate, free capital movement, and an independent monetary policy. In practice the second equation is replaced by $r = r^{\text{US}} + \text{spread}$, leaving the goods market alone to determine $Y$. Egypt after the November 2016 float sits at the opposite corner: it surrendered the fixed rate in order to regain some monetary independence. The mathematics is identical in both cases; what differs is which variable the institutional regime treats as exogenous.

---

## Exercise 7 — Least squares

### Problem

An Egyptian economist has four years of data on oil price $P$ and fiscal revenue $R$ (billion EGP):

| Year  | 2020 | 2021 | 2022 | 2023 |
| ----- | ---- | ---- | ---- | ---- |
| $P$ | 50   | 60   | 80   | 100  |
| $R$ | 420  | 480  | 570  | 650  |

She wants to fit $R = a + bP$.

1. Set up the matrix $A$ (with a column of ones) and the vector $y$. State their shapes.
2. Use `scipy.linalg.lstsq` to find $\hat{a}$ and $\hat{b}$.
3. Predict fiscal revenue when $P = 90$.
4. Compute and print the four residuals. Which year has the largest prediction error?

### Hand calculation

<!-- TODO -->

### Python

```python
# TODO: your solution here
```

**Output:**

```
TODO
```

### Verification

<!-- TODO -->

### Economic interpretation

<!-- TODO -->

---

## Exercise 8 — Eigenvalues

### Problem

A two-sector economy (oil $O$, non-oil $N$) has the transition matrix:

$$
A = \begin{bmatrix} 0.6 & 0.2 \\ 0.4 & 0.8 \end{bmatrix}
$$

1. Find the eigenvalues by solving $\det(A - \lambda I) = 0$ by hand.
2. Find the eigenvector for the dominant eigenvalue.
3. Verify your eigenvector satisfies $Av = \lambda v$ by hand.
4. Use Python to confirm. Then interpret the dominant eigenvector as the long-run sector composition.

### Hand calculation

<!-- TODO -->

### Python

```python
# TODO: your solution here
```

**Output:**

```
TODO
```

### Verification

<!-- TODO -->

### Economic interpretation

<!-- TODO -->

---

## Exercise 9 — Neumann's theorem

### Problem

Saudi Arabia's three-sector Leontief model:

$$
A = \begin{bmatrix} 0.20 & 0.10 & 0.05 \\ 0.15 & 0.25 & 0.10 \\ 0.05 & 0.10 & 0.30 \end{bmatrix}, \qquad d = (200,\ 300,\ 150) \text{ billion SAR (final demand)}
$$

1. Compute $\rho(A)$. Is the economy viable?
2. Solve $x = (I - A)^{-1} d$ using `scipy.linalg.solve`.
3. Verify your answer: does $(I - A)x \approx d$?
4. Compute the intermediate inputs $Ax$. By how much does each sector's gross output exceed its final demand?

### Hand calculation

<!-- TODO -->

### Python

```python
# TODO: your solution here
```

**Output:**

```
TODO
```

### Verification

<!-- TODO -->

### Economic interpretation

<!-- TODO -->

---

## Exercise 10 — Positive definite matrices

### Problem

A Saudi firm minimises the quadratic cost function $C(x) = x'Ax - b'x$, where $x = (x_1, x_2)$ are production levels at two plants:

$$
A = \begin{bmatrix} 5 & 1 \\ 1 & 4 \end{bmatrix}, \qquad b = \begin{pmatrix} 8 \\ 6 \end{pmatrix}
$$

1. Verify $A$ is positive definite using eigenvalues.
2. Derive the gradient $\partial C / \partial x$.
3. Set the gradient to zero and solve for $x^*$.
4. Verify in Python that the gradient at $x^*$ is numerically zero.
5. Compute the minimum cost $C(x^*)$.

### Hand calculation

<!-- TODO -->

### Python

```python
# TODO: your solution here
```

**Output:**

```
TODO
```

### Verification

<!-- TODO -->

### Economic interpretation

<!-- TODO -->

---

## Summary of methods

| Exercise | NumPy / SciPy tools                            | Concept verified                                  |
| -------- | ---------------------------------------------- | ------------------------------------------------- |
| 1        | `np.array`, `+`, `*`, `np.allclose`    | Commutativity of vector addition                  |
| 2        | `@`, `np.linalg.norm`                      | Inner product, Euclidean distance                 |
| 3        | `np.column_stack`, `np.linalg.matrix_rank` | Linear dependence, span                           |
| 4        | `.T`, `@`                                  | Transpose, weighted averages                      |
| 5        | `@`, `np.allclose`                         | Linearity:$A(x+y) = Ax + Ay$                    |
| 6        | `scipy.linalg.solve`, `det`                | Unique solution when$\det(A) \neq 0$            |
| 7        | `scipy.linalg.lstsq`                         | OLS as a projection                               |
| 8        | `scipy.linalg.eig`                           | Long-run dynamics from the dominant eigenvalue    |
| 9        | `eig`, `solve`                             | Viability when$\rho(A) < 1$                     |
| 10       | `eig`, `solve`                             | Optimisation via the gradient of a quadratic form |

---

*Solutions written as part of my study of quantitative economics. Feedback and corrections are welcome — open an issue or reach me on [GitHub](https://github.com/Vghazi-econ).*
