"""
Content for the three code-walkthrough lessons of the Neural Network Lab.

Each lesson is a list of cells:
  ("md",   "markdown text")
  ("code", "python code")

Inside a code cell, a line may end with   # >> explanation   to attach a
line-by-line explanation. A line ending with   # >>^   shares the explanation
of the line above (used for statements that span several lines).

tools/build_walkthroughs.py turns this into a Jupyter notebook (executed, with
outputs) and an HTML page for the website, so the two always match.
Only NumPy, Matplotlib and (for the last notebook) scikit-learn are needed.

Author: Eng. Saeed Serwan Abdulsattar
"""

STYLE = '''plt.rcParams["figure.figsize"] = (7, 4)  # >> Default figure size (width, height) in inches.
plt.rcParams["axes.spines.top"] = False  # >> A cleaner look: hide the top and right borders of every chart.
plt.rcParams["axes.spines.right"] = False  # >>^
plt.rcParams["axes.grid"] = True  # >> Light grid lines make values easier to read.
plt.rcParams["grid.color"] = "#ececec"  # >>^'''

LOADER = '''import gzip, os, urllib.request  # >> Standard-library tools to download and unzip the data files.

URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"  # >> A public copy of the MNIST files on GitHub.

def load(name, offset):  # >> Download one MNIST file (once) and read it as bytes.
    if not os.path.exists(name):  # >> Only download if we do not have the file yet.
        urllib.request.urlretrieve(URL + name, name)  # >>^
    with gzip.open(name) as f:  # >> The files are gzip-compressed.
        return np.frombuffer(f.read(), np.uint8, offset=offset)  # >> Skip the header (16 bytes for images, 8 for labels) and read the rest as numbers 0–255.

X_train = load("train-images-idx3-ubyte.gz", 16).reshape(-1, 784) / 255  # >> 60,000 training pictures, each a row of 784 pixels, scaled to 0 … 1.
y_train = load("train-labels-idx1-ubyte.gz", 8)  # >> Their labels: the digit 0–9 each picture shows.
X_test = load("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 784) / 255  # >> 10,000 test pictures the network never trains on.
y_test = load("t10k-labels-idx1-ubyte.gz", 8)  # >>^
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # >> Check the sizes.'''


NETWORK = {
    "slug": "network",
    "page": "code-network.html",
    "notebook": "network_numpy.ipynb",
    "eyebrow": "Practice · A network in Python",
    "title": "A neural network in NumPy, <em>line by line.</em>",
    "doc_title": "Neurons, Layers and the Forward Pass in NumPy, Line by Line",
    "lead": "A complete, runnable notebook for chapters 2–3. Compute a neuron by hand, plot sigmoid and ReLU, load the MNIST digits, "
            "build the 784 → 16 → 16 → 10 network, count its 13,002 parameters and run a forward pass. "
            "Only NumPy and Matplotlib are needed. Hover over (or tap) any line of code to see what it does.",
    "learn": [
        "Compute one neuron: weighted sum, bias, sigmoid",
        "Plot sigmoid and ReLU and their slopes",
        "Load MNIST and look at the pictures as numbers",
        "Build the 784 → 16 → 16 → 10 network and count its parameters",
        "Run a forward pass and compute the cost of one example",
    ],
    "cells": [
        ("md", """## 1. Import the libraries

**NumPy** for arrays and maths, **Matplotlib** for pictures and charts. No deep-learning library: every step is written out by hand."""),
        ("code", """%config InlineBackend.figure_format = "retina"  # >> A Jupyter setting (not Python): sharper plots on high-resolution screens.
import numpy as np  # >> NumPy: arrays and maths. `np` is the standard nickname.
import matplotlib.pyplot as plt  # >> Matplotlib's plotting interface.

""" + STYLE),

        ("md", """## 2. One neuron by hand

The spam neuron of Exercise 1: each word present gives the input 1, absent gives 0. The neuron computes $\\sigma(w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + b)$."""),
        ("code", """def sigmoid(z):  # >> The squishing function σ(z) = 1 / (1 + e^(−z)).
    return 1 / (1 + np.exp(-z))  # >> np.exp works on single numbers and on whole arrays.

words = ["prize", "money", "dear", "win"]  # >> The four words the neuron looks for.
w = np.array([2.0, 1.5, -1.0, 1.0])  # >> One weight per word: positive words push towards spam, “dear” pushes away.
b = -1.5  # >> The bias: how much evidence is needed before the neuron fires.

x = np.array([0, 1, 1, 0])  # >> The email contains only “money” and “dear”.
z = w @ x + b  # >> `@` is the dot product: 2·0 + 1.5·1 − 1·1 + 1·0, then add the bias.
print("weighted sum + bias:", z, " sigma:", round(sigmoid(z), 2))  # >> −1.0 and σ(−1) = 0.27.
print("spam?", sigmoid(z) >= 0.5)  # >> Below 0.5, so not spam.

x2 = np.array([1, 1, 1, 0])  # >> Add the word “prize”.
print("with 'prize':", round(sigmoid(w @ x2 + b), 2))  # >> The sum becomes 1.0 and σ = 0.73: spam."""),

        ("md", """## 3. Activation functions

Sigmoid squishes into 0 … 1; ReLU keeps positive numbers and turns negative ones into 0. The dashed lines are the **slopes**: the sigmoid’s is never above 0.25."""),
        ("code", """z = np.linspace(-6, 6, 300)  # >> 300 evenly spaced values of z from −6 to 6.
s = sigmoid(z)  # >> σ for every value at once.
relu = np.maximum(0, z)  # >> ReLU(z) = max(0, z), element by element.

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))  # >> Two charts side by side.
ax[0].plot(z, s, label="sigmoid")  # >> The S-shaped curve.
ax[0].plot(z, s * (1 - s), "--", label="its slope σ(1 − σ)")  # >> The slope of the sigmoid; its peak is 0.25 at z = 0.
ax[0].legend(); ax[0].set_title("Sigmoid")  # >>^
ax[1].plot(z, relu, label="ReLU")  # >> A hinge: flat at 0, then a straight line.
ax[1].plot(z, (z > 0).astype(float), "--", label="its slope")  # >> The slope of ReLU: 0 or 1.
ax[1].legend(); ax[1].set_title("ReLU")  # >>^
plt.show()  # >> Draw the figure.
print("largest sigmoid slope:", (s * (1 - s)).max().round(3))  # >> 0.25: remember it for vanishing gradients."""),

        ("md", """## 4. Load the handwritten digits

MNIST: 70,000 pictures of 28 × 28 pixels. Each picture becomes a row of 784 numbers between 0 (black) and 1 (white)."""),
        ("code", LOADER),

        ("code", """threes = X_test[y_test == 3][:40]  # >> The first 40 test pictures that show a 3.
fig, axes = plt.subplots(4, 10, figsize=(10, 4.2))  # >> A grid of 4 × 10 small pictures.
for img, ax in zip(threes, axes.ravel()):  # >> Pair each picture with one small chart.
    ax.imshow(img.reshape(28, 28), cmap="gray")  # >> Turn the 784 numbers back into a 28 × 28 square.
    ax.axis("off")  # >> No axes around the pictures.
plt.suptitle("40 handwritten 3s: no two have the same pixels")  # >> One title for the whole grid.
plt.show()  # >>^

print("one picture as numbers (rows 10 to 13, columns 8 to 20):")  # >> Peek at the numbers the network will receive.
print(np.round(threes[0].reshape(28, 28)[10:14, 8:20], 1))  # >>^"""),

        ("md", """## 5. Build the 784 → 16 → 16 → 10 network

One weight matrix and one bias vector per connection between layers. Row k of a weight matrix holds the weights of neuron k. All start as small random numbers."""),
        ("code", """sizes = [784, 16, 16, 10]  # >> Input, two hidden layers, output.
rng = np.random.default_rng(0)  # >> A random number generator with a fixed seed, so results repeat.

W = [rng.normal(0, 1 / np.sqrt(n_in), (n_out, n_in))  # >> One matrix per connection, shape (neurons out, neurons in). Scaling by 1/√inputs keeps the sums a sensible size.
     for n_in, n_out in zip(sizes[:-1], sizes[1:])]  # >>^ zip pairs each layer with the next: (784, 16), (16, 16), (16, 10).
b = [np.zeros(n_out) for n_out in sizes[1:]]  # >> One bias per neuron, starting at 0.

for i, (Wi, bi) in enumerate(zip(W, b)):  # >> Show the shapes and count the knobs, layer by layer.
    print(f"layer {i+1}: W {Wi.shape}, b {bi.shape}, parameters {Wi.size + bi.size:,}")  # >>^
print("total parameters:", f"{sum(Wi.size + bi.size for Wi, bi in zip(W, b)):,}")  # >> 13,002."""),

        ("md", """## 6. The forward pass

Each layer is $a^{(1)} = \\sigma\\big(W a^{(0)} + b\\big)$. Three matrix multiplications, three sigmoids, and the 784 pixels have become 10 numbers."""),
        ("code", """def forward(x):  # >> Run one picture (or a whole batch of pictures) through the network.
    a = x  # >> The input layer: the pixels themselves.
    for Wi, bi in zip(W, b):  # >> For each layer in turn …
        a = sigmoid(a @ Wi.T + bi)  # >> … weighted sums of all neurons at once (a matrix multiply), add the biases, squish.
    return a  # >> The 10 output activations.

out = forward(X_test[0])  # >> The first test picture.
print("label:", y_test[0])  # >> What the picture really shows.
print("outputs:", np.round(out, 2))  # >> Ten numbers between 0 and 1, one per digit.
print("answer:", out.argmax())  # >> The brightest output neuron is the answer."""),

        ("md", """## 7. How bad is a random network?

The **cost** of one example: $C = \\sum_j (a_j - y_j)^2$, where $y$ is 1 for the right digit and 0 for the others. Then the accuracy over all 10,000 test pictures."""),
        ("code", """y = np.zeros(10); y[y_test[0]] = 1  # >> The wanted output: a 1 at the correct digit, 0 elsewhere (“one-hot”).
cost = ((out - y) ** 2).sum()  # >> Square the differences and add them up.
print("cost of this example:", round(cost, 2))  # >> Large: the random network is unsure and wrong.

all_out = forward(X_test)  # >> All 10,000 test pictures in one go: a 10,000 × 10 array.
acc = (all_out.argmax(axis=1) == y_test).mean()  # >> Compare each answer with the label; the mean of True/False is the fraction correct.
Y = np.eye(10)[y_test]  # >> One-hot rows for all labels (row k of the identity matrix is the one-hot vector of k).
print(f"accuracy of random weights: {acc:.1%}")  # >> About 10%: guessing one of 10 digits.
print("average cost:", ((all_out - Y) ** 2).sum(axis=1).mean().round(2))  # >> About 3. The next notebook brings it down to 0.08."""),

        ("md", """## 8. What you built

| Piece | In code |
|---|---|
| neuron | `sigmoid(w @ x + b)` |
| layer | `sigmoid(a @ W.T + b)` |
| network | a loop over layers |
| parameters | 13,002 numbers in `W` and `b` |
| cost | `((out - y) ** 2).sum()` |

The network is a function with 13,002 knobs, all random. **Next notebook:** turn the knobs with gradient descent and backpropagation."""),
    ],
}


TRAINING = {
    "slug": "training",
    "page": "code-training.html",
    "notebook": "training_numpy.ipynb",
    "eyebrow": "Practice · Training in Python",
    "title": "Gradient descent and backpropagation, <em>line by line.</em>",
    "doc_title": "Training a Neural Network with Backpropagation in NumPy, Line by Line",
    "lead": "A complete, runnable notebook for chapters 4–6. Roll a ball down a cost curve, write backpropagation by hand, check it numerically, "
            "train the 784 → 16 → 16 → 10 network on 60,000 digits to about 95% accuracy, and look at what it learned and where it fails. "
            "Only NumPy and Matplotlib. Hover over (or tap) any line of code to see what it does.",
    "learn": [
        "Take gradient descent steps and see the effect of the learning rate",
        "Write backpropagation for a network of any depth",
        "Check backpropagation against a numerical slope",
        "Train on mini-batches and plot cost and accuracy",
        "Inspect learned weights, mistakes and overconfidence",
    ],
    "cells": [
        ("md", """## 1. Setup and data

The same libraries and the same MNIST loader as in the first notebook."""),
        ("code", """%config InlineBackend.figure_format = "retina"  # >> Sharper plots in Jupyter.
import numpy as np  # >> Arrays and maths.
import matplotlib.pyplot as plt  # >> Charts.

""" + STYLE + "\n\n" + LOADER),

        ("md", """## 2. Gradient descent on one weight

Pretend the cost depends on a single weight: $C(w) = 0.05w^4 - 0.5w^2 + 0.3w + 2.2$, with two valleys. Gradient descent repeats $w \\leftarrow w - \\eta\\,\\frac{\\partial C}{\\partial w}$."""),
        ("code", """C = lambda w: 0.05 * w**4 - 0.5 * w**2 + 0.3 * w + 2.2  # >> The cost curve.
dC = lambda w: 0.2 * w**3 - 1.0 * w + 0.3  # >> Its slope (the derivative), worked out by hand.

def descend(w, eta, steps=25):  # >> Run gradient descent and record every position.
    path = [w]  # >>^
    for _ in range(steps):  # >>^
        w = w - eta * dC(w)  # >> One step against the slope.
        path.append(w)  # >>^
    return np.array(path)  # >>^

ws = np.linspace(-4, 4, 400)  # >> Points to draw the curve.
plt.plot(ws, C(ws), color="black")  # >> The cost curve.
for start, eta, col in [(3.4, 0.05, "tab:blue"), (3.4, 0.3, "tab:green"), (-0.6, 0.3, "tab:orange"), (3.4, 1.4, "tab:red")]:  # >> Four runs: different starts and learning rates.
    p = descend(start, eta)  # >>^
    plt.plot(p, C(p), "o-", ms=3, color=col, label=f"start {start}, η = {eta}")  # >> Each dot is one step.
plt.ylim(0, 5); plt.xlim(-4, 4); plt.legend(); plt.xlabel("w"); plt.ylabel("cost")  # >> Keep the view on the curve.
plt.show()  # >>^"""),
        ("md", """Small η crawls; a good η settles in a valley; the start decides **which** valley (the right one is a *local minimum*); a too-large η jumps around and can fly away."""),

        ("md", """## 3. The network and its forward pass

This time the forward pass **keeps every activation**, because backpropagation needs them."""),
        ("code", """sigmoid = lambda z: 1 / (1 + np.exp(-z))  # >> σ, written as a one-line function.
sizes = [784, 16, 16, 10]  # >> The lecture’s network.

def init(seed=0):  # >> Fresh random weights and zero biases.
    rng = np.random.default_rng(seed)  # >>^
    W = [rng.normal(0, 1 / np.sqrt(a), (b, a)) for a, b in zip(sizes[:-1], sizes[1:])]  # >> Matrices of shape (neurons out, neurons in).
    B = [np.zeros(b) for b in sizes[1:]]  # >>^
    return W, B  # >>^

def forward(X, W, B):  # >> X is a batch: one picture per row.
    acts = [X]  # >> acts[0] = the inputs; acts[l] = activations of layer l.
    for Wl, bl in zip(W, B):  # >>^
        acts.append(sigmoid(acts[-1] @ Wl.T + bl))  # >> a = σ(W a + b) for the whole batch at once.
    return acts  # >>^"""),

        ("md", """## 4. Backpropagation

Start at the output with $\\frac{\\partial C}{\\partial a} = 2(a - y)$. Multiply by the sigmoid’s slope $a(1-a)$ to get δ, the blame on each neuron’s weighted sum. The gradient of a weight is δ × the activation it multiplies. Pass δ back through the weights to the layer before, and repeat. That is the chain rule, layer by layer."""),
        ("code", """def backward(acts, Y, W):  # >> Gradients of the average cost over the batch.
    n = len(Y)  # >> Number of examples in the batch.
    delta = 2 * (acts[-1] - Y) * acts[-1] * (1 - acts[-1])  # >> Output layer: ∂C/∂a × ∂a/∂z.
    gW, gB = [None] * len(W), [None] * len(W)  # >> Empty slots, filled from the last layer backwards.
    for l in range(len(W) - 1, -1, -1):  # >> l = 2, 1, 0: from the output back to the input.
        gW[l] = delta.T @ acts[l] / n  # >> ∂C/∂w = δ × the incoming activation, averaged over the batch.
        gB[l] = delta.mean(axis=0)  # >> ∂C/∂b = δ (a bias multiplies 1).
        if l > 0:  # >> Not needed before the input layer.
            delta = (delta @ W[l]) * acts[l] * (1 - acts[l])  # >> Pass the blame back through the weights, times this layer’s sigmoid slope.
    return gW, gB  # >>^"""),

        ("md", """## 5. Check backpropagation numerically

Nudge one weight by a tiny ε, measure how the cost changes, and compare with what backpropagation says. If the two agree, the code is right."""),
        ("code", """W, B = init(1)  # >> A random network.
X, Y = X_train[:20], np.eye(10)[y_train[:20]]  # >> 20 examples and their one-hot labels.
cost = lambda W, B: ((forward(X, W, B)[-1] - Y) ** 2).sum(axis=1).mean()  # >> Average cost on these 20 examples.
gW, gB = backward(forward(X, W, B), Y, W)  # >> The gradient from backpropagation.

eps = 1e-5  # >> A tiny nudge.
for (l, j, i) in [(0, 3, 400), (1, 5, 7), (2, 2, 11)]:  # >> Three weights in three different layers.
    W[l][j, i] += eps; up = cost(W, B)  # >> Cost with the weight nudged up …
    W[l][j, i] -= 2 * eps; down = cost(W, B)  # >> … and nudged down.
    W[l][j, i] += eps  # >> Put it back.
    print(f"layer {l}: numerical {(up - down) / (2 * eps):+.6f}   backprop {gW[l][j, i]:+.6f}")  # >> The two slopes should match to several decimals."""),

        ("md", """## 6. Train on mini-batches

Shuffle the 60,000 training digits, cut them into mini-batches of 32, and take one gradient descent step per batch. One pass over all of them is an **epoch**. With learning rate η = 3 this takes about 10 seconds for 30 epochs."""),
        ("code", """W, B = init(0)  # >> Start again from random weights.
eta, batch, epochs = 3.0, 32, 30  # >> Learning rate, mini-batch size, number of passes.
Y_train, Y_test = np.eye(10)[y_train], np.eye(10)[y_test]  # >> One-hot versions of all labels.
rng = np.random.default_rng(0)  # >> For shuffling.
history = []  # >> (epoch, test cost, test accuracy) after each epoch.

def evaluate():  # >> Cost and accuracy on the 10,000 test digits.
    out = forward(X_test, W, B)[-1]  # >>^
    return ((out - Y_test) ** 2).sum(axis=1).mean(), (out.argmax(axis=1) == y_test).mean()  # >>^

history.append((0, *evaluate()))  # >> Before training: cost ≈ 3, accuracy ≈ 10%.
for ep in range(1, epochs + 1):  # >> 30 epochs.
    order = rng.permutation(len(X_train))  # >> A new random order every epoch.
    for k in range(0, len(order), batch):  # >> Walk through the data 32 at a time.
        idx = order[k:k + batch]  # >> Indices of this mini-batch.
        gW, gB = backward(forward(X_train[idx], W, B), Y_train[idx], W)  # >> Forward pass, then backpropagation.
        for l in range(len(W)):  # >> Gradient descent step for every layer:
            W[l] -= eta * gW[l]  # >> w ← w − η · gradient.
            B[l] -= eta * gB[l]  # >>^
    history.append((ep, *evaluate()))  # >>^
    if ep % 5 == 0 or ep == 1: print(f"epoch {ep:2d}: test cost {history[-1][1]:.3f}, accuracy {history[-1][2]:.1%}")  # >> Report now and then."""),

        ("code", """h = np.array(history)  # >> Columns: epoch, cost, accuracy.
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))  # >>^
ax[0].plot(h[:, 0], h[:, 1], "o-", ms=3); ax[0].set_title("test cost"); ax[0].set_xlabel("epoch")  # >> The cost falls fast, then slowly.
ax[1].plot(h[:, 0], h[:, 2], "o-", ms=3, color="tab:green"); ax[1].set_title("test accuracy"); ax[1].set_xlabel("epoch")  # >> From about 10% to about 95%.
plt.show()  # >>^"""),

        ("md", """## 7. What did it learn?

Each first-layer neuron has 784 weights: draw them as a 28 × 28 picture (green positive, red negative)."""),
        ("code", """fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))  # >> 16 neurons in hidden layer 1.
for k, ax in enumerate(axes.ravel()):  # >>^
    m = np.abs(W[0][k]).max()  # >> Use the same colour scale for + and −.
    ax.imshow(W[0][k].reshape(28, 28), cmap="RdYlGn", vmin=-m, vmax=m)  # >> Red = negative weight, green = positive.
    ax.set_title(k + 1, fontsize=8); ax.axis("off")  # >>^
plt.suptitle("first-layer weights: a few strokes, mostly noise")  # >>^
plt.show()  # >>^"""),

        ("md", """## 8. Its mistakes

Show some test digits it gets wrong, and the confusion matrix: rows are the true digit, columns what the network said."""),
        ("code", """pred = forward(X_test, W, B)[-1].argmax(axis=1)  # >> The network’s answer for every test digit.
wrong = np.where(pred != y_test)[0]  # >> Indices of the mistakes.
print(f"{len(wrong)} mistakes out of {len(y_test)}")  # >>^

fig, axes = plt.subplots(2, 10, figsize=(10, 2.6))  # >> 20 of them.
for i, ax in zip(wrong[:20], axes.ravel()):  # >>^
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray"); ax.axis("off")  # >>^
    ax.set_title(f"said {pred[i]}, was {y_test[i]}", fontsize=7)  # >>^
plt.show()  # >>^

conf = np.zeros((10, 10), int)  # >> Confusion matrix.
np.add.at(conf, (y_test, pred), 1)  # >> Add 1 at (true, said) for every test digit.
print(conf)  # >> Large numbers off the diagonal are common confusions (4 ↔ 9, 3 ↔ 5, …)."""),

        ("md", """## 9. It does not know what it does not know

Feed it random noise. It still answers, often confidently, because it was never taught to say “not a digit”."""),
        ("code", """noise = np.random.default_rng(1).random((5, 784))  # >> Five pictures of pure random pixels.
out = forward(noise, W, B)[-1]  # >>^
for o in out:  # >>^
    print("answer", o.argmax(), " confidence", o.max().round(2))  # >> An answer every time: confidence outside the training data means nothing."""),

        ("md", """## 10. What you built

Forward pass, cost, backpropagation, gradient descent: about 30 lines of NumPy, the same loop as `loss.backward()` and `optimizer.step()` in PyTorch. **Next notebook:** convolution filters and the three training problems of chapter 10."""),
    ],
}


PROBLEMS = {
    "slug": "problems",
    "page": "code-problems.html",
    "notebook": "filters_and_problems.ipynb",
    "eyebrow": "Practice · Filters & problems in Python",
    "title": "Convolution and training problems, <em>line by line.</em>",
    "doc_title": "Convolution Filters, Data Imbalance and Vanishing Gradients in Python, Line by Line",
    "lead": "A complete, runnable notebook for chapters 9–10. Slide filters over a digit and pool the feature maps, "
            "reproduce the fraud experiment (accuracy against recall, class weights, SMOTE, undersampling), "
            "measure vanishing gradients layer by layer and see why clipping stops exploding gradients. "
            "NumPy, Matplotlib and scikit-learn. Hover over (or tap) any line of code to see what it does.",
    "learn": [
        "Write a 2-D convolution and max-pooling by hand",
        "Show that accuracy hides a rare class",
        "Use class weights, SMOTE and undersampling, and tune the threshold",
        "Measure the gradient in every layer of a deep network",
        "Explain exploding gradients and gradient clipping",
    ],
    "cells": [
        ("md", """## 1. Setup"""),
        ("code", """%config InlineBackend.figure_format = "retina"  # >> Sharper plots in Jupyter.
import numpy as np  # >> Arrays and maths.
import matplotlib.pyplot as plt  # >> Charts.
from sklearn.linear_model import LogisticRegression  # >> A single sigmoid neuron, trained for us.
from sklearn.metrics import accuracy_score, recall_score, precision_score  # >> The three measures of chapter 10.

""" + STYLE + "\n\n" + LOADER.replace("X_train = ", "X_train = ").replace('print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  # >> Check the sizes.', 'print(X_test.shape)  # >> We only need a few test digits here.')),

        ("md", """## 2. Convolution by hand

A 3 × 3 filter slides over the 28 × 28 picture. At each position: multiply the 9 filter weights by the 9 pixels under them and add up: $y_{i,j} = \\sum_u \\sum_v k_{u,v}\\, x_{i+u,\\,j+v}$."""),
        ("code", """def conv2d(img, k):  # >> “Valid” convolution: the filter stays inside the picture, so 28 → 26.
    H, W = img.shape[0] - 2, img.shape[1] - 2  # >> Output size.
    out = np.zeros((H, W))  # >>^
    for i in range(H):  # >> Every row position …
        for j in range(W):  # >> … and every column position.
            out[i, j] = (img[i:i + 3, j:j + 3] * k).sum()  # >> The 3 × 3 patch times the filter, added up.
    return out  # >>^

def maxpool(fm):  # >> 2 × 2 max-pooling: keep the largest value of each block, 26 → 13.
    h, w = fm.shape[0] // 2, fm.shape[1] // 2  # >>^
    return fm[:2 * h, :2 * w].reshape(h, 2, w, 2).max(axis=(1, 3))  # >> Group pixels into 2 × 2 blocks and take each block’s maximum.

filters = {  # >> Four classic filters, written by hand (a CNN learns its own).
    "vertical edge": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),  # >> Responds where brightness changes from left to right.
    "horizontal edge": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),  # >> … from top to bottom.
    "diagonal": np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),  # >> … along a diagonal.
    "blur": np.ones((3, 3)) / 9,  # >> The average of the 9 pixels: smooths the picture.
}  # >>^

img = X_test[4].reshape(28, 28)  # >> One test digit (a 4).
fig, axes = plt.subplots(3, 5, figsize=(10, 6.2))  # >> Rows: input, feature map, pooled map.
axes[0, 0].imshow(img, cmap="gray"); axes[0, 0].set_title("input 28×28", fontsize=8)  # >>^
for c, (name, k) in enumerate(filters.items(), start=1):  # >> One column per filter.
    fm = np.maximum(0, conv2d(img, k))  # >> Convolution followed by ReLU.
    axes[0, c].imshow(k, cmap="RdYlGn"); axes[0, c].set_title(name, fontsize=8)  # >> The filter itself.
    axes[1, c].imshow(fm, cmap="Greens"); axes[1, c].set_title("conv + ReLU 26×26", fontsize=8)  # >> Bright where the pattern is found.
    axes[2, c].imshow(maxpool(fm), cmap="Greens"); axes[2, c].set_title("max-pool 13×13", fontsize=8)  # >> Smaller, keeps the strongest responses.
for ax in axes.ravel(): ax.axis("off")  # >>^
plt.show()  # >>^
print("weights in one 3×3 filter: 9   ·   weights from 784 pixels to 16 dense neurons:", 784 * 16)  # >> Why CNNs need far fewer weights."""),

        ("md", """## 3. Data imbalance: 10,000 payments, 2% fraud

Simulate two features per payment (think: scaled amount and time of night). Frauds are shifted, but overlap with normal payments. 3,000 payments are held out for testing."""),
        ("code", """rng = np.random.default_rng(2024)  # >> Fixed seed: same data every run.
n = 10_000  # >> Payments.
y = (np.arange(n) % 50 == 0).astype(int)  # >> Every 50th payment is fraud: 2%.
X = rng.normal(size=(n, 2))  # >> Legitimate payments: centred at (0, 0).
X[y == 1] = rng.normal(size=(y.sum(), 2)) * 0.95 + [1.5, 1.2]  # >> Frauds: shifted, but overlapping.
perm = rng.permutation(n); X, y = X[perm], y[perm]  # >> Shuffle.
Xtr, ytr, Xte, yte = X[:7000], y[:7000], X[7000:], y[7000:]  # >> 7,000 to train, 3,000 to test.
print("frauds in training set:", ytr.sum(), "  in test set:", yte.sum())  # >> About 140 and 60."""),

        ("code", """def smote(X, y, k=3, seed=0):  # >> SMOTE (Chawla et al., 2002), written out.
    r = np.random.default_rng(seed)  # >>^
    F = X[y == 1]  # >> The minority (fraud) points.
    need = (y == 0).sum() - len(F)  # >> How many synthetic points to reach 50 : 50.
    d = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)  # >> Squared distance between every pair of frauds.
    nb = np.argsort(d, axis=1)[:, 1:k + 1]  # >> The k nearest fraud neighbours of each fraud (column 0 is itself).
    a = r.integers(len(F), size=need)  # >> Pick a random fraud …
    b = nb[a, r.integers(k, size=need)]  # >> … and one of its neighbours.
    t = r.random((need, 1))  # >> A random position between them.
    new = F[a] + t * (F[b] - F[a])  # >> The synthetic frauds.
    return np.vstack([X, new]), np.concatenate([y, np.ones(need, int)])  # >> Original data plus the new points.

def undersample(X, y, ratio=4, seed=0):  # >> Keep all frauds and only 4 legitimate payments per fraud.
    r = np.random.default_rng(seed)  # >>^
    keep = np.concatenate([np.where(y == 1)[0], r.choice(np.where(y == 0)[0], ratio * y.sum(), replace=False)])  # >>^
    return X[keep], y[keep]  # >>^

def report(name, model):  # >> One line of the results table, on the untouched test set.
    p = model.predict(Xte)  # >> Threshold 0.5.
    print(f"{name:30s} accuracy {accuracy_score(yte, p):6.1%}   recall {recall_score(yte, p):4.0%}   "  # >>^
          f"precision {precision_score(yte, p, zero_division=0):4.0%}   caught {int(((p == 1) & (yte == 1)).sum())} of {yte.sum()}")  # >>^

print(f"{'always legitimate':30s} accuracy {(yte == 0).mean():6.1%}   recall   0%")  # >> The lazy model: high accuracy, catches nothing.
report("trained on the data as it is", LogisticRegression().fit(Xtr, ytr))  # >> Accuracy about 98%, recall tiny.
report("+ class weights", LogisticRegression(class_weight="balanced").fit(Xtr, ytr))  # >> A missed fraud now costs about 49× more.
report("+ SMOTE (training set only)", LogisticRegression().fit(*smote(Xtr, ytr)))  # >> Synthetic frauds in the training set only.
report("+ undersampling 4:1", LogisticRegression().fit(*undersample(Xtr, ytr)))  # >> Fewer legitimate payments."""),

        ("md", """### Moving the threshold

The plain model outputs a probability. Lowering the threshold from 0.5 catches more fraud but raises more false alarms."""),
        ("code", """plain = LogisticRegression().fit(Xtr, ytr)  # >> The model trained on the data as it is.
prob = plain.predict_proba(Xte)[:, 1]  # >> Its fraud probability for each test payment.
ts = np.linspace(0.01, 0.6, 60)  # >> Thresholds to try.
rec = [recall_score(yte, prob >= t) for t in ts]  # >> Recall at each threshold.
pre = [precision_score(yte, prob >= t, zero_division=0) for t in ts]  # >> Precision at each threshold.
plt.plot(ts, rec, label="recall"); plt.plot(ts, pre, label="precision")  # >>^
plt.axvline(0.5, color="gray", ls="--", lw=1)  # >> The default threshold.
plt.xlabel("threshold"); plt.legend(); plt.title("the same model, different thresholds")  # >>^
plt.show()  # >> Where to stand is a business decision: missed fraud against annoyed customers."""),

        ("md", """## 4. Vanishing gradients, measured

A 10-layer network of 32 neurons per layer. Push 64 random inputs through, backpropagate a cost, and record the average size of the weight gradients in each layer."""),
        ("code", """def grad_sizes(act, depth=10, width=32, seed=3):  # >> Average |∂C/∂w| in every layer.
    r = np.random.default_rng(seed)  # >>^
    f, df = {"sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda a: a * (1 - a)),  # >> The activation and its slope (written with a = f(z)).
             "relu": (lambda z: np.maximum(0, z), lambda a: (a > 0).astype(float))}[act]  # >>^
    scale = np.sqrt(2 / width) if act == "relu" else 1 / np.sqrt(width)  # >> He initialisation for ReLU, Xavier for sigmoid.
    W = [r.normal(0, scale, (width, width)) for _ in range(depth)]  # >>^
    acts = [r.normal(size=(64, width))]  # >> 64 random inputs.
    for Wl in W: acts.append(f(acts[-1] @ Wl.T))  # >> Forward pass, keeping every activation.
    delta = (acts[-1] - r.normal(size=acts[-1].shape)) * df(acts[-1])  # >> Output error towards random targets.
    sizes = []  # >>^
    for l in range(depth - 1, -1, -1):  # >> Backpropagate.
        sizes.append(np.abs(delta.T @ acts[l]).mean() / 64)  # >> Size of this layer’s weight gradient.
        delta = (delta @ W[l]) * df(acts[l])  # >> Pass back, times the slope of the layer below.
    return sizes[::-1]  # >> Layer 1 first.

layers = np.arange(1, 11)  # >>^
plt.bar(layers - 0.2, grad_sizes("sigmoid"), 0.4, label="sigmoid")  # >> Shrinks layer after layer.
plt.bar(layers + 0.2, grad_sizes("relu"), 0.4, label="ReLU")  # >> Stays about the same size.
plt.yscale("log"); plt.xlabel("layer (1 = next to the input)"); plt.ylabel("gradient size"); plt.legend()  # >> Log scale: each step is 10×.
plt.show()  # >>^
s = grad_sizes("sigmoid"); print(f"sigmoid: last layer / first layer = {s[-1] / s[0]:,.0f}×")  # >> The first layers barely learn."""),

        ("md", """## 5. Exploding gradients and clipping

In an RNN the same factor is multiplied once per time step. A factor of 1.1 over 100 steps is about 14,000. Gradient clipping caps the gradient’s length: $\\text{if } \\lVert g \\rVert > c:\\ g \\leftarrow c\\,g/\\lVert g \\rVert$."""),
        ("code", """steps = np.arange(101)  # >> 0 … 100 time steps.
for f in [0.9, 1.0, 1.1]:  # >> Three factors per step.
    plt.plot(steps, f ** steps, label=f"factor {f}")  # >> The product after t steps.
plt.yscale("log"); plt.xlabel("steps back in time"); plt.legend(); plt.title("vanishing (0.9) and exploding (1.1)")  # >>^
plt.show()  # >>^

def clip(g, c=1.0):  # >> Gradient clipping by length (norm).
    norm = np.linalg.norm(g)  # >> The length of the gradient vector.
    return g * (c / norm) if norm > c else g  # >> Too long: shrink to length c, same direction.

g = np.array([300.0, -400.0])  # >> An exploding gradient of length 500.
print("before:", g, " length", np.linalg.norm(g))  # >>^
print("after: ", clip(g), " length", np.linalg.norm(clip(g)))  # >> Length 1, direction unchanged."""),

        ("md", """## 6. What you built

| Problem | Symptom | Fix |
|---|---|---|
| data imbalance | high accuracy, low recall | class weights, SMOTE, threshold |
| vanishing gradient | first layers do not learn | ReLU, good initialisation, residual connections |
| exploding gradient | loss jumps or becomes NaN | gradient clipping, smaller η |

Symptom → cause → fix: say all three."""),
    ],
}

LESSONS = [NETWORK, TRAINING, PROBLEMS]
