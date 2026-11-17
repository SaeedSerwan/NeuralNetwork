<p align="center">
  <img src="assets/img/utb-logo.svg" alt="University of Technology Bahrain" width="64">
</p>

<h1 align="center">Neural Network Lab</h1>

<p align="center">
  <strong>Interactive lessons on how neural networks learn: neurons, layers, cost, gradient descent, backpropagation, CNN, RNN, GAN and training problems.</strong>
</p>

<p align="center">
  <a href="https://saeedserwan.github.io/NeuralNetwork/"><strong>▶ Open the lab</strong></a>
</p>

---

## Lessons

| Page | What you can do |
|---|---|
| [Overview](https://saeedserwan.github.io/NeuralNetwork/) | 40 handwritten 3s, copying nature, the learning loop with one weight |
| [Shapes in 5 minutes](https://saeedserwan.github.io/NeuralNetwork/shapes.html) | Draw a circle, square or triangle; follow inputs, weights, bias, activation, forward and back propagation step by step |
| [Neurons & layers](https://saeedserwan.github.io/NeuralNetwork/neuron.html) | Pixels as numbers, weighted sums, sigmoid and ReLU, the bias, counting 13,002 parameters, a = σ(Wa + b) |
| [How it learns](https://saeedserwan.github.io/NeuralNetwork/learning.html) | Cost of one example, gradient descent in 1-D and 2-D, learning rate, local minima, backpropagation number by number |
| [The digit network](https://saeedserwan.github.io/NeuralNetwork/digits.html) | Draw a digit for a 784-16-16-10 network, see what it learned, train it live in the browser, read its mistakes |
| [Playground](https://saeedserwan.github.io/NeuralNetwork/playground.html) | Build a small network and watch its decision boundary and hidden neurons change; eight challenges |
| [Types of networks](https://saeedserwan.github.io/NeuralNetwork/types.html) | Convolution filters and pooling, RNN memory vs LSTM, a 1-D GAN, attention, name that network |
| [Training problems](https://saeedserwan.github.io/NeuralNetwork/problems.html) | Fraud detection with class weights, SMOTE and thresholds; vanishing and exploding gradients; clipping |
| [A network in Python](https://saeedserwan.github.io/NeuralNetwork/code-network.html) · [Training in Python](https://saeedserwan.github.io/NeuralNetwork/code-training.html) · [Filters & problems in Python](https://saeedserwan.github.io/NeuralNetwork/code-problems.html) | Runnable notebooks explained line by line (NumPy, Matplotlib, scikit-learn), also in [`notebooks/`](notebooks) and on Google Colab |

Hover (or tap) any underlined term or any part of a formula to see an explanation.

## Run locally

Open `index.html` in a browser. No installation needed.

To rebuild the pages after editing `tools/pages/` or `tools/walkthrough_content.py`:

```bash
python tools/build_pages.py
pip install numpy matplotlib scikit-learn nbformat nbclient ipykernel markdown pillow
python tools/build_walkthroughs.py
```

## Author

**Eng. Saeed Serwan Abdulsattar** · University of Technology Bahrain · MIT License
