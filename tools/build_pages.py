"""
Assemble the interactive lesson pages of the Neural Network Lab.

Each lesson body lives in tools/pages/<page>.html. This script wraps it with the shared
<head>, header, table of contents (built from the <section id> / <h2> of the body) and footer,
and writes <page>.html at the root of the site.

Usage:  python tools/build_pages.py
Author: Eng. Saeed Serwan Abdulsattar
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from site_shell import header_html, FOOTER, head_html, NAME  # noqa: E402

PAGES = [
    ("index.html", "Overview", "How neural networks learn", "overview.js",
     "An interactive, minimal lab for learning how neural networks and deep learning work: neurons, layers, cost functions, gradient descent, backpropagation, CNN, RNN, GAN and training problems."),
    ("shapes.html", "Shapes in 5 minutes", "Circle, square or triangle? A neural network in 5 minutes", "shapes.js",
     "A five-minute interactive tour of how a neural network works: a network that tells circles, squares and triangles apart, with inputs, weights, bias, activation, forward propagation and backpropagation."),
    ("neuron.html", "Neurons & layers", "Neurons, weights, biases and layers", "neuron.js",
     "What a neuron holds, what a weighted sum and a bias do, sigmoid and ReLU, counting parameters and one layer as a matrix multiply, all interactive."),
    ("learning.html", "How it learns", "Cost functions, gradient descent and backpropagation", "learning.js",
     "How a neural network learns: the cost function, gradient descent on a landscape, the learning rate, local minima and backpropagation step by step."),
    ("digits.html", "The digit network", "A network that reads handwritten digits", "digits.js",
     "Draw a digit and watch a 784-16-16-10 network read it, train the same network live in your browser, and see what it actually learned."),
    ("playground.html", "Playground", "Neural network playground", "playground.js",
     "Train a small neural network on two-dimensional data and watch its decision boundary, hidden neurons and loss change step by step."),
    ("types.html", "Types of networks", "CNN, RNN, GAN and friends", "types.js",
     "Multilayer perceptrons, convolutional networks, recurrent networks, GANs and transformers: what each is for, with interactive filters, memory and a forger and a detective."),
    ("problems.html", "Training problems", "Data imbalance, vanishing and exploding gradients", "problems.js",
     "Why 98% accuracy can mean nothing, how class weights and SMOTE help, and why gradients vanish or explode in deep networks, all interactive."),
]


S = '<script src="assets/js/data/samples.js"></script>\n'
N = '<script src="assets/js/data/net.js"></script>\n'
EXTRA = {"index.html": S, "neuron.html": S + N, "learning.html": S + N, "digits.html": S + N + '<script src="assets/js/data/digits.js"></script>\n', "types.html": S}


def build(fname, toc_title, title, js, desc):
    extra = EXTRA.get(fname, "")
    body = (ROOT / "tools" / "pages" / fname).read_text(encoding="utf-8")
    secs = re.findall(r'<section id="([^"]+)" class="chapter">\s*<h2><span class="chap">\d+</span>(.*?)</h2>', body, re.S)
    toc = "\n".join(f'      <li><a href="#{i}">{re.sub(r"<[^>]+>", "", t).strip()}</a></li>' for i, t in secs)
    html = f'''<!DOCTYPE html>
<html lang="en">
{head_html(f"{NAME} · {title}", desc)}
<body>

{header_html(fname)}

<div class="layout">
  <aside class="toc" aria-label="Contents">
    <p class="toc-title">{toc_title}</p>
    <ol>
{toc}
    </ol>
  </aside>

  <main>
{body.rstrip()}
  </main>
</div>

{FOOTER}

<script src="assets/vendor/katex/katex.min.js"></script>
<script src="assets/vendor/katex/auto-render.min.js"></script>
<script src="assets/js/lib.js"></script>
<script src="assets/js/tips.js"></script>
{extra}<script src="assets/js/{js}"></script>
</body>
</html>
'''
    (ROOT / fname).write_text(html, encoding="utf-8")
    print(f"{fname}: {len(secs)} sections")


if __name__ == "__main__":
    only = sys.argv[1:]
    for p in PAGES:
        if not only or p[0] in only:
            build(*p)
