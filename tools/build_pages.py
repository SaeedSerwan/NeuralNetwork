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

S = '<script src="assets/js/data/samples.js"></script>\n'
N = '<script src="assets/js/data/net.js"></script>\n'
D = '<script src="assets/js/data/digits.js"></script>\n'
J = lambda *f: "".join(f'<script src="assets/js/{x}"></script>\n' for x in f)

# (file, chapter title, short name, page <title>, scripts before the main script, main script, description, bridge to this chapter from the previous one)
CHAPTERS = [
    ("index.html", "Why neural networks?", "Why networks", "Why neural networks? Learning instead of rules", S, "overview.js",
     "Lecture 6, chapter 1: why some tasks need learning, copying the brain, and the learning loop.", ""),
    ("neuron.html", "One neuron", "One neuron", "One neuron: multiply, add, squish", S + N, "neuron.js",
     "Lecture 6, chapter 2: what one artificial neuron computes: weights, bias, sigmoid and ReLU.", "We decided to copy the brain. Start with its smallest unit."),
    ("network.html", "From neurons to a network", "A network", "From neurons to a network", S + N + J("digits.js"), "neuron.js",
     "Lecture 6, chapter 3: layers, the 784-16-16-10 digit network, counting parameters, a = σ(Wa + b) and a live forward pass.", "One neuron answers one small question. Put many in layers."),
    ("learning.html", "Measuring mistakes, rolling downhill", "Cost & descent", "The cost function and gradient descent", S + N, "learning.js",
     "Lecture 6, chapter 4: the cost function and gradient descent, with the learning rate and local minima.", "The network has 13,002 knobs. How wrong is it, and which way should each knob turn?"),
    ("backprop.html", "Backpropagation", "Backprop", "Backpropagation: blame flows backwards", S + N, "learning.js",
     "Lecture 6, chapter 5: backpropagation number by number with the chain rule.", "Gradient descent needs the slope of every knob. Where do the slopes come from?"),
    ("training.html", "Training: the whole loop", "Training", "Training: the whole loop", S + N + D + J("digits.js", "learning.js"), "overview.js",
     "Lecture 6, chapter 6: the full training loop animated, training the digit network live, learned weights, mistakes and overconfidence.", "Every piece is ready. Run the whole loop."),
    ("shapes.html", "Recap in five minutes", "Recap", "Recap: a neural network in five minutes", "", "shapes.js",
     "Lecture 6, chapter 7: the whole story once more on circles, squares and triangles.", "You have seen every piece. Now the whole story once more, quickly."),
    ("playground.html", "Playground", "Playground", "Neural network playground", "", "playground.js",
     "Lecture 6, chapter 8: experiment with layers, activations, learning rate and overfitting.", "Time to experiment with everything you know."),
    ("types.html", "Types of networks", "Types", "Types of networks: CNN, RNN, GAN, transformers", S, "types.js",
     "Lecture 6, chapter 9: CNN, RNN, LSTM, GAN and transformers: same neurons, different wiring.", "Same neurons and training, different wiring for images, sequences and creation."),
    ("problems.html", "Training problems", "Problems", "Training problems: imbalance, vanishing and exploding gradients", "", "problems.js",
     "Lecture 6, chapter 10: data imbalance, vanishing and exploding gradients, and a summary of the lecture.", "When training goes wrong, and how to fix it."),
]


def build(n):
    fname, ctitle, short, title, extra, js, desc, _ = CHAPTERS[n]
    body = (ROOT / "tools" / "pages" / fname).read_text(encoding="utf-8")
    body = re.sub(r'\s*<a class="next-card".*?</a>\n', "\n", body, flags=re.S)
    k = [0]
    def num(m):
        k[0] += 1
        return f'<span class="chap">{k[0]:02d}</span>'
    body = re.sub(r'<span class="chap">\d+</span>', num, body)
    secs = re.findall(r'<section id="([^"]+)" class="chapter">\s*<h2><span class="chap">\d+</span>(.*?)</h2>', body, re.S)
    toc = "\n".join(f'        <li><a href="#{i}">{re.sub(r"<[^>]+>", "", t).strip()}</a></li>' for i, t in secs)
    chap = "\n".join(
        (f'      <li class="cur"><a href="{f}">{c}</a>\n        <ol>\n{toc}\n        </ol>\n      </li>' if i == n else f'      <li><a href="{f}">{c}</a></li>')
        for i, (f, c, *_r) in enumerate(CHAPTERS))
    nav = '    <nav class="chapnav">\n'
    if n > 0:
        pf, pc = CHAPTERS[n - 1][0], CHAPTERS[n - 1][1]
        nav += f'      <a class="prev" href="{pf}"><span class="next-eyebrow">← Chapter {n}</span><span class="next-title">{pc}</span></a>\n'
    if n + 1 < len(CHAPTERS):
        nf, nc, nb = CHAPTERS[n + 1][0], CHAPTERS[n + 1][1], CHAPTERS[n + 1][7]
        nav += f'      <a class="next-card" href="{nf}"><span class="next-eyebrow">Next · Chapter {n + 2}</span><span class="next-title">{nc} →</span><span class="next-text">{nb}</span></a>\n'
    else:
        nav += '      <a class="next-card" href="code-network.html"><span class="next-eyebrow">Practice</span><span class="next-title">Build it in Python →</span><span class="next-text">Three runnable notebooks: a network in NumPy, training it with backpropagation, and filters and training problems.</span></a>\n'
    nav += '    </nav>\n'
    html = f'''<!DOCTYPE html>
<html lang="en">
{head_html(f"{NAME} · {title}", desc)}
<body data-page="{fname[:-5]}">

{header_html(fname)}

<div class="layout">
  <aside class="toc" aria-label="Contents">
    <p class="toc-title">Lecture 6 · chapters</p>
    <ol class="chaplist">
{chap}
    </ol>
  </aside>

  <main>
{body.rstrip()}
{nav}  </main>
</div>

{FOOTER}

<script src="assets/vendor/katex/katex.min.js"></script>
<script src="assets/vendor/katex/auto-render.min.js"></script>
<script src="assets/js/lib.js"></script>
<script src="assets/js/anim.js"></script>
<script src="assets/js/tips.js"></script>
{extra}<script src="assets/js/{js}"></script>
<script src="assets/js/quizzes.js"></script>
</body>
</html>
'''
    (ROOT / fname).write_text(html, encoding="utf-8")
    print(f"{fname}: {len(secs)} sections")


V3_BODY = """<div class="v3">
  <aside class="v3-left">
    <p class="v3-title">Draw a digit</p>
    <div id="v3-pad" class="v3-padbox"></div>
    <div class="v3-btns"><button id="v3-clear">Clear</button><button id="v3-rand">Random digit</button></div>
    <div class="v3-down"><div id="v3-small" class="plot"></div><span>Downsampled drawing<br>28 × 28 = 784 inputs</span></div>
    <div class="v3-guess">
      <div><span>First guess</span><b id="v3-g1">–</b><em id="v3-g1p"></em></div>
      <div><span>Second guess</span><b id="v3-g2">–</b><em id="v3-g2p"></em></div>
    </div>
    <p class="v3-small">784 → 300 → 100 → 10 · ReLU, softmax · 266,610 parameters · 98.8% on 10,000 test digits</p>
  </aside>
  <div class="v3-stage">
    <canvas id="v3-canvas" aria-label="3D view of the neural network"></canvas>
    <div id="v3-tip" class="v3-tip" hidden></div>
    <p class="v3-help">drag to rotate · scroll to zoom · click a neuron</p>
  </div>
  <aside class="v3-right">
    <p class="v3-title">Neuron inspector</p>
    <div id="v3-ins"></div>
  </aside>
  <div class="v3-bottom">
    <div class="v3-group v3-modes"><button data-m="forward" class="on">Forward pass</button><button data-m="back">Backward pass</button><button id="v3-fwd" class="ghost">Replay</button></div>
    <div class="v3-group"><span>correct digit</span><div id="v3-labels" class="v3-labels"></div><button id="v3-learn">Learn from this drawing</button></div>
    <div class="v3-group"><button id="v3-train">▶ Train from random weights</button><button id="v3-restore" class="ghost">Restore trained weights</button><div id="v3-acc" class="plot v3-acc"></div><span id="v3-train-st">trained weights (98.8% on 10,000 test digits)</span></div>
    <div class="v3-group"><label><input type="checkbox" id="v3-paths" checked> signal paths</label><label><input type="checkbox" id="v3-rot"> auto-rotate</label><button id="v3-reset-view" class="ghost">Reset view</button></div>
    <p id="v3-nar" class="v3-nar"></p>
  </div>
</div>

<div class="layout v3-notes">
  <main>
    <section id="read" class="chapter">
      <h2><span class="chap">01</span>How to read the 3D view</h2>
      <p>The bottom layer is your drawing: 784 squares, one per pixel. Above it are the 300 neurons of the first hidden layer, the 100 of the second, and at the top the 10 output neurons, one per digit. A brighter square is a more active neuron. The lines show the strongest contributions flowing upwards: green adds to the neuron above, red subtracts. Click any neuron to see its three steps, exactly as in <a href="neuron.html">chapter 2</a>: <b>weighted input</b> (each input × its weight), <b>calculation</b> (add them up, add the bias) and <b>output</b> (ReLU, or softmax for the 10 outputs).</p>
    </section>
    <section id="backward" class="chapter">
      <h2><span class="chap">02</span>Backward pass and learning</h2>
      <p>Choose the correct digit and press <b>Backward pass</b>. The error starts at the top: δ = output − wanted, orange where the output is too high, blue where it is too low. It then flows down the same connections, weighted by the same weights, layer by layer: this is <a href="backprop.html">backpropagation (chapter 5)</a>. Neurons that were inactive (ReLU = 0) pass no blame. <b>Learn from this drawing</b> then moves every weight by −η × input × δ and shows how the probability of the correct digit changes.</p>
      <p><b>Train from random weights</b> wipes all 266,610 knobs and trains the network in your browser on 5,000 digits with mini-batches of 16, while you watch the layers reorganise and the test accuracy rise: the loop of <a href="training.html">chapter 6</a> on a network 20 times larger.</p>
    </section>
    <section id="credit" class="chapter">
      <h2><span class="chap">03</span>Credit</h2>
      <p>This view is modelled on Adam W. Harley, “An Interactive Node-Link Visualization of Convolutional Neural Networks”, ISVC 2015 (<a href="https://adamharley.com/nn_vis/" target="_blank" rel="noopener">adamharley.com/nn_vis</a>), which uses the same 784 → 300 → 100 → 10 layer sizes. The network, rendering, backward pass and in-browser training on this page were written from scratch for this lab.</p>
    </section>
  </main>
</div>
"""


def build_3d():
    html = f'''<!DOCTYPE html>
<html lang="en">
{head_html(f"{NAME} · The network in 3D", "A 784-300-100-10 neural network in 3D: draw a digit, watch it flow through the layers, inspect any neuron, see backpropagation and train it in your browser.")}
<body data-page="network3d" class="dark3d">

{header_html("network3d.html")}

{V3_BODY}
{FOOTER}

<script src="assets/vendor/katex/katex.min.js"></script>
<script src="assets/vendor/katex/auto-render.min.js"></script>
<script src="assets/js/lib.js"></script>
<script src="assets/js/anim.js"></script>
<script src="assets/js/tips.js"></script>
<script src="assets/js/data/samples.js"></script>
<script src="assets/js/data/digits.js"></script>
<script src="assets/js/data/net3d.js"></script>
<script src="assets/js/network3d.js"></script>
<script>UL.start();</script>
</body>
</html>
'''
    (ROOT / "network3d.html").write_text(html, encoding="utf-8")
    print("network3d.html")


if __name__ == "__main__":
    only = sys.argv[1:]
    for n, c in enumerate(CHAPTERS):
        if not only or c[0] in only:
            build(n)
    if not only or "network3d.html" in only:
        build_3d()
