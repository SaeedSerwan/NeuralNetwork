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


if __name__ == "__main__":
    only = sys.argv[1:]
    for n, c in enumerate(CHAPTERS):
        if not only or c[0] in only:
            build(n)
