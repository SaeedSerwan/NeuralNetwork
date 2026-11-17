/* Chapter quizzes and exercises. Each chapter only asks about ideas taught up to that chapter.
   Author: Eng. Saeed Serwan Abdulsattar */
(function () {
  'use strict';
  const UL = window.UL, page = document.body.dataset.page;
  const Q = {
    index: [
      { q: 'Why do we not write if-statements to read handwritten digits?', o: ['Computers cannot compare numbers', 'Nobody can write exact rules that cover every way of writing a 3; learning from examples works better', 'It would be too fast', 'Digits are not numbers'], a: 1, why: 'No two 3s have the same pixels. The rules are too many and too fuzzy to write by hand.' },
      { q: 'Which task is better solved with ordinary code than with a neural network?', o: ['Reading handwriting', 'Recognising faces', 'Adding 10% VAT to a bill', 'Understanding speech'], a: 2, why: 'VAT has an exact rule. The others have no rules we can write down.' },
      { q: 'In the brain, learning changes…', o: ['the number of neurons', 'the strength of the connections between neurons', 'the colour of the neurons', 'nothing'], a: 1, why: 'Artificial networks copy this: learning changes the weights of the connections.' },
      { q: 'Put the learning loop in order.', o: ['Nudge, guess, measure, direction', 'Guess, measure how wrong, find the direction, nudge the weights', 'Measure, nudge, guess, direction', 'Guess, nudge, measure, direction'], a: 1, why: 'Every later chapter fills in one of these four steps.' },
    ],
    neuron: [
      { q: 'What does a neuron in the input layer of a digit network hold?', o: ['A weight', 'The brightness of one pixel, between 0 and 1', 'A digit from 0 to 9', 'A bias'], a: 1, why: 'Each input neuron simply holds one pixel’s brightness: its activation.' },
      { q: 'Inputs (1, 0, 1) with weights (2, 5, −1) and bias 0. Weighted sum?', o: ['6', '1', '2', '−1'], a: 1, why: '1·2 + 0·5 + 1·(−1) = 1. The input that is 0 contributes nothing.' },
      { q: 'A neuron has bias −10. What does that mean?', o: ['It is switched off', 'Its weighted sum must be above about 10 before it becomes active', 'All its weights are −10', 'It fires 10 times'], a: 1, why: 'The bias is added before σ: σ(sum − 10) is above 0.5 only when the sum is above 10.' },
      { q: 'σ(0) = ?', o: ['0', '0.25', '0.5', '1'], a: 2, why: '1 / (1 + e⁰) = 1 / 2.' },
      { q: 'ReLU(−3) = ?', o: ['−3', '0', '3', '0.05'], a: 1, why: 'ReLU keeps positive numbers and turns negative ones into 0.' },
    ],
    network: [
      { q: 'How many weights connect a layer of 784 neurons to a layer of 16?', o: ['800', '784 + 16', '12,544', '13,002'], a: 2, why: 'Every one of the 16 neurons has one weight per input: 784 × 16 = 12,544. (13,002 is the whole network, biases included.)' },
      { q: 'In a = σ(Wa + b), what is row k of W?', o: ['The activations of layer k', 'All the weights going into neuron k', 'The biases', 'The output'], a: 1, why: 'Multiplying row k by the input column gives neuron k’s weighted sum.' },
      { q: 'Why do networks need activation functions between layers?', o: ['To make them faster', 'Without them, all layers together are still just one linear model', 'To count the parameters', 'To store the pixels'], a: 1, why: 'Straight lines on straight lines are still straight; try “none” in the curve-fitting lab.' },
      { q: 'You draw a 7 and output neuron 7 has activation 0.93 while all others are below 0.1. The network’s answer is…', o: ['0.93', '7', 'unknown until it is trained', '10'], a: 1, why: 'The answer is the index of the brightest output neuron.' },
      { q: 'What is the “hope” behind hidden layers?', o: ['They make the network smaller', 'Each layer builds slightly bigger pieces (edges → parts → digits) from the layer below', 'They store the training data', 'They choose the learning rate'], a: 1, why: 'Whether a trained network really does this is something you will check in chapter 6.' },
    ],
    learning: [
      { q: 'The cost of one example is 0. What does that mean?', o: ['The network is untrained', 'Every output is exactly what we wanted', 'The gradient is very large', 'All weights are 0'], a: 1, why: 'Each squared difference is 0 only when the output equals the wanted value.' },
      { q: 'What are the inputs of the cost function of the digit network?', o: ['The 784 pixels', 'The 13,002 weights and biases', 'The 10 outputs', 'The learning rate'], a: 1, why: 'The training data is fixed; the knobs are what the cost depends on.' },
      { q: 'The slope ∂C/∂w of a weight is positive. Gradient descent will…', o: ['increase w', 'decrease w', 'leave w unchanged', 'set w to 0'], a: 1, why: 'Positive slope: the cost rises when w rises, so w ← w − η · (positive) makes w smaller.' },
      { q: 'What happens with a learning rate that is much too large?', o: ['Training is slow but safe', 'The steps overshoot and the cost jumps around or grows', 'The network gets more layers', 'Nothing'], a: 1, why: 'Try η = 1.5 on the 1-D curve: the ball jumps over the valley.' },
      { q: 'Gradient descent stops in a valley that is not the lowest. This is…', o: ['a global minimum', 'a local minimum', 'overfitting', 'backpropagation'], a: 1, why: 'Try a different start, a larger learning rate or momentum.' },
    ],
    backprop: [
      { q: 'What does backpropagation compute?', o: ['The output of the network', 'The slope of the cost for every weight and bias', 'The learning rate', 'The test accuracy'], a: 1, why: 'Backpropagation computes the gradient; gradient descent (chapter 4) then uses it.' },
      { q: 'In ∂C/∂w = ∂z/∂w · ∂a/∂z · ∂C/∂a, what is ∂z/∂w?', o: ['The learning rate', 'The activation that the weight multiplies', 'The cost', 'Always 1'], a: 1, why: 'z = w·a + …, so changing w by a little changes z by a times that little.' },
      { q: 'Why does backpropagation need the forward pass first?', o: ['It does not', 'The local slopes depend on the activations computed in the forward pass', 'To pick the learning rate', 'To shuffle the data'], a: 1, why: 'For example ∂a/∂z = a(1 − a) and ∂z/∂w = the incoming activation.' },
      { q: 'The output was too low (a = 0.49, want 1). The sign of ∂C/∂a is…', o: ['positive', 'negative', 'zero', 'undefined'], a: 1, why: '2(a − y) = 2(0.49 − 1) < 0: raising a lowers the cost.' },
    ],
    training: [
      { q: 'Put one training round in order.', o: ['Update, forward, cost, backward', 'Forward pass, cost, backpropagation, gradient descent step', 'Backward, forward, update, cost', 'Cost, update, forward, backward'], a: 1, why: 'Guess, measure, find the direction, nudge: the loop from chapter 1.' },
      { q: 'Why train on mini-batches of 32 digits instead of all 60,000 at once?', o: ['It is more accurate', 'Each step is much cheaper, so we take many more steps; on average they still go downhill', 'It avoids the cost function', 'MNIST requires it'], a: 1, why: 'The mini-batch gradient is a noisy but cheap estimate of the full gradient.' },
      { q: 'One epoch means…', o: ['one gradient step', 'one pass over the whole training set', 'one hidden layer', 'one test digit'], a: 1, why: 'The digit network was trained for 30 epochs.' },
      { q: 'The first-layer weight pictures mostly look like noise. What does that tell you?', o: ['The network is broken', 'It found a working solution that is not the tidy edges-and-loops one we hoped for', 'The learning rate was 0', 'The pictures are drawn wrongly'], a: 1, why: 'About 95% correct with messy features: it minimises the cost, not our idea of neatness.' },
      { q: 'Why does the network still name a digit for random noise?', o: ['Noise looks like a digit', 'It was only trained on digits and has no way to say “not a digit”', 'Its bias is too large', 'It is a bug'], a: 1, why: 'Ten outputs always come out and one is always the largest. Outside its training data its confidence means nothing.' },
    ],
  };
  const EX = {
    'ex1-list': [
      { q: 'An email contains only “money” and “dear”. Compute the weighted sum and σ. Is it spam if the rule is σ ≥ 0.5? (σ(−1) = 0.27, σ(0) = 0.50, σ(1) = 0.73, σ(2) = 0.88)', a: '1.5 − 1.0 − 1.5 = −1.0; σ(−1) = 0.27 &lt; 0.5 → <b>not spam</b>.' },
      { q: 'Which one word would you add to make it spam? What is the new σ?', a: 'Add “prize”: the sum becomes 1.0 and σ = 0.73 → spam. (Adding “win” gives a sum of 0 and σ = 0.50: exactly on the edge.)' },
      { q: 'In the horizontal-edge neuron, a pixel has weight 0. What does that mean?', a: 'That pixel is ignored by this neuron: whatever its brightness, it adds 0 to the sum.' },
    ],
    'ex1b-list': [
      { q: 'Count the parameters of a 784 → 16 → 16 → 10 network without looking back. And of 784 → 32 → 10?', a: '12,560 + 272 + 170 = <b>13,002</b>. For 784 → 32 → 10: 25,120 + 330 = <b>25,450</b>.' },
      { q: 'A layer has 3 inputs and 2 neurons. What is the shape of W, and how many numbers are in b?', a: 'W is 2 × 3 (one row per neuron, one column per input); b has 2 numbers.' },
      { q: 'You remove every activation function from a 5-layer network. What kind of model is left?', a: 'A single linear model: straight lines on straight lines are still straight.' },
    ],
    'ex2-list': [
      { q: 'A network sees a “1”. Its outputs are 0.1 for “0”, 0.6 for “1”, 0.3 for “7” and 0 for the others. What is the cost of this example?', a: '0.1² + (0.6 − 1)² + 0.3² = 0.01 + 0.16 + 0.09 = <b>0.26</b>.' },
      { q: 'w = 0.5, gradient = −2, η = 0.1. What is the new w? Did w go up or down, and why?', a: '0.5 − 0.1 · (−2) = <b>0.7</b>: up, because the slope is negative, so the cost falls when w increases.' },
      { q: 'How many inputs does the cost function of the 784 → 16 → 16 → 10 network have? How many outputs?', a: '<b>13,002 inputs</b> (every weight and bias) and <b>1 output</b> (the cost).' },
      { q: 'A ball stops in a valley that is not the lowest. What is this called? Name one thing you could try.', a: 'A <b>local minimum</b>. Try a different random start, a larger learning rate, or momentum (as in Adam).' },
      { q: 'Accuracy before training is about 10%. Why exactly 10%?', a: 'Random weights guess one of 10 digits more or less at random: right 1 time in 10.' },
    ],
  };
  if (Q[page] && document.getElementById('qz')) UL.quiz('qz', Q[page]);
  Object.keys(EX).forEach((id) => { if (document.getElementById(id)) UL.exercises(id, EX[id]); });
  UL.start();
})();
