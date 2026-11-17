/* Hover explanations: keywords in the text and every part of every formula (Neural Network Lab).
   Formulas: known parts are tagged before KaTeX renders them (\htmlData{tip=...}), so each part can be hovered.
   Hovering a part explains that part; hovering anywhere else on a display formula explains the whole formula, part by part. */
(function () {
  'use strict';
  const UL = window.UL;

  /* ------------------------------------------------------------------ formula parts */
  // [TeX to find, part id, optional core: the piece of `find` that gets tagged]. Longest first.
  const PAT = [
    ['\\frac{1}{N}\\sum_{n=1}^{N} C_n', 'avgC'], ['\\frac{\\partial C}{\\partial w}', 'dCdw'], ['\\frac{\\partial z}{\\partial w}', 'dzdw'],
    ['\\frac{\\partial a}{\\partial z}', 'dadz'], ['\\frac{\\partial C}{\\partial a}', 'dCda'], ['\\nabla C(\\mathbf{w})', 'nabla'],
    ['\\mathbf{w} \\leftarrow', 'wvec', '\\mathbf{w}'], ['- \\eta', 'eta', '\\eta'], ['\\eta', 'eta'],
    ['\\sum_{j=0}^{9}', 'sumj'], ['\\sum_{i=1}^{m}', 'sumi'], ['\\sum_{u}\\sum_{v}', 'sumuv'], ['\\prod_{l=2}^{L}', 'prodl'],
    ['C_{\\text{total}}', 'Ctot'], ['C = ', 'C', 'C'],
    ['a^{(1)}', 'a1'], ['W a^{(0)}', 'W', 'W'], ['a^{(0)}', 'a0'],
    ['w_1 a_1', 'wa'], ['w_2 a_2', 'wa'], ['w_n a_n', 'wa'], ['x_i w_i', 'xw'], ['w_0', 'w0'], ['\\dots', 'dots'],
    ['+ b\\big)', 'b', 'b'], ['a = \\sigma', 'aout', 'a'], ['\\hat y', 'yhat'], ['g\\Big(', 'g', 'g'],
    ['a_j', 'aj'], ['y_j', 'yj'], ['e^{-z}', 'ez'], ['\\max(0, z)', 'relu'], ['\\mathrm{ReLU}', 'reluname'],
    ['\\sigma\'(z_l)', 'dsig'], ['\\sigma', 'sigma'], ['w_l', 'wl'], ['g_L', 'gL'], ['g_1', 'g1'],
    ['k_{u,v}', 'kuv'], ['x_{i+u,\\,j+v}', 'xij'], ['y_{i,j}', 'yij'],
    ['W_h\\, h_{t-1}', 'Wh', 'W_h'], ['h_{t-1}', 'hprev'], ['W_x\\, x_t', 'Wx', 'W_x'], ['x_t', 'xt'], ['h_t', 'ht'], ['\\tanh', 'tanh'],
    ['TP', 'TP'], ['FN', 'FN'], ['FP', 'FP'], ['\\lVert g \\rVert', 'norm'], ['c\\,\\frac', 'clip', 'c'],
    ['\\min_G \\max_D', 'minmax'], ['\\mathbb{E}', 'E'], ['D(x)', 'Dx'], ['G(z)', 'Gz'], ['\\leftarrow', 'arrow'],
  ];
  const SYM = {
    aout: ['a', 'The activation of this neuron: the number it holds after the calculation, between 0 and 1 for a sigmoid neuron.'],
    sigma: ['\\sigma', 'The sigmoid function: it squishes any number into the range 0…1. Very negative → close to 0, very positive → close to 1, σ(0) = 0.5.'],
    wa: ['w_i a_i', 'One input times its weight: the activation a_i of a neuron in the previous layer, multiplied by the weight w_i of the connection. A positive weight lets a bright input push the sum up, a negative weight pushes it down.'],
    dots: ['\\dots', '… and so on for every neuron of the previous layer: 784 terms for the first hidden layer of the digit network.'],
    b: ['b', 'The bias: a number added before squishing. It sets how high the weighted sum must be before the neuron lights up. b = −10 means: fire only when the sum is above about 10.'],
    a1: ['a^{(1)}', 'The activations of the next layer, all at once: a column of numbers, one per neuron (16 for the first hidden layer).'],
    a0: ['a^{(0)}', 'The activations of the previous layer as a column of numbers. For the first layer: the 784 pixel values.'],
    W: ['W', 'The weight matrix. Row k holds every weight going into neuron k of the next layer, so W a computes all weighted sums in one matrix multiplication. Size 16 × 784 for the first layer.'],
    xw: ['x_i w_i', 'Input number i times its weight.'],
    w0: ['w_0', 'The bias, written as weight number 0 (some books write b).'],
    sumi: ['\\sum_{i=1}^{m}', 'Add up over all m inputs.'],
    yhat: ['\\hat y', 'The output of the neuron: its prediction.'],
    g: ['g', 'The activation function (sigmoid, ReLU, …): the non-linear squish after the sum.'],
    ez: ['e^{-z}', 'e (about 2.718) to the power −z. Large positive z makes it tiny, so σ → 1; large negative z makes it huge, so σ → 0.'],
    relu: ['\\max(0, z)', 'Keep z if it is positive, otherwise output 0. A straight line with slope 1 for positive inputs.'],
    reluname: ['\\mathrm{ReLU}', 'Rectified Linear Unit: the modern default activation inside deep networks.'],
    C: ['C', 'The cost of this one example: one number saying how bad the network’s answer was. 0 = perfect.'],
    Ctot: ['C_{\\text{total}}', 'The cost of the whole network: the average cost over all training examples. This is the one number that learning makes small.'],
    avgC: ['\\frac{1}{N}\\sum_{n=1}^{N} C_n', 'Add the cost of every one of the N training examples (60,000 for MNIST) and divide by N: the average.'],
    sumj: ['\\sum_{j=0}^{9}', 'Add up over the 10 output neurons, one for each digit 0 … 9.'],
    aj: ['a_j', 'What output neuron j says: its activation between 0 and 1.'],
    yj: ['y_j', 'What we wanted output neuron j to say: 1 for the correct digit, 0 for every other one.'],
    eta: ['\\eta', 'The learning rate (eta): how big each step is. Too small: learning crawls. Too large: the steps overshoot the valley and the cost jumps around.'],
    dCdw: ['\\frac{\\partial C}{\\partial w}', 'The slope of the cost with respect to this one weight: how much the cost changes if w is nudged up a tiny bit. Its sign says which way to move w; its size says how much the cost cares.'],
    dzdw: ['\\frac{\\partial z}{\\partial w}', 'How the weighted sum z changes when the weight changes: simply the input that the weight multiplies (a from the layer before).'],
    dadz: ['\\frac{\\partial a}{\\partial z}', 'How the activation changes when z changes: the slope of the activation function, σ′(z) = σ(z)(1 − σ(z)) for a sigmoid.'],
    dCda: ['\\frac{\\partial C}{\\partial a}', 'How the cost changes when this neuron’s activation changes. For an output neuron it is 2(a − y); for a hidden neuron it is collected from the layer after it: the blame passed back.'],
    nabla: ['\\nabla C(\\mathbf{w})', 'The gradient: the list of all slopes ∂C/∂w, one per weight and bias (13,002 numbers). It points uphill, in the direction of steepest increase of the cost.'],
    wvec: ['\\mathbf{w}', 'All weights and biases of the network, written as one long list (vector).'],
    arrow: ['\\leftarrow', '“Becomes”: replace the value on the left by the expression on the right. An update step, not an equation.'],
    dsig: ['\\sigma\'(z_l)', 'The slope of the sigmoid in layer l. It is never larger than 0.25.'],
    wl: ['w_l', 'A weight of layer l.'],
    prodl: ['\\prod_{l=2}^{L}', 'Multiply together, one factor for every layer between layer 1 and the output layer L.'],
    gL: ['g_L', 'The gradient at the output layer, where the error is measured.'],
    g1: ['g_1', 'The gradient that reaches the first layer after being passed back through every layer.'],
    kuv: ['k_{u,v}', 'One weight of the small filter (kernel), at row u and column v. A 3 × 3 filter has 9 such weights, shared over the whole image.'],
    xij: ['x_{i+u,\\,j+v}', 'The input pixel under that filter weight when the filter sits at position (i, j).'],
    yij: ['y_{i,j}', 'One number of the output (feature map): how strongly the pattern of the filter is present at position (i, j).'],
    sumuv: ['\\sum_{u}\\sum_{v}', 'Add up over every row u and column v of the filter: 9 products for a 3 × 3 filter.'],
    ht: ['h_t', 'The hidden state after reading element t: the network’s memory of everything it has read so far.'],
    hprev: ['h_{t-1}', 'The hidden state from the step before: yesterday’s memory.'],
    xt: ['x_t', 'The input at step t: one word, one sensor reading, one value.'],
    Wh: ['W_h', 'The weights that mix the old memory into the new one. The same matrix is used at every step.'],
    Wx: ['W_x', 'The weights that bring the new input into the memory. Also shared over all steps.'],
    tanh: ['\\tanh', 'An activation like the sigmoid, but squishing into −1 … 1.'],
    TP: ['TP', 'True positives: frauds that the model flagged as fraud.'],
    FN: ['FN', 'False negatives: frauds the model missed (called them legitimate).'],
    FP: ['FP', 'False positives: legitimate cases the model wrongly flagged as fraud.'],
    norm: ['\\lVert g \\rVert', 'The length (size) of the whole gradient vector.'],
    clip: ['c', 'The clipping threshold, e.g. 1 or 5: the largest gradient length allowed.'],
    minmax: ['\\min_G \\max_D', 'The discriminator D chooses its weights to make this expression as large as possible; the generator G chooses its weights to make it as small as possible.'],
    E: ['\\mathbb{E}', 'The average (expected value) over many examples.'],
    Dx: ['D(x)', 'The discriminator’s answer for x: the probability that x is real.'],
    Gz: ['G(z)', 'A fake made by the generator from random noise z.'],
  };
  // Whole display formulas: match = a piece of the source, then a title and a plain reading.
  const FORM = [
    ['a = \\sigma\\big(w_1 a_1', 'One neuron', 'Multiply each input activation by the weight of its connection, add them all up, add the bias, then squish the result into 0…1 with the sigmoid. That number is the neuron’s activation.'],
    ['a^{(1)} = \\sigma', 'One whole layer at once', 'Stack the previous layer’s activations into a column a⁽⁰⁾. Multiplying by the weight matrix W gives every weighted sum of the next layer in one go; add the bias column b and apply σ to every entry.'],
    ['\\sigma(z) = \\frac{1}{1 + e^{-z}}', 'The sigmoid', 'For z = 0 the fraction is 1/2. As z grows, e^{−z} shrinks and σ approaches 1; as z becomes very negative, e^{−z} explodes and σ approaches 0.'],
    ['\\mathrm{ReLU}(z) = \\max(0, z)', 'ReLU', 'Negative inputs become 0, positive inputs pass through unchanged. Simple, fast, and its slope of 1 keeps gradients alive in deep networks.'],
    ['\\hat y = g\\Big(w_0', 'The general neuron', 'Inputs x₁ … x_m, each multiplied by its weight, summed, plus the bias w₀, then passed through an activation function g.'],
    ['C = \\sum_{j=0}^{9}', 'The cost of one example', 'For each of the 10 outputs, take the difference between what the network said and what we wanted, square it (so it is positive and big errors count much more), and add up. Small when the network is confident and right.'],
    ['C_{\\text{total}}', 'The cost of the network', 'The average of the cost over all training examples. Its inputs are the 13,002 weights and biases; its output is one number.'],
    ['w \\leftarrow w - \\eta', 'One gradient descent step', 'Compute the slope of the cost for this weight. If the slope is positive, the cost rises when w rises, so make w smaller; if negative, make w bigger. η sets the step size.'],
    ['\\mathbf{w} \\leftarrow \\mathbf{w}', 'Gradient descent for all weights', 'The same step for all 13,002 weights at once: move every weight a little against its slope. The whole network rolls one step downhill on the cost landscape.'],
    ['\\frac{\\partial C}{\\partial w} = \\frac{\\partial z}{\\partial w}', 'The chain rule', 'A weight affects the cost only through a chain: w changes z, z changes a, a changes C. Multiply the three local slopes to get the total effect. Backpropagation applies this from the output back to every weight.'],
    ['y_{i,j} = \\sum_{u}', 'Convolution', 'Place the small filter on the image at position (i, j), multiply each filter weight by the pixel under it, and add up. Slide to the next position and repeat: the results form the feature map.'],
    ['h_t = \\tanh', 'A recurrent step', 'The new memory mixes the old memory and the new input with two weight matrices, adds a bias and squishes. The same weights are reused at every step, so the sequence can have any length.'],
    ['\\text{recall} = \\frac{TP}', 'Recall and precision', 'Recall: of all real frauds, how many did we catch? Precision: of everything we flagged, how much really was fraud? Accuracy hides both when one class is rare.'],
    ['g_1 = g_L', 'Why gradients vanish or explode', 'On the way back, the gradient is multiplied once per layer by a weight and a slope. With sigmoid slopes ≤ 0.25 the product shrinks layer after layer (vanishing); with factors above 1 it grows without limit (exploding).'],
    ['\\text{if } \\lVert g \\rVert', 'Gradient clipping', 'If the gradient is longer than c, shrink it to length c but keep its direction. Steps can no longer be huge, so one bad batch cannot throw the weights to NaN.'],
    ['\\min_G \\max_D', 'The GAN game', 'The discriminator D tries to give real data a high score and fakes a low one; the generator G tries to make fakes that D scores high. Each trains against the other.'],
  ];
  const OVR = {};

  UL.texTips = function (tex) {
    const slots = []; let s = tex;
    for (const [find, id, core] of PAT) {
      if (!s.includes(find)) continue;
      const c = core || find, i = find.indexOf(c), pre = find.slice(0, i), post = find.slice(i + c.length);
      s = s.split(find).join(pre + '\u0001' + slots.length + '\u0002' + post); slots.push(`\\htmlData{tip=${id}}{${c}}`);
    }
    return s.replace(/\u0001(\d+)\u0002/g, (_, n) => slots[+n]);
  };

  /* ------------------------------------------------------------------ keywords */
  // [term, aliases (regex sources, case-insensitive), explanation]
  const KW = [
    ['Neural network', ['neural networks?', 'neural nets?'], 'A function built from many simple units (neurons) arranged in layers. Each neuron computes a weighted sum of the layer before, adds a bias and applies an activation function. Learning means adjusting the weights and biases from examples.'],
    ['Deep learning', ['deep learning'], 'Machine learning with neural networks that have many layers. “Deep” refers to the number of layers. Each layer builds slightly more complex features from the one below.'],
    ['Neuron', ['neurons?'], 'A unit that holds one number, its activation. An artificial neuron takes a weighted sum of the neurons in the previous layer, adds a bias and squishes the result with an activation function.'],
    ['Activation', ['activations?(?! function)'], 'The number a neuron holds, e.g. between 0 and 1 for a sigmoid neuron. For input neurons it is a pixel’s brightness; for output neurons, how strongly the network thinks the answer is that digit.'],
    ['Weight', ['weights?'], 'A number on each connection between two neurons. Positive weights let a bright input push the neuron up, negative weights push it down. Weights are what the network learns.'],
    ['Bias', ['bias(?:es)?'], 'A number added to the weighted sum before the activation function. It shifts how easily the neuron fires: a large negative bias means the weighted sum must be large before the neuron lights up.'],
    ['Weighted sum', ['weighted sums?'], 'Each input multiplied by its weight, all added up: w₁a₁ + w₂a₂ + … + wₙaₙ. Large when the inputs match the pattern the weights describe.'],
    ['Layer', ['layers?'], 'A group of neurons that all compute from the same previous layer. The input layer holds the data, hidden layers compute features, the output layer gives the answer.'],
    ['Input layer', ['input layers?'], 'The first layer: it simply holds the data. For a 28 × 28 image, 784 neurons, one per pixel.'],
    ['Hidden layer', ['hidden layers?'], 'A layer between input and output. Its neurons are not given meaning by us; the network decides what they detect while it learns.'],
    ['Output layer', ['output layers?'], 'The last layer. For digit recognition, 10 neurons, one per digit; the brightest one is the network’s answer.'],
    ['Activation function', ['activation functions?'], 'The non-linear function applied after the weighted sum, such as sigmoid or ReLU. Without it, any number of stacked layers would collapse into one linear model.'],
    ['Sigmoid', ['sigmoids?', 'logistic function'], 'σ(z) = 1 / (1 + e^{−z}). Squishes any number into 0…1: very negative → 0, very positive → 1, σ(0) = 0.5. Its slope is never more than 0.25, which causes vanishing gradients in deep networks.'],
    ['ReLU', ['ReLU'], 'Rectified Linear Unit: ReLU(z) = max(0, z). Zero for negative inputs, the input itself for positive ones. The modern default inside deep networks because its slope is 1 for positive inputs.'],
    ['Parameters', ['parameters?', 'knobs?'], 'Everything the network learns: all weights and biases. The 784 → 16 → 16 → 10 network has 13,002 of them.'],
    ['Matrix multiplication', ['matrix multipl(?:y|ication|ies)', 'matrix'], 'Multiplying the weight matrix W by the column of activations computes every weighted sum of a layer in one operation. GPUs are built to do this very fast.'],
    ['Forward pass', ['forward pass(?:es)?'], 'Computing the output from the input, layer by layer. During training every activation of the forward pass is stored, because backpropagation needs them.'],
    ['Cost function', ['cost functions?', 'loss functions?', 'cost', 'loss'], 'One number that says how bad the network is. For one example: add up (output − wanted)² over the outputs. For the network: average that over all training examples. Learning means making it small.'],
    ['Gradient', ['gradients?'], 'The list of slopes of the cost, one for every weight and bias. It points in the direction in which the cost increases fastest; its sign tells each weight which way to move, its size how much that weight matters.'],
    ['Gradient descent', ['gradient descent'], 'Repeatedly take a small step against the gradient: w ← w − η · gradient. Like a ball rolling downhill on the cost landscape, feeling only the slope under its feet.'],
    ['Learning rate', ['learning rates?', 'step size'], 'η (eta): how big each gradient descent step is. Too small and training crawls; too large and the steps overshoot, so the cost jumps around or even grows.'],
    ['Local minimum', ['local minim(?:um|a)'], 'A valley of the cost landscape that is lower than its surroundings but not the lowest point overall. Gradient descent stops there because the slope is zero.'],
    ['Backpropagation', ['backpropagation', 'backprop', 'backward pass'], 'The algorithm that computes the gradient efficiently. Starting from the cost, it moves backwards through the layers and multiplies local slopes (the chain rule), so every weight learns how it contributed to the error.'],
    ['Chain rule', ['chain rule'], 'A rule from calculus: if w affects z, z affects a and a affects C, then the effect of w on C is the product of the three local effects.'],
    ['Mini-batch', ['mini-batch(?:es)?', 'batch(?:es)?'], 'A small random group of training examples (e.g. 32). The gradient is averaged over a mini-batch and one step is taken, instead of using all 60,000 examples for every step: much faster and still heading downhill on average.'],
    ['Stochastic gradient descent', ['stochastic gradient descent', 'SGD'], 'Gradient descent on random mini-batches. Each step is a little noisy, but there are many more steps per second.'],
    ['Epoch', ['epochs?', 'passes over the training set'], 'One pass through the whole training set. Training the digit network for 30 epochs means every training image was used 30 times.'],
    ['Training set', ['training sets?', 'training data'], 'The labelled examples the network learns from. Its weights are tuned to make the cost small on these.'],
    ['Test set', ['test sets?', 'test data', 'unseen data'], 'Examples kept aside and never used for training. Accuracy on the test set shows how well the network handles new data.'],
    ['Accuracy', ['accuracy'], 'The fraction of examples the model gets right. Misleading when one class is rare: always saying “legitimate” is 98% accurate if 2% are fraud.'],
    ['MNIST', ['MNIST'], 'A classic dataset of 70,000 handwritten digits (60,000 for training, 10,000 for testing), each a 28 × 28 greyscale picture.'],
    ['Overfitting', ['overfit(?:ting|s)?', 'memoris(?:e|ing)'], 'Learning the training data too well, including its noise, so that performance on new data is worse. A network can even memorise random labels.'],
    ['Decision boundary', ['decision boundar(?:y|ies)'], 'The line (or curve) where the network switches from one answer to the other. A single neuron can only draw a straight line; hidden layers can bend it.'],
    ['Multilayer perceptron', ['multilayer perceptrons?', 'MLPs?', 'dense layers?', 'fully connected'], 'The basic neural network: layers where every neuron is connected to every neuron of the layer before, like our digit network.'],
    ['Perceptron', ['perceptrons?'], 'The first artificial neuron (Rosenblatt, 1958): a weighted sum and a hard threshold.'],
    ['CNN', ['CNNs?', 'convolutional(?: neural)? networks?'], 'Convolutional neural network: small filters slide over the image and look for local patterns with the same few weights everywhere. Layers find edges, then parts, then objects. The standard network for images.'],
    ['Convolution', ['convolutions?'], 'Sliding a small filter over the input; at each position multiply the filter weights by the pixels under them and add up. The result is a feature map.'],
    ['Filter', ['filters?', 'kernels?'], 'A small grid of weights (e.g. 3 × 3) that a CNN slides over the image. It responds strongly where the image contains its pattern, such as a vertical edge. The values are learned.'],
    ['Feature map', ['feature maps?'], 'The output of one filter over the whole image: bright where the pattern was found.'],
    ['Pooling', ['pooling', 'max-pool(?:ing)?'], 'Shrinking a feature map by keeping, e.g., the largest value of every 2 × 2 block. Makes the network smaller and less sensitive to small shifts.'],
    ['RNN', ['RNNs?', 'recurrent(?: neural)? networks?'], 'Recurrent neural network: reads a sequence one element at a time and keeps a hidden state that carries memory from step to step. Used for text, speech and time series.'],
    ['Hidden state', ['hidden states?'], 'The memory of an RNN: a vector updated at every step from the previous hidden state and the new input.'],
    ['LSTM', ['LSTMs?', 'GRUs?'], 'Long Short-Term Memory (and the simpler GRU): an RNN with gates that decide what to keep, what to forget and what to output. It remembers over much longer sequences.'],
    ['Transformer', ['transformers?'], 'A network built on attention: every element of a sequence looks at every other element directly. The architecture behind modern language models such as GPT and Claude.'],
    ['Attention', ['attention'], 'A mechanism that lets each element compute how relevant every other element is and mix in their information accordingly.'],
    ['GAN', ['GANs?', 'generative adversarial networks?'], 'Generative adversarial network: a generator makes fakes from random noise, a discriminator tries to tell fakes from real data. Trained against each other, the generator learns to produce realistic data.'],
    ['Generator', ['generators?', 'forger'], 'The part of a GAN that turns random noise into new data, such as a fake digit.'],
    ['Discriminator', ['discriminators?', 'detective'], 'The part of a GAN that classifies inputs as real or fake. Its feedback is how the generator learns.'],
    ['Autoencoder', ['autoencoders?'], 'A network that compresses its input into a small code and rebuilds it. Inputs that rebuild badly are unusual: useful for anomaly detection.'],
    ['Diffusion model', ['diffusion(?: models?)?'], 'A generative model that learns to remove noise step by step; used in text-to-image systems.'],
    ['Deepfake', ['deepfakes?'], 'Realistic fake images, video or audio of real people made with generative models.'],
    ['Data imbalance', ['data imbalance', 'class imbalance', 'imbalanced?'], 'When one class is far rarer than another (2% fraud, 98% legitimate). A model can reach high accuracy by ignoring the rare class completely.'],
    ['Recall', ['recall'], 'Of all real positives (e.g. frauds), the fraction the model caught: TP / (TP + FN).'],
    ['Precision', ['precision'], 'Of everything the model flagged as positive, the fraction that really was: TP / (TP + FP).'],
    ['F1 score', ['F1(?: score)?'], 'The harmonic mean of precision and recall: high only when both are high.'],
    ['Class weights', ['class weights?', 'reweight(?:ing)?'], 'Make mistakes on the rare class cost more in the cost function, e.g. a missed fraud counts 49 times as much as a false alarm.'],
    ['SMOTE', ['SMOTE'], 'Synthetic Minority Over-sampling Technique (Chawla et al., 2002): create new minority examples on the line between a minority point and one of its nearest minority neighbours. Apply to the training set only.'],
    ['Undersampling', ['undersampling', 'downsampl(?:e|ing)'], 'Throw away part of the majority class so the classes are more balanced.'],
    ['Oversampling', ['oversampling', 'resampl(?:e|ing)'], 'Repeat or synthesise examples of the minority class so the classes are more balanced.'],
    ['Threshold', ['thresholds?'], 'The cut-off on the model’s score above which it says “positive”. Lowering it catches more positives (higher recall) at the cost of more false alarms.'],
    ['Vanishing gradient', ['vanishing gradients?', 'vanish(?:es|ing)?'], 'In deep networks the gradient is multiplied by a small factor (e.g. a sigmoid slope ≤ 0.25) once per layer on the way back, so the first layers receive almost no signal and stop learning.'],
    ['Exploding gradient', ['exploding gradients?', 'explod(?:e|es|ing)'], 'The opposite of vanishing: factors larger than 1 multiplied over many layers or time steps make the gradient huge, the steps jump wildly and the loss becomes NaN.'],
    ['Gradient clipping', ['gradient clipping', 'clipping', 'clip'], 'If the gradient is longer than a threshold, scale it down to that length before the step. The standard fix for exploding gradients in RNNs.'],
    ['Batch normalisation', ['batch normali[sz]ation', 'batch norm'], 'Re-centre and re-scale the activations of each layer during training so that signals stay a healthy size.'],
    ['Residual connection', ['residual connections?', 'skip connections?', 'shortcuts?'], 'A shortcut that adds a layer’s input to its output. The gradient can flow back through the shortcut without shrinking, so very deep networks (ResNets) can be trained.'],
    ['He / Xavier initialisation', ['He initiali[sz]ation', 'Xavier', 'initiali[sz]ation'], 'Start the random weights at a size that keeps signals and gradients about the same size from layer to layer (scaled by 1/√inputs, or √(2/inputs) for ReLU).'],
    ['Momentum', ['momentum'], 'Keep part of the previous step in the new one, like a heavy ball. It speeds up along consistent directions and can roll through small bumps and shallow local minima.'],
    ['Adam', ['Adam'], 'A popular optimiser: gradient descent with momentum and a separate, automatically adapted step size for every weight.'],
    ['Softmax', ['softmax'], 'Turns the 10 output numbers into probabilities that add up to 1: exponentiate each and divide by the sum.'],
    ['Linear model', ['linear models?', 'linear'], 'A model whose output is a weighted sum of the inputs: it can only draw straight lines (planes). Stacked layers without activation functions are still one linear model.'],
    ['Synapse', ['synapses?'], 'The connection between two real neurons. Its strength changes with learning; artificial weights copy this idea.'],
    ['Out-of-distribution', ['out-of-distribution', 'outside the training data'], 'Inputs unlike anything in the training data. The network still gives an answer, often a confident one, because it has no way to say “I do not know”.'],
    ['Biomimicry', ['biomimicry', 'copy(?:ing)? nature'], 'Engineering by copying clever designs from nature: wings, Velcro, sonar, and the brain itself.'],
    ['PyTorch', ['PyTorch'], 'A popular Python library for deep learning. loss.backward() runs backpropagation for you.'],
  ];
  const KWRE = (() => {
    const parts = []; KW.forEach((k, i) => k[1].forEach((a) => parts.push({ re: a, i })));
    parts.sort((a, b) => b.re.length - a.re.length);
    return { re: new RegExp('(?<![\\w\\-])(' + parts.map((p) => p.re).join('|') + ')(?![\\w\\-])', 'gi'), parts: parts.map((p) => ({ i: p.i, re: new RegExp('^(?:' + p.re + ')$', 'i') })) };
  })();
  const which = (m) => { for (const p of KWRE.parts) if (p.re.test(m)) return p.i; return -1; };

  function markKeywords() {
    const main = document.querySelector('main'); if (!main) return;
    const SKIP = '.nb-notes,.code-lines,.nb-code,h1,h2,h3,h4,button,label,a,code,pre,select,option,.katex,.quiz,.stats,.calc,.ctl,.seg,.status,.kw,.grid-legend,.legend,.toc,nav,svg,canvas,.plot,.lab-head,table.qt,.stream,.eyebrow,.pill';
    const blocks = main.querySelectorAll('p, li, td, .hint, .takeaway, figcaption, .note, .callout, blockquote, dd, .card p');
    const seen = new Map(); // section id -> Set of keyword indices already marked
    blocks.forEach((b) => {
      if (b.closest(SKIP) || b.dataset.kw) return; b.dataset.kw = '1';
      const sec = b.closest('section'); const sid = sec ? sec.id || 'x' : 'x'; if (!seen.has(sid)) seen.set(sid, new Set());
      const done = seen.get(sid);
      const walker = document.createTreeWalker(b, NodeFilter.SHOW_TEXT, { acceptNode: (n) => (n.parentElement.closest(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT) });
      const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
      nodes.forEach((n) => {
        const t = n.nodeValue; KWRE.re.lastIndex = 0; let m, last = 0, frag = null;
        const strong = !!n.parentElement.closest('strong,b,em,dfn');
        while ((m = KWRE.re.exec(t))) {
          const i = which(m[1]); if (i < 0) continue;
          if (done.has(i) && !strong) continue; done.add(i);
          frag = frag || document.createDocumentFragment();
          frag.appendChild(document.createTextNode(t.slice(last, m.index)));
          const sp = document.createElement('span'); sp.className = 'kw'; sp.dataset.k = i; sp.tabIndex = 0; sp.textContent = m[1]; frag.appendChild(sp); last = m.index + m[1].length;
        }
        if (frag) { frag.appendChild(document.createTextNode(t.slice(last))); n.parentNode.replaceChild(frag, n); }
      });
    });
  }

  /* ------------------------------------------------------------------ the box */
  let box, cur = null, hideT = null;
  const tex = (s) => { try { return window.katex ? katex.renderToString(s, { throwOnError: false }) : s; } catch (e) { return s; } };
  function show(target, html, wide) {
    clearTimeout(hideT); if (!box) { box = document.createElement('div'); box.className = 'tipbox'; box.setAttribute('role', 'tooltip'); document.body.appendChild(box); box.addEventListener('mouseenter', () => clearTimeout(hideT)); box.addEventListener('mouseleave', hide); }
    if (cur === target && box.classList.contains('on')) return;
    if (cur) cur.classList.remove('tip-on'); cur = target; target.classList.add('tip-on');
    box.innerHTML = html; box.classList.toggle('wide', !!wide); box.classList.add('on');
    const r = target.getBoundingClientRect(), bw = box.offsetWidth, bh = box.offsetHeight, vw = document.documentElement.clientWidth;
    let x = Math.min(Math.max(8, r.left + r.width / 2 - bw / 2), vw - bw - 8), y = r.bottom + 10;
    if (y + bh > window.innerHeight - 8) y = r.top - bh - 10 > 8 ? r.top - bh - 10 : Math.max(8, window.innerHeight - bh - 8);
    box.style.left = x + 'px'; box.style.top = y + 'px';
  }
  function hide() { hideT = setTimeout(() => { if (box) box.classList.remove('on'); if (cur) cur.classList.remove('tip-on'); cur = null; }, 120); }
  function formulaOf(el) {
    const ann = el.querySelector('annotation'); const src = ann ? ann.textContent : '';
    for (const f of FORM) if (src.includes(UL.texTips(f[0])) || src.includes(f[0])) return f;
    return null;
  }
  function overrides(el) { const ann = el && el.querySelector('annotation'); const src = ann ? ann.textContent : ''; for (const k in OVR) if (src.includes(UL.texTips(k)) || src.includes(k)) return OVR[k]; return {}; }
  function partHTML(id, ov) { const s = SYM[id]; if (!s) return ''; return `<div class="tip-h">${tex(s[0])}</div><p>${(ov && ov[id]) || s[1]}</p>`; }
  function formulaHTML(el) {
    const f = formulaOf(el), ov = overrides(el), ids = [];
    el.querySelectorAll('[data-tip]').forEach((p) => { if (!ids.includes(p.dataset.tip)) ids.push(p.dataset.tip); });
    const ann = el.querySelector('annotation'), src = ann ? ann.textContent : '', at = (id) => { const i = src.indexOf('tip=' + id + '}'); return i < 0 ? 1e9 : i; };
    ids.sort((a, b) => at(a) - at(b));
    const rows = ids.filter((id) => SYM[id]).map((id) => `<li><span class="tip-sym">${tex(SYM[id][0])}</span><span>${ov[id] || SYM[id][1]}</span></li>`).join('');
    return `<div class="tip-k">Formula</div>${f ? `<div class="tip-h">${f[1]}</div><p>${f[2]}</p>` : ''}${rows ? `<div class="tip-k">Part by part</div><ul class="tip-parts">${rows}</ul>` : ''}<div class="tip-foot">Hover a single symbol to see only that part.</div>`;
  }
  function onOver(e) {
    const part = e.target.closest && e.target.closest('[data-tip]');
    if (part && SYM[part.dataset.tip]) { const disp = part.closest('.katex-display, .katex'); show(part, `<div class="tip-k">Part of the formula</div>${partHTML(part.dataset.tip, overrides(disp))}`); return; }
    const disp = e.target.closest && e.target.closest('.katex-display');
    if (disp && disp.querySelector('[data-tip]')) { show(disp, formulaHTML(disp), true); return; }
    const kw = e.target.closest && e.target.closest('.kw');
    if (kw) { const k = KW[+kw.dataset.k]; show(kw, `<div class="tip-k">Key term</div><div class="tip-h">${k[0]}</div><p>${k[2]}</p>`); }
  }
  function onOut(e) {
    const t = e.target.closest && e.target.closest('[data-tip], .katex-display, .kw'); if (!t) return;
    const to = e.relatedTarget; if (to && (t.contains(to) || (box && box.contains(to)))) return;
    if (to && to.closest && to.closest('[data-tip], .katex-display, .kw')) return;
    hide();
  }
  let wired = false;
  UL.tips = function () {
    markKeywords();
    if (wired) return; wired = true;
    document.addEventListener('mouseover', onOver); document.addEventListener('mouseout', onOut);
    document.addEventListener('focusin', onOver); document.addEventListener('focusout', () => hide());
    document.addEventListener('click', (e) => { // touch: tap to open, tap elsewhere to close
      if (e.target.closest && (e.target.closest('.kw, [data-tip], .katex-display'))) onOver(e); else if (!(box && box.contains(e.target))) hide();
    });
    window.addEventListener('scroll', () => { if (box && box.classList.contains('on')) { box.classList.remove('on'); if (cur) cur.classList.remove('tip-on'); cur = null; } }, { passive: true });
  };
})();
