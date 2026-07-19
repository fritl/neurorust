# Roadmap

## Phase 0: Setup

## Phase 1: Linear Algebra
- [X] Implement matrix type
- [X] Implement matrix operations (add, subtract, multiply, transpose, element-wise multiply)
- [X] Unit test each operation against hand-calculated examples

## Phase 2: Activation Functions
- [ ] Implement Sigmoid + derivative
- [ ] Implement ReLU + derivative
- [ ] Implement Tanh + derivative
- [ ] Unit test each against hand-calculated values

## Phase 3: Forward Pass
- [ ] Implement a single layer's forward computation
- [ ] Chain multiple layers into a full forward pass
- [ ] Cache intermediate values needed later for backprop
- [ ] Test forward pass output against a hand-calculated tiny network

## Phase 4: Weight Initialization
- [ ] Implement random weight/bias initialization
- [ ] Implement at least one sensible init scheme (e.g. Xavier/He)
- [ ] Verify initialized values fall in expected ranges

## Phase 5: Loss Functions
- [ ] Implement MSE + derivative
- [ ] Implement Softmax + Cross-Entropy + derivative
- [ ] Unit test each against hand-calculated values

## Phase 6: Backpropagation
- [ ] Work out gradients by hand for a small 2-3 layer network on paper
- [ ] Implement output layer gradient calculation
- [ ] Implement gradient propagation backward through hidden layers (chain rule)
- [ ] Accumulate weight and bias gradients
- [ ] Verify against the hand-calculated example (and/or numerical gradient checking)

## Phase 7: Optimizer
- [ ] Implement plain SGD weight update
- [ ] Implement mini-batch gradient accumulation
- [ ] (Optional) Implement momentum
- [ ] (Optional) Implement Adam

## Phase 8: Training Loop
- [ ] Implement epoch/batch loop
- [ ] Track and print loss over time
- [ ] Train and validate on XOR (sanity check: not linearly separable)
- [ ] Write a parser for a real dataset (e.g. MNIST CSV/IDX) by hand
- [ ] Train and validate on that dataset

## Phase 9: Improvements (optional)
- [ ] Parallelize with std::thread
- [ ] Explore SIMD (std::simd, nightly) for matrix multiplication
- [ ] Add save/load for trained weights

## Ongoing
- [ ] Test every phase in isolation with hand-verified values before moving on — backprop bugs are hard to catch once the network "sort of" learns
