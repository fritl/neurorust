# Roadmap

## Phase 0: Setup

## Phase 1: Linear Algebra
- [X] Implement matrix type
- [X] Implement matrix operations (add, subtract, multiply, transpose, element-wise multiply)
- [X] Unit test each operation against hand-calculated examples

## Phase 2: Activation Functions
- [X] Implement Sigmoid + derivative
- [X] Implement ReLU + derivative
- [X] Implement Tanh + derivative
- [X] Unit test each against hand-calculated values

## Phase 3: Forward Pass
- [X] Implement a single layer's forward computation
- [X] Chain multiple layers into a full forward pass
- [X] Cache intermediate values needed later for backprop
- [X] Test forward pass output against a hand-calculated tiny network

## Phase 4: Weight Initialization
- [X] Implement random weight/bias initialization
- [X] Implement at least one sensible init scheme (e.g. Xavier/He)
- [X] Verify initialized values fall in expected ranges

## Phase 5: Loss Functions
- [X] Implement MSE + derivative
- [X] Implement Softmax + Cross-Entropy + derivative
- [X] Unit test each against hand-calculated values

## Phase 6: Backpropagation
- [ ] Work out gradients by hand for a small 2-3 layer network on paper
- [X] Implement output layer gradient calculation
- [X] Implement gradient propagation backward through hidden layers (chain rule)
- [X] Accumulate weight and bias gradients
- [ ] Verify against the hand-calculated example (and/or numerical gradient checking)

## Phase 7: Optimizer
- [X] Implement plain SGD weight update
- [X] Implement mini-batch gradient accumulation
- [ ] (Optional) Implement momentum
- [ ] (Optional) Implement Adam

## Phase 8: Training Loop
- [X] Implement epoch/batch loop
- [X] Track and print loss over time
- [X] Train and validate on XOR (sanity check: not linearly separable)
- [X] Write a parser for a real dataset (e.g. MNIST CSV/IDX) by hand
- [X] Train and validate on that dataset

## Phase 9: Improvements (optional)
- [ ] Parallelize with std::thread
- [ ] Explore SIMD (std::simd, nightly) for matrix multiplication
- [ ] Add save/load for trained weights

## Ongoing
- [ ] Test every phase in isolation with hand-verified values before moving on — backprop bugs are hard to catch once the network "sort of" learns
