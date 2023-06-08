import numpy as np

def rotate_training_examples(training_examples):
    inputs, probs, rewards = zip(*training_examples)
    rotated_inputs = []
    for i in inputs:
        for k in range(4):
            rotated_inputs.append(np.rot90(i, k=k, axes=(1, 2)))
    rotated_probs = []
    for p in probs:
        # left -> down
        # down -> right
        # right -> up
        # up -> left
        for k in range(4):
            rotated_probs.append(np.roll(p, k))
    rotated_rewards = []
    for r in rewards:
        rotated_rewards.extend([r] * 4)
    
    return zip(rotated_inputs, rotated_probs, rotated_rewards)