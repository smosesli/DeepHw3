r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = 'ACT I.'
    temperature = .57
    # DONE: Tweak the parameters to generate a literary masterpiece.
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
We want to use the GPU which has a limited memory. so we cannot load the entire corpus.
"""

part1_q2 = r"""
**Your answer:**
We save the hidden state between sequences. During the training we also save the hidden state between batches. 
"""

part1_q3 = r"""
**Your answer:**
We want the model to learn how to generate text in any size, not limited to seq_len. We want the generated text to
be learned according to the entire corpus with its logical order, shuffling will ruin this order.
"""

part1_q4 = r"""
**Your answer:**
1. The temperature affects the variance of the predicted chars distribution. The higher the temperature the more 
uniform the distribution gets. When sampling, we would prefer to control the distributions and make them less uniform to
increase the chance of sampling the char(s) with the highest scores compared to the others.
During training we would like the distribution to be more uniform to allow for wide range of possibilities.

2. The text looks like it was drawn from a uniform distribution. It basically looks like some one tried giving a monkey
a keyboard.

3. The text has less mistakes and looks very similar to the text in the train set. It is basically looks like someone 
referenced from the original text.     

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


