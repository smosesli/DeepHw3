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
        batch_size=16,
        h_dim=512, z_dim=32, x_sigma2=2,
        learn_rate=1e-4, betas=(0.9, 0.999),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
This hyper-parameter controls the variation of the generated samples. Low sigma will result in samples that are very
similar to the input data, this is a trade off between how rich the new data that the network produces and between how
similar the data is to the original. High sigma will result in much more erroneous data.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=32, z_dim=100,
        data_label=1, label_noise=0.3,
        discriminator_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            lr=0.002,
            betas=(0.5, 0.999),
        ),
        generator_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            lr=0.0004,
            betas=(0.5, 0.999),
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    # ========================
    return hypers


part3_q1 = r"""
When training the discriminator we create samples from the generator. 
However we don't want the gradients to back-propagate to the generator, thus we do two things:
1) We create the generated images with grad 
2) We detach the samples from the generators graph before feeding it to the discriminator.

When we train the generator we don't want to detach the generated samples because we want to train the generator using
 the discriminator.

"""

part3_q2 = r"""
**Your answer:**
1)No, We also depent on the performance of the discriminator. for example: if the discriminator always produces one we
 will get zero loss .
2)It means that the discriminator learned faster than the generator

"""

part3_q3 = r"""
**Your answer:**
We can see that the VAE learned faster and gave results that are all very similar to each other. But also it was more
human like
The GAN was much harder to train but each result looked very different from each other. but the results also looks less
human then the VAE

"""

# ==============

