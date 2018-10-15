from DL.Losses.WassersteinLoss import improved_wasserstein_loss

import tensorflow as tf
import numpy as np


def corr_loss(true_input, model_input):
    true_ac = np.random.


def mode_loss(discriminator, real_data: tf.Tensor, fake_data: tf.Tensor,
              discriminator_real: tf.Tensor, discriminator_fake: tf.Tensor,
              batch_size: int = 64, g_penalty_lambda: int = 10, gamma: float = 0.5):
    """

    :param discriminator:
    :param real_data:
    :param fake_data:
    :param discriminator_real:
    :param discriminator_fake:
    :param batch_size:
    :param g_penalty_lambda:
    :param gamma:
    :return:
    """

    # Compute the improved EM loss
    w_generator_loss, w_discriminator_loss = improved_wasserstein_loss(discriminator=discriminator, real_data=real_data,
                                                                       fake_data=fake_data,
                                                                       discriminator_real=discriminator_real,
                                                                       discriminator_fake=discriminator_fake,
                                                                       batch_size=batch_size,
                                                                       g_penalty_lambda=g_penalty_lambda)

    # Compute the the Correlative loss
