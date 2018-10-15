from DL.Losses.WassersteinLoss import improved_wasserstein_loss
from DL.Utilities.MathOps import auto_corr2d

import tensorflow as tf
import numpy as np


def corr_loss(true_input: np.ndarray, model_input: np.ndarray):
    """

    :param true_input:
    :param model_input:
    :return:
    """

    true_ac = auto_corr2d(in_arr=true_input, kernel=(3, 3), stride=(2, 2), mode='valid')
    model_ac = auto_corr2d(in_arr=model_input, kernel=(3, 3), stride=(2, 2), mode='valid')
    ac_loss = np.mean(np.abs(true_ac - model_ac))

    return ac_loss


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
    ac_loss = tf.py_func(corr_loss, [real_data, fake_data])
    gen_loss = w_generator_loss + gamma * ac_loss

    return gen_loss, w_discriminator_loss
