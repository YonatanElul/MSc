import tensorflow as tf


def improved_wasserstein_loss(discriminator, real_data: tf.Tensor, fake_data: tf.Tensor,
                              discriminator_real: tf.Tensor, discriminator_fake: tf.Tensor,
                              batch_size: int = 64, g_penalty_lambda: int = 10):
    """
    Description:

    :param discriminator:
    :param real_data:
    :param fake_data:
    :param discriminator_real:
    :param discriminator_fake:
    :param batch_size:
    :param g_penalty_lambda:
    :return:
    """

    gen_cost = - tf.reduce_mean(discriminator_fake)
    disc_cost = tf.reduce_mean(discriminator_fake) - tf.reduce_mean(discriminator_real)

    alpha = tf.random_uniform(
        shape=[batch_size, 1],
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha * differences)
    gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    disc_cost += g_penalty_lambda * gradient_penalty

    return gen_cost, disc_cost


def eart_mover_loss(discriminator_real: tf.Tensor, discriminator_fake: tf.Tensor):
    """
    Description:

    :param discriminator_real:
    :param discriminator_fake:
    :return:
    """

    gen_cost = - tf.reduce_mean(discriminator_fake)
    disc_cost = tf.reduce_mean(discriminator_fake) - tf.reduce_mean(discriminator_real)

    return gen_cost, disc_cost




