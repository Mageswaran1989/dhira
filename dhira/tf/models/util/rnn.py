import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper

class SwitchableDropoutWrapper(DropoutWrapper):
    """
    A wrapper of tensorflow.contrib.rnn.DropoutWrapper that does not apply
    dropout if is_train is not True (dropout only in training).
    """
    def __init__(self, cell, is_train, input_keep_prob=1.0,
                 output_keep_prob=1.0, seed=None):
        super(SwitchableDropoutWrapper, self).__init__(
            cell,
            input_keep_prob=input_keep_prob,
            output_keep_prob=output_keep_prob,
            seed=seed)
        self.is_train = is_train

    def __call__(self, inputs, state, scope=None):
        # Get the dropped-out outputs and state
        outputs_do, new_state_do = super(SwitchableDropoutWrapper,
                                         self).__call__(
                                             inputs, state, scope=scope)
        tf.get_variable_scope().reuse_variables()
        # Get the un-dropped-out outputs and state
        outputs, new_state = self._cell(inputs, state, scope)

        # Set the outputs and state to be the dropped out version if we are
        # training, and no dropout if we are not training.
        outputs = tf.cond(self.is_train, lambda: outputs_do,
                          lambda: outputs * (self._output_keep_prob))
        if isinstance(state, tuple):
            new_state = state.__class__(
                *[tf.cond(self.is_train, lambda: new_state_do_i,
                          lambda: new_state_i)
                  for new_state_do_i, new_state_i in
                  zip(new_state_do, new_state)])
        else:
            new_state = tf.cond(self.is_train, lambda: new_state_do,
                                lambda: new_state)
        return outputs, new_state


def mean_pool(input_tensor, sequence_length=None):
    """
    Given an input tensor (e.g., the outputs of a LSTM), do mean pooling
    over the last dimension of the input.

    For example, if the input was the output of a LSTM of shape
    (batch_size, sequence length, hidden_dim), this would
    calculate a mean pooling over the last dimension (taking the padding
    into account, if provided) to output a tensor of shape
    (batch_size, hidden_dim).

    :param input_tensor: Tensor
        An input tensor, preferably the output of a tensorflow RNN.
        The mean-pooled representation of this output will be calculated
        over the last dimension.

    :param sequence_length: Tensor, optional (default=None)
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    :return mean_pooled_output: Tensor
        A tensor of one less dimension than the input, with the size of the
        last dimension equal to the hidden dimension state size.
    """
    with tf.name_scope("mean_pool"):
        # shape (batch_size, sequence_length)
        input_tensor_sum = tf.reduce_sum(input_tensor, axis=-2)

        # If sequence_length is None, divide by the sequence length
        # as indicated by the input tensor.
        if sequence_length is None:
            sequence_length = tf.shape(input_tensor)[-2]

        # Expand sequence length from shape (batch_size,) to
        # (batch_size, 1) for broadcasting to work.
        expanded_sequence_length = tf.cast(tf.expand_dims(sequence_length, -1),
                                           "float32") + 1e-08

        # Now, divide by the length of each sequence.
        # shape (batch_size, sequence_length)
        mean_pooled_input = (input_tensor_sum /
                             expanded_sequence_length)
        return mean_pooled_input


def last_relevant_output(output, sequence_length):
    """
    Given the outputs of a LSTM, get the last relevant output that
    is not padding. We assume that the last 2 dimensions of the input
    represent (sequence_length, hidden_size).

    :param output: Tensor
        A tensor, generally the output of a tensorflow RNN.
        The tensor index sequence_lengths+1 is selected for each
        instance in the output.

    :param sequence_length: Tensor
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    :return last_relevant_output: Tensor
        The last relevant output (last element of the sequence), as retrieved
        by the output Tensor and indicated by the sequence_length Tensor.
    """
    with tf.name_scope("last_relevant_output"):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[-2]
        out_size = int(output.get_shape()[-1])
        index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant
