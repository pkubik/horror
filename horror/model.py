import tensorflow as tf

from horror.data.input import EMBEDDING_SIZE
from horror.data.utils import CLASSES
from horror.utils import DictWrapper


class Features(DictWrapper):
    def __init__(self):
        self.id = None
        self.text = None
        self.text_length = None


class Labels(DictWrapper):
    def __init__(self):
        self.author = None


class Params(DictWrapper):
    def __init__(self):
        self.num_epochs = 80
        self.batch_size = 64
        self.max_word_idx = None
        self.num_rnn_units = 100
        self.keys_units = 50
        self.num_token_encoder_units = 50
        self.learning_rate = 0.001
        self.dropout_rate = 0.1


NUM_CLASSES = sum(1 for _ in CLASSES)


DEFAULT_PARAMS = Params().as_dict()


def model_fn(mode, features, labels, params):
    _params = Params.from_dict(params)
    _features = Features.from_dict(features)

    _labels = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        _labels = Labels.from_dict(labels)

    return build_model(mode, _features, _labels, _params)


def build_model(mode: tf.estimator.ModeKeys,
                features: Features,
                labels: Labels,
                params: Params) -> tf.estimator.EstimatorSpec:

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    global_step = tf.contrib.framework.get_global_step()

    with tf.device("/cpu:0"):
        embeddings = tf.placeholder(tf.float32, [None, EMBEDDING_SIZE], name='embeddings')

    embedded_text = tf.nn.embedding_lookup(embeddings, tf.nn.relu(features.text))

    with tf.variable_scope("encoder"):
        with tf.variable_scope("token"):
            token_encoder = DenseLayer(embedded_text, params.num_token_encoder_units, is_training, params.dropout_rate)

        with tf.variable_scope("full"):
            full_encoder = RNNLayer(
                token_encoder.outputs,
                features.text_length,
                params.num_rnn_units,
                params.dropout_rate)

    with tf.variable_scope("output"):
        keys_layer = tf.keras.layers.Dense(params.keys_units, activation=lrelu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        keys = keys_layer(full_encoder.outputs)
        tf.summary.histogram('keys_kernel', keys_layer.kernel)

        all_outputs = tf.concat([keys, token_encoder.outputs], -1)
        counts = tf.reduce_sum(all_outputs, -2)
        logits = tf.layers.dense(tf.concat((counts, full_encoder.final_state), -1), NUM_CLASSES,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

        prediction = tf.argmax(logits, -1)
        scores = tf.nn.sigmoid(logits)

    # Assign a default value to the train_op and loss to be passed for modes other than TRAIN
    loss = None
    train_op = None
    eval_metric_ops = None
    # Following part of the network will be constructed only for training
    if mode != tf.estimator.ModeKeys.PREDICT:
        hot_author = 0.995 * tf.one_hot(labels.author, NUM_CLASSES) + 0.005

        author_loss = tf.losses.sigmoid_cross_entropy(
            hot_author,
            logits)

        loss = tf.losses.get_total_loss()
        tf.summary.scalar('author_loss', author_loss)

        learning_rate = tf.train.exponential_decay(params.learning_rate, global_step,
                                                   12000, 0.5, staircase=False)
        tf.summary.scalar('learning_rate', learning_rate)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer="Adam")

        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(
                labels.author,
                prediction,
                name='accuracy')

            eval_metric_ops = {
                'accuracy': accuracy
            }

    prediction = {
        'id': features.id,
        'text': features.text,
        'text_length': features.text_length,
        'author': prediction,
        'scores': scores
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=prediction,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def lrelu(x):
    return tf.maximum(x, 0.005 * x)


class DenseLayer:
    def __init__(self, inputs: tf.Tensor, num_units: int, training=False, dropout_rate=0.0):
        self.dense = tf.keras.layers.Dense(num_units, activation=lrelu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.outputs = self.dense(inputs)

        if dropout_rate > 0.0:
            self.outputs = tf.layers.dropout(self.outputs, dropout_rate, training=training)


class RNNLayer:
    def __init__(self,
                 inputs: tf.Tensor,
                 inputs_lengths: tf.Tensor,
                 num_hidden: int,
                 dropout_rate=0.0,
                 initial_states: tuple = None):

        fw_cell = tf.nn.rnn_cell.GRUCell(num_hidden, activation=lrelu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())
        bw_cell = tf.nn.rnn_cell.GRUCell(num_hidden, activation=lrelu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer())

        if initial_states is not None:
            fw_initial_state, bw_initial_state = initial_states
        else:
            fw_initial_state, bw_initial_state = None, None

        if dropout_rate > 0.0:
            dropout_keep_prob = 1 - dropout_rate
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                fw_cell,
                input_keep_prob=dropout_keep_prob,
                output_keep_prob=dropout_keep_prob,
                variational_recurrent=True,
                input_size=inputs.shape[-1],
                dtype=tf.float32)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                bw_cell,
                input_keep_prob=dropout_keep_prob,
                output_keep_prob=dropout_keep_prob,
                variational_recurrent=True,
                input_size=inputs.shape[-1],
                dtype=tf.float32)

        self.outputs_tuple, self.final_states_tuple = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, inputs, inputs_lengths,
            initial_state_fw=fw_initial_state,
            initial_state_bw=bw_initial_state,
            dtype=tf.float32)

    @property
    def outputs(self):
        return tf.reduce_sum(self.outputs_tuple, axis=0)

    @property
    def final_state(self):
        return tf.reduce_sum(self.final_states_tuple, axis=0)


def softsign_glu(values: tf.Tensor, gate_values: tf.Tensor):
    with tf.name_scope("softsign_glu"):
        return tf.nn.softsign(gate_values) * values
