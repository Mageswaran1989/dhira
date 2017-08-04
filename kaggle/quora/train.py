import tensorflow as tf
import os
import time
import datetime

from tqdm import tqdm

from input_processor import InputProcessor, load_embeddings, batch_iter, train_batch_iter, val_batch_iter
from siamese_network import SiameseLSTM
from random import random

import global_config
import text_convnet

FLAGS = tf.app.flags.FLAGS

def prepare_data(embedding_path, num_words=200000, max_len=50):
    inp_processor = InputProcessor("train.csv", "test.csv",
                                   num_words=num_words, max_len=max_len)
    inp_processor.pickle_data_sets()
    inp_processor.pickle_embeddings(embedding_path, "embedding_matrix.p")


def step(x1_batch, x2_batch, x1_len, x2_len, y_batch,
         siamese_model, sess, global_step, tr_op_set,
         dropout_keep_prob, embedding_matrix,
         summary_writer, summary_op, evaluate=False):
    """
    A single training step
    """
    dkpb = dropout_keep_prob
    if evaluate:
        dkpb = 1.0

    if random() > 0.5:
        feed_dict = {
            siamese_model.input_x1: x1_batch,
            siamese_model.input_x2: x2_batch,
            siamese_model.input_x1_length: x1_len,
            siamese_model.input_x2_length: x2_len,
            siamese_model.input_y: y_batch,
            siamese_model.embedding_placeholder: embedding_matrix,
            siamese_model.dropout_keep_prob: dkpb,
        }
    else:
        feed_dict = {
            siamese_model.input_x1: x2_batch,
            siamese_model.input_x2: x1_batch,
            siamese_model.input_x1_length: x1_len,
            siamese_model.input_x2_length: x2_len,
            siamese_model.input_y: y_batch,
            siamese_model.embedding_placeholder: embedding_matrix,
            siamese_model.dropout_keep_prob: dkpb,
        }

    if evaluate:
        g_step, loss, pred, summaries = sess.run(
            [global_step, siamese_model.loss, siamese_model.pred, summary_op], feed_dict)
    else:
        _, g_step, loss, pred, out1, summaries = sess.run(
            [tr_op_set, global_step, siamese_model.loss,
             siamese_model.pred, siamese_model.out1, summary_op], feed_dict)
        # print("Out1 shape is : {}".format(len(out1)))
    if g_step % 1000 == 0:
        print('Train loss at step {} is {}'.format(g_step, loss))
    summary_writer.add_summary(summaries, g_step)

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # g_step, siamese_left, siamese_right, summaries = sess.run(
    #         [global_step, siamese_model.out1, siamese_model.out2, summary_op], feed_dict)
    #
    # summary_writer.add_summary(summaries, g_step)
    #
    #
    # conv_model = text_convnet.TextConvNet(FLAGS, True)
    # batch_size = len(y_batch)
    # # conv_model._labels
    # conv_model.build_graph_for_pretrained_layer(batch_size=batch_size,
    #                                             prev_layer=merged_channels,
    #                                             keep_prob=dropout_keep_prob)
    return


def train_network(config = global_config.FLAGS):

    embedding_matrix = load_embeddings("embedding_matrix.p")

    txt_suffix = ("""layers_%(layers)s-dense_units_%(dense_units)s-hidden_%(hidden)s-l2_%(l2)s-dropout_%(dropout)s-multiply%(multiply)s-basiclstm_%(basic_lstm)s-ignore_%(ignore)s"""
                  % {"layers": config.num_layers,
                     "dense_units": config.dense_units,
                     "hidden": config.hidden_units,
                     "l2": config.l2_reg_lambda,
                     "dropout": config.siamese_keep_prob,
                     "multiply": config.multiply,
                     "basic_lstm": config.basic_lstm,
                     "ignore": config.ignore_one_in_every}).replace('\n', ' ').replace('\r', '')

    txt_suffix = txt_suffix + "-" + str(datetime.datetime.now().isoformat()) + ".txt"
    print("Text file name: ", txt_suffix)
    txt_file = open(txt_suffix,'w')

    print("starting graph def")
    with tf.Graph().as_default(), tf.device("/gpu:0"):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        print("started session")
        with sess.as_default():
            siamese_model = SiameseLSTM(config, vocab_size=len(embedding_matrix))

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.lr)
            print("initialized siameseModel object")

        grads_and_vars = optimizer.compute_gradients(siamese_model.loss)
        tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        print("defined training_ops")

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", siamese_model.loss)
        # acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        print("init all variables")
        graph_def = tf.get_default_graph().as_graph_def()
        graphpb_txt = str(graph_def)
        with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
            f.write(graphpb_txt)


        last_validation_loss = 1000
        checkpoint_model = ''
        for epoch in range(config.num_epochs):
            batches = train_batch_iter(config.batch_size)
            loss = 0.0
            num = 0
            i = 0
            print('Starting epoch: {} at {}'.format(epoch, datetime.datetime.now().isoformat()))
            for batch in tqdm(batches):
                i += 1
                if i % config.ignore_one_in_every == 0:
                    continue
                x1_batch, x2_batch, x1_len, x2_len, y_batch, ids = zip(*batch)
                step_loss = step(x1_batch, x2_batch, x1_len, x2_len, y_batch,
                                 siamese_model, sess, global_step, tr_op_set,
                                 config.dropout_keep_prob, embedding_matrix,
                                 train_summary_writer, train_summary_op)
                loss += step_loss
                num += len(batch) / config.batch_size

                txt_file.write('Train loss at iteration {} is {}'.
                          format(i, loss / num))
                txt_file.write("\n")
                txt_file.flush()

                if num % 100 == 0:
                    print('Train [{}] loss at step {} is {}'.
                          format(datetime.datetime.now().isoformat(),
                                 num, loss / num))
            current_step = tf.train.global_step(sess, global_step)
            print("Train [{}]: after epoch {} loss is {}"
                  .format(datetime.datetime.now().isoformat(),
                          epoch, loss / num))

            if epoch % config.evaluate_every == 0:
                print("\n Evaluation after epoch: {}".format(epoch))
                dev_batches = val_batch_iter(config.batch_size)
                loss = 0.0
                num = 0
                i=0
                for db in tqdm(dev_batches):
                    if len(db) < 1:
                        continue
                    x1_dev, x2_dev, x1_len_dev, x2_len_dev, y_dev, id_dev = zip(*db)
                    if len(y_dev) < 1:
                        continue
                    step_loss = step(x1_dev, x2_dev, x1_len_dev, x2_len_dev, y_dev,
                                     siamese_model, sess, global_step, tr_op_set,
                                     config.siamese_keep_prob, embedding_matrix,
                                     dev_summary_writer, dev_summary_op, evaluate=True)
                    loss += step_loss
                    num += len(db) / config.batch_size
                    txt_file.write('Validation loss at iteration {} is {}'.
                                   format(i, loss / num))
                    txt_file.write("\n")
                    txt_file.flush()
                    i += 1
                print("Validation [{}]:  after epoch {} loss is {}"
                      .format(datetime.datetime.now().isoformat(),
                              epoch, loss / num))

                saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(),
                                     checkpoint_prefix, "graph" + str(epoch) + ".pb",
                                     as_text=False)
                print("Saved model {} with validation loss ={} checkpoint to {}\n"
                      .format(epoch, loss / num, checkpoint_prefix))
                checkpoint_model = checkpoint_prefix + "-" + str(current_step)

                if loss > last_validation_loss:
                    if config.early_stopping:
                        return last_checkpoint_model, last_validation_loss

                last_validation_loss = loss
                last_checkpoint_model = checkpoint_model

        print("Done!!!")
        txt_file.close()
        return checkpoint_model,last_validation_loss
