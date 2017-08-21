import os
import random
import sys

from input_processor import train_batch_iter, val_batch_iter, test_batch_iter

from dhira.predict import generate_prediction
from train import train_network

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
module_path = os.path.abspath(os.path.join('.'))

if module_path not in sys.path:
    sys.path.append(module_path)

def run_network_and_gen_preds(n_layers, dense_units, hidden_units, l2_reg_lambda,
                              dropout_keep_prob, attention, multiply, basic_lstm,
                              ignore_one_in_every):
    max_doc_length = 50
    embedding_dim = 300
    batch_size = 1024
    num_epochs = 20
    evaluate_every = 1
    lr = 1e-3

    (checkpoint_file, loss) = train_network(n_layers=n_layers,
                                    dense_units=dense_units,
                                    max_doc_length=max_doc_length,
                                    embedding_dim=embedding_dim, hidden_units=hidden_units,
                                    l2_reg_lambda=l2_reg_lambda,
                                    batch_size=batch_size, num_epochs=num_epochs,
                                    evaluate_every=evaluate_every,
                                    dropout_keep_prob=dropout_keep_prob, lr=lr,
                                    multiply=multiply, basic_lstm=basic_lstm,
                                    ignore_one_in_every=ignore_one_in_every)

    csv_suffix = ("""layers_%(layers)s-dense_units_%(dense_units)s-hidden_%(hidden)s
                        -l2_%(l2)s-dropout_%(dropout)s-multiply%(multiply)s-basiclstm_%(basic_lstm)s
                        -ignore_%(ignore)s-loss_%(loss)s
                        """ % {"layers": n_layers, "dense_units": dense_units,
                               "hidden": hidden_units, "l2": l2_reg_lambda,
                               "dropout": dropout_keep_prob, "multiply": multiply,
                               "basic_lstm": basic_lstm, "ignore": ignore_one_in_every,
                               "loss": loss}).replace('\n', ' ').replace('\r', '')

    csv_suffix = "".join(csv_suffix.split())

    print(csv_suffix)

    train_pred = generate_prediction(checkpoint_file= checkpoint_file, batch_size=1024,
                                     batch_generator=train_batch_iter, has_labels=True)
    train_pred.to_csv(("train-" + csv_suffix), index = False)

    valid_pred = generate_prediction(checkpoint_file=checkpoint_file, batch_size=1024,
                                     batch_generator=val_batch_iter, has_labels=True)
    valid_pred.to_csv("valid-" + csv_suffix, index=False)

    test_pred = generate_prediction(checkpoint_file=checkpoint_file, batch_size=1024,
                              batch_generator=test_batch_iter, has_labels=False)
    test_pred.to_csv("test-" + csv_suffix, index = False)


def run():
    n_layers_params = [1,2]
    ignore_one_in_every_params = [3,4,5]
    hidden_units_params = [64, 100, 200, 300]
    l2_reg_params = [0.0, 0.05, 0.1, 0.15]
    dropout_params = [0.3, 0.4, 0.5, 0.6]
    dense_units_params = [512, 1024, 2048]
    bool_params = [True, False]

    n_layers_params = [3]
    ignore_one_in_every_params = [3]
    hidden_units_params = [256]
    l2_reg_params = [0.1]
    dropout_params = [0.6]
    dense_units_params = [1024]
    bool_params = [True]

    run_network_and_gen_preds(n_layers=random.choice(n_layers_params),
                              dense_units= random.choice(dense_units_params),
                              hidden_units=random.choice(hidden_units_params),
                              l2_reg_lambda=random.choice(l2_reg_params),
                              dropout_keep_prob=random.choice(dropout_params),
                              attention=random.choice(bool_params),
                              multiply=random.choice(bool_params),
                              basic_lstm=False,
                              ignore_one_in_every=random.choice(ignore_one_in_every_params))

run()