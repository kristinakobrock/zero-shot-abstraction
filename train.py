# code based on https://github.com/XeniaOhmer/hierarchical_reference_game/blob/master/train.py
# and https://github.com/jayelm/emergent-generalization/blob/master/code/train.py

import argparse
import torch.nn as nn
import egg.core as core
from language_analysis_local import *
import os
import pickle

import dataset
from archs import Sender, Receiver, RSASender
import itertools
import time

SPLIT = (0.6, 0.2, 0.2)
SPLIT_ZERO_SHOT = (0.75, 0.25)


def get_params(params):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--load_dataset', type=str, default=None,
                        help='If provided that data set is loaded. Datasets can be generated with pickle.ds'
                             'This makes sense if running several runs with the exact same dataset.')
    parser.add_argument('--dimensions', nargs='+', type=int, default=None)
    parser.add_argument('--attributes', type=int, default=3)
    parser.add_argument('--values', type=int, default=4)
    parser.add_argument('--game_size', type=int, default=10)
    parser.add_argument('--scaling_factor', type=int, default=10,
                        help='For scaling up the symbolic datasets.')
    parser.add_argument('--vocab_size_factor', type=int, default=3,
                        help='Factor applied to minimum vocab size to calculate actual vocab size')
    parser.add_argument('--vocab_size_user', type=int, default=16,
                        help='Determines the vocab size. Use only if vocab size factor is None.')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Size of the hidden layer of Sender and Receiver,\
                             the embedding will be half the size of hidden ')
    parser.add_argument('--sender_cell', type=str, default='gru',
                        help='Type of the cell used for Sender {rnn, gru, lstm}')
    parser.add_argument('--receiver_cell', type=str, default='gru',
                        help='Type of the cell used for Receiver {rnn, gru, lstm}')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for Sender's and Receiver's parameters ")
    parser.add_argument('--temperature', type=float, default=2,
                        help="Starting GS temperature for the sender")
    parser.add_argument('--length_cost', type=float, default=0.0,
                        help="linear cost term per message length")
    parser.add_argument('--temp_update', type=float, default=0.99,
                        help="Minimum is 0.5")
    parser.add_argument('--save', type=bool, default=False, help="If set results are saved")
    parser.add_argument('--num_of_runs', type=int, default=1, help="How often this simulation should be repeated")
    parser.add_argument('--zero_shot', type=bool, default=False,
                        help="If set then zero_shot dataset will be trained and tested")
    parser.add_argument('--zero_shot_test', type=str, default=None,
                        help="Must be either 'generic' or 'specific'.")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Specifies the device for tensor computations. Defaults to 'cuda'.")
    parser.add_argument('--path', type=str, default="",
                        help="Path where to save the results - needed for running on HPC3.")
    parser.add_argument('--context_unaware', type=bool, default=False,
                        help="If set to True, then the speakers will be trained context-unaware, i.e. without access to the distractors.")
    parser.add_argument('--max_mess_len', type=int, default=None,
                        help="Allows user to specify a maximum message length. (defaults to the number of attributes in a dataset)")
    parser.add_argument("--early_stopping", type=bool, default=False,
                        help="Use for early stopping with loss, specify patience and min_delta for correct usage.")
    parser.add_argument("--patience", type=int, default=10,
                        help="How many epochs to wait for a significant improvement of loss before early stopping.")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="How much of an improvement to consider a significant improvement of loss before early "
                             "stopping.")
    parser.add_argument("--min_acc_early_stopping", type=float, default=0.00,
                        help="Minimum validation accuracy that needs to reached before early stopping can apply.")
    parser.add_argument("--save_test_interactions", type=bool, default=False,
                        help="Use to save test interactions.")
    parser.add_argument("--save_test_interactions_as", type=str, default="test",
                        help="Specify folder in which to save the test interactions (useful for comparing multiple "
                             "scenarios).")
    parser.add_argument("--load_checkpoint", type=bool, default=False,
                        help="Skip training and load pretrained models from checkpoint.")
    parser.add_argument("--load_interaction", type=bool, default=False,
                        help="If given, load all interactions from previous runs in the folder. Else, take interaction"
                             "from the current run.")

    args = core.init(parser, params)

    return args


def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    Loss needs to be defined for gumbel softmax relaxation.
    For a discriminative game, accuracy is computed by comparing the index with highest score in Receiver
    output (a distribution of unnormalized probabilities over target positions) and the corresponding 
    label read from input, indicating the ground-truth position of the target.
    Adaptation to concept game with multiple targets after Mu & Goodman (2021) with BCEWithLogitsLoss
        receiver_output: Tensor of shape [batch_size, n_objects]
        labels: Tensor of shape [batch_size, n_objects]
    """
    # after Mu & Goodman (2021):
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(receiver_output, labels)
    receiver_pred = (receiver_output > 0).float()
    per_game_acc = (receiver_pred == labels).float().mean(1).cpu().numpy()  # all labels have to be predicted correctly
    acc = per_game_acc.mean()
    return loss, {'acc': acc}


def train(opts, datasets, verbose_callbacks=False):
    """
    Train function completely copied from hierarchical_reference_game.
    """

    if opts.save:
        if not os.path.exists(opts.save_path):
            os.makedirs(opts.save_path)
        if not opts.save_test_interactions:
            pickle.dump(opts, open(opts.save_path + '/params.pkl', 'wb'))
        save_epoch = opts.n_epochs
    else:
        save_epoch = None

    train, val, test = datasets
    # print("train", train)
    dimensions = train.dimensions

    train = torch.utils.data.DataLoader(train, batch_size=opts.batch_size, shuffle=True)
    val = torch.utils.data.DataLoader(val, batch_size=opts.batch_size, shuffle=False, drop_last=True)
    test = torch.utils.data.DataLoader(test, batch_size=opts.batch_size, shuffle=False)

    sender = Sender(opts.hidden_size, sum(dimensions), opts.game_size, opts.context_unaware)
    receiver = Receiver(sum(dimensions), opts.hidden_size)

    if opts.vocab_size_factor != 0:
        minimum_vocab_size = dimensions[0] + 1  # plus one for 'any'
        vocab_size = minimum_vocab_size * opts.vocab_size_factor + 1  # multiply by factor plus add one for eos-symbol
    else:
        vocab_size = opts.vocab_size_user
    print("vocab size", vocab_size)
    # allow user to specify a maximum message length
    if opts.max_mess_len:
        max_len = opts.max_mess_len
    # default: number of attributes
    else:
        max_len = len(dimensions)
    print("message length", max_len)

    # initialize game
    sender = core.RnnSenderGS(sender,
                              vocab_size,
                              int(opts.hidden_size / 2),
                              opts.hidden_size,
                              cell=opts.sender_cell,
                              max_len=max_len,
                              temperature=opts.temperature)

    receiver = core.RnnReceiverGS(receiver,
                                  vocab_size,
                                  int(opts.hidden_size / 2),
                                  opts.hidden_size,
                                  cell=opts.receiver_cell)

    game = core.SenderReceiverRnnGS(sender, receiver, loss, length_cost=opts.length_cost)

    # set learning rates
    optimizer = torch.optim.Adam([
        {'params': game.sender.parameters(), 'lr': opts.learning_rate},
        {'params': game.receiver.parameters(), 'lr': opts.learning_rate}
    ])

    # setup training and callbacks
    # results/ data set name/ kind_of_dataset/ run/
    callbacks = [SavingConsoleLogger(print_train_loss=True, as_json=True,
                                     save_path=opts.save_path, save_epoch=save_epoch),
                 core.TemperatureUpdater(agent=sender, decay=opts.temp_update, minimum=0.5)]
    if opts.save:
        callbacks.extend([core.callbacks.InteractionSaver([opts.n_epochs],
                                                          test_epochs=[opts.n_epochs],
                                                          checkpoint_dir=opts.save_path),
                          core.callbacks.CheckpointSaver(opts.save_path, checkpoint_freq=0)])
    if opts.early_stopping:
        callbacks.extend([InteractionSaverEarlyStopping([opts.n_epochs],
                                                        test_epochs=[opts.n_epochs],
                                                        checkpoint_dir=opts.save_path),
                          EarlyStopperLossWithPatience(patience=opts.patience, min_delta=opts.min_delta,
                                                       min_acc=opts.min_acc_early_stopping)])

    trainer = core.Trainer(game=game, optimizer=optimizer,
                           train_data=train, validation_data=val, callbacks=callbacks)

    # if checkpoint path is given, load checkpoint and skip training
    if opts.load_checkpoint:
        trainer.load_from_checkpoint(opts.checkpoint_path, map_location=opts.device)
    else:
        trainer.train(n_epochs=opts.n_epochs)

    # after training evaluate performance on the test data set
    if len(test):
        trainer.validation_data = test
        eval_loss, interaction = trainer.eval()
        acc = torch.mean(interaction.aux['acc']).item()
        print("test accuracy: " + str(acc))
        if opts.save:
            if not opts.save_test_interactions:
                loss_and_metrics_path = os.path.join(opts.save_path, 'loss_and_metrics.pkl')
            else:
                loss_and_metrics_path = os.path.join(opts.save_path, 'loss_and_metrics_' +
                                                     opts.save_test_interactions_as + '.pkl')
            if os.path.exists(loss_and_metrics_path):
                with open(loss_and_metrics_path, 'rb') as pickle_file:
                    loss_and_metrics = pickle.load(pickle_file)
            else:
                loss_and_metrics = {}

            loss_and_metrics['final_test_loss'] = eval_loss
            loss_and_metrics['final_test_acc'] = acc
            if not opts.save_test_interactions:
                pickle.dump(loss_and_metrics, open(opts.save_path + '/loss_and_metrics.pkl', 'wb'))
            else:
                pickle.dump(loss_and_metrics, open(opts.save_path + '/loss_and_metrics_' +
                                                   opts.save_test_interactions_as + '.pkl', 'wb'))
                InteractionSaver.dump_interactions(interaction, mode=opts.save_test_interactions_as, epoch=0,
                                                   rank=str(trainer.distributed_context.rank),
                                                   dump_dir=opts.save_interactions_path)

def main(params):
    """
    Dealing with parameters and loading dataset. Copied from hierarchical_reference_game and adapted.
    """
    opts = get_params(params)

    # NOTE: I checked and the default device seems to be cuda
    # Otherwise there is an option in a later pytorch version (don't know about compatibility with egg):
    # torch.set_default_device(opts.device)

    # has to be executed in Project directory for consistency
    # assert os.path.split(os.getcwd())[-1] == 'emergent-abstractions'

    # dimensions calculated from attribute-value pairs:
    if not opts.dimensions:
        opts.dimensions = list(itertools.repeat(opts.values, opts.attributes))

    data_set_name = '(' + str(len(opts.dimensions)) + ',' + str(opts.dimensions[0]) + ')'
    folder_name = (data_set_name + '_game_size_' + str(opts.game_size)
                   + '_vsf_' + str(opts.vocab_size_factor))
    folder_name = os.path.join("results", folder_name)

    # define game setting from args
    if opts.context_unaware:
        context_path = 'context_unaware'
    else:
        context_path = 'context_aware'
    if opts.length_cost != 0.0:
        lc_path = 'length_cost'
    else:
        lc_path = ''
    opts.game_path = os.path.join(opts.path, folder_name, lc_path, context_path)
    opts.save_path = opts.game_path  # Keep game path for calculating which run, i.e. folder to save in

    # if name of precreated data set is given, load dataset
    if opts.load_dataset:
        data_set = torch.load(opts.path + 'data/' + opts.load_dataset)
        print('data loaded from: ' + 'data/' + opts.load_dataset)

    for run in range(opts.num_of_runs):

        # zero-shot                
        if opts.zero_shot:
            # either the zero-shot test condition is given (with pre-generated dataset)
            if opts.zero_shot_test is not None:
                # if not opts.save_test_interactions:
                # create subfolder if necessary
                opts.save_path = os.path.join(opts.game_path, 'zero_shot',
                                              opts.zero_shot_test, str(run))
                if not os.path.exists(opts.save_path) and opts.save:
                    os.makedirs(opts.save_path)
                if opts.save_test_interactions:
                    opts.save_interactions_path = os.path.join(opts.save_path, "interactions")
                    if not os.path.exists(opts.save_interactions_path) and opts.save_test_interactions:
                        os.makedirs(opts.save_interactions_path)

                if not opts.load_dataset:
                    data_set = dataset.DataSet(opts.dimensions,
                                               game_size=opts.game_size,
                                               scaling_factor=opts.scaling_factor,
                                               device=opts.device,
                                               zero_shot=True,
                                               zero_shot_test=opts.zero_shot_test)
            # or both test conditions are generated        
            else:
                # implement two zero-shot conditions: test on most generic vs. test on most specific dataset
                for cond in ['generic', 'specific']:
                    print("Zero-shot condition:", cond)
                    # create subfolder if necessary
                    opts.zs_game_path = os.path.join(opts.game_path, 'zero_shot', cond)
                    opts.save_path = os.path.join(opts.zs_game_path, str(run))
                    if not os.path.exists(opts.save_path) and opts.save:
                        os.makedirs(opts.save_path)
                    if opts.save_test_interactions:
                        opts.save_interactions_path = os.path.join(opts.save_path, "interactions")
                        if not os.path.exists(opts.save_interactions_path) and opts.save_test_interactions:
                            os.makedirs(opts.save_interactions_path)

                    data_set = dataset.DataSet(opts.dimensions,
                                               game_size=opts.game_size,
                                               scaling_factor=opts.scaling_factor,
                                               device=opts.device,
                                               zero_shot=True,
                                               zero_shot_test=cond)
                    train(opts, data_set, verbose_callbacks=False)

            # set checkpoint path
            if opts.load_checkpoint:
                opts.checkpoint_path = os.path.join(opts.save_path, 'final.tar')
                if not os.path.exists(opts.checkpoint_path):
                    raise ValueError(
                        f"Checkpoint file {opts.checkpoint_path} not found.")

            # set interaction path
            if opts.load_interaction:
                opts.interaction_path = os.path.join(opts.save_path, 'interactions', opts.test_rsa,
                                                     'epoch_' +
                                                     str(opts.n_epochs), 'interaction_gpu0')

        if opts.zero_shot_test != None or not opts.zero_shot:
            raise ValueError(f"zero_shot can't be {opts.zero_shot} and/or zero_shot_test can't be {opts.zero_shot_test}"
                             f".")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
