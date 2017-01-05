#!/usr/bin/env python3

"""
    Example using VotedPerceptrons in a MulticlassClassifier for MNIST digit classification.

    Example
    -------
    Prior to training set the desired kernel parameters by changing values in the
    KERNEL_PARAMETERS global variable.

    To train a MulticlassClassifier on MNIST with the following parameters:
        fraction_of_mnist: 0.1 (e.g. 6000 MNIST training images (10% of 60000 total))
        strategy: OVA (one-vs-all)
        error_threshold: 0
        max_epochs: 1
    and save it to multicc_mnist.dill.xz::

        $ python mnist_example.py -f0.1 train OVA 0 1

    To test the MulticlassClassifier multicc_mnist.dill.xz with the following parameters:
        fraction_of_mnist: 0.1 (e.g. 1000 MNIST test images (10% of 10000 total))
    and see the error rate::

        $ python mnist_example.py -f0.1 test

    Notes
    -----
    For this example the following are required:
    -Python 3.3 or higher (>=3.2 for argparse and >=3.3 for lzma).
    -At same level of this script a folder named data containing mnist.hdf5.xz or mnist.hdf5
    -dill module (should already be installed by multiprocess module).
    -h5py module.
    Note on debian based systems to install h5py via pip you may need to do::

        apt-get install libhdf5-dev
        HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/ pip install h5py

    In this script the created instance of MulticlassClassifier is serialized via dill and saved
    to a file.
    Saved MulticlassClassifier and VotedPerceptron instances can be loaded later to:
        -Evaluate its error rate on some test data (as we do here in this example).
        -Trained further by calling its train method with more data.
        -Used in an interactive manner as part of a larger program that calls the instance's
         train and predict methods as needed (e.g. as part of some game AI logic).

         For the last use one should keep in mind that there is no inherit limit set on how many
         prediction vectors a VotedPerceptron can accumulate during training.
         If enforcing a limit is desired (e.g. for resource reasons), one can for example
         periodically check the length of one of the VotedPerceptron attributes:
         prediction_vector_term_coeffs
         prediction_vector_terms
         prediction_vector_votes
         (note they are all in 1-1 correspondence with one another so they have the same size)
         and if beyond a certain size then truncate the first say N entries from each where
         0 <= N <= length(prediction_vector_votes). A theoretical analysis on the pros and cons
         of this truncation strategy is beyond the scope of this note.
"""

import argparse
import lzma
import os
import numpy as np
import h5py
import dill

from context import votedperceptron
from votedperceptron import VotedPerceptron, MulticlassClassifier

# Path to MNIST data.
# Note: If only mnist.hdf5.xz found an uncompressed version, mnist.hdf5, will be created.
DATA_FILE_PATH = os.path.join('.', 'data', 'mnist.hdf5')

# Dictionary of kernel parameters to determine kernel used by the underlying binary classifiers.
KERNEL_PARAMETERS = {'kernel_type': 'polynomial', 'degree': 4, 'gamma': 1, 'coef0': 1}
#KERNEL_PARAMETERS = {'kernel_type': 'gaussian_rbf', 'gamma': 0.03125}

def create_mnist_dictionary(dataset_name, labelset_name):
    """ Creates dictionary of MNIST data and labels for the given dataset_name and labelset_name.

    Parameters
    ----------
    dataset_name : str
        Name of data dataset in file to load. Valid choices are:
        train_images, t10k_images
    labelset_name : str
        Name of label dataset in file to load. Valid choices are:
        train_labels, t10k_labels

    Returns
    -------
    dict
        MNIST data and labels for the given dataset_name and labelset_name.

    Raises
    ------
    FileNotFoundError
        If unable to locate data/mnist.hdf5 or data/mnist.hdf5.xz
    """
    # Initialize mnist, filepath, filepath_compressed.
    mnist = {}
    filepath = DATA_FILE_PATH
    filepath_compressed = filepath + '.xz'

    # Check if uncompressed file exists. If not then make uncompressed version.
    if not os.path.isfile(filepath):
        if os.path.isfile(filepath_compressed):
            with lzma.open(filepath_compressed, 'rb') as compressed_file:
                decoded = compressed_file.read()

            with open(filepath, 'wb') as uncompressed_file:
                uncompressed_file.write(decoded)
        else:
            raise FileNotFoundError('Unable to locate ' + filepath + ' or ' + filepath_compressed)

    # Read uncompressed file.
    with h5py.File(filepath, 'r') as mnist_hdf5:
        data = mnist_hdf5.get(dataset_name)
        labels = mnist_hdf5.get(labelset_name)

        # Copy to mnist dictionary.
        mnist[dataset_name] = np.empty(data.shape, data.dtype)
        mnist[labelset_name] = np.empty(labels.shape, labels.dtype)
        data.read_direct(mnist[dataset_name])
        labels.read_direct(mnist[labelset_name])

    return mnist

def get_data_and_labels(data_type, fraction_of_mnist):
    """ Return appropriate type and amount of mnist data and labels scaled to 0-1 range.

    Parameters
    ----------
    data_type : str
        'train': For training data and labels.
        'test': For test data and labels.
    fraction_of_mnist : float (0-1)
        Fractional amount of data and labels to return.

    Returns
    -------
    2 ndarrays
        data and labels appropriate for the given data_type and fraction_of_mnist.
    """
    # Determine the appropriate datasets to use from the mnist hdf5 file.
    if data_type == 'train':
        dataset_name = 'train_images'
        labelset_name = 'train_labels'

    elif data_type == 'test':
        dataset_name = 't10k_images'
        labelset_name = 't10k_labels'

    # Set mnist dictionary.
    mnist = create_mnist_dictionary(dataset_name, labelset_name)

    # mnist data scaled to 0-1 range.
    num_input_vectors = round(fraction_of_mnist * len(mnist[dataset_name]))
    data = mnist[dataset_name][:num_input_vectors].astype(np.float32) / 255
    labels = mnist[labelset_name][:num_input_vectors]

    return data, labels

def train_mnist(args):
    """ Instantiates and trains a MulticlassClassifier for MNIST and saves it to a file.

    Parameters
    ----------
    args : argparse.Namespace
        argparse.Namespace object containing the command line argument values as attributes.

    Returns
    -------
    list of 2-tuples
        For a given 2-tuple the first element is a binary classifier key and
        the second element is the number of prediction vectors making up the binary classifier.
    """
    # Arguments from the command line.
    strategy = args.strategy
    error_threshold = args.error_threshold
    max_epochs = args.max_epochs
    process_count = args.process_count
    fraction_of_mnist = args.fraction_of_mnist

    # Possible labels are 0-9.
    possible_labels = tuple(range(10))

    # Create instance of MulticlassClassifier.
    multicc = MulticlassClassifier(strategy, possible_labels, VotedPerceptron, (),
                                   {'kernel_parameters':KERNEL_PARAMETERS})
    other_bc_train_args = tuple([error_threshold, max_epochs])
    other_bc_train_kwargs = {}

    # Get data and labels for training.
    print('Loading MNIST data')
    data, labels = get_data_and_labels('train', fraction_of_mnist)

    # Train instance of MulticlassClassifier.
    print('Training')
    multicc.train(data, labels, other_bc_train_args, other_bc_train_kwargs, process_count)

    # Save trained MulticlassClassifier.
    print('Saving MulticlassClassifier')
    save_filepath = 'multicc_mnist.dill.xz'
    with lzma.open(save_filepath, 'wb') as multicc_file:
        dill.dump(multicc, multicc_file)

    # Return number of prediction vectors making up each binary classifier.
    bc_vector_counts = [(k, len(v.prediction_vector_votes))
                        for k, v in multicc.binary_classifiers.items()]

    return bc_vector_counts

def test_mnist(args):
    """ Calculates error rate on MNIST for a previously saved MulticlassClassifier instance.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from command line.

    Returns
    -------
    error_rate : float
        The calculated error rate.

    Raises
    ------
    FileNotFound
        If unable to locate multicc_mnist.dill.xz
    """
    # Arguments from the command line.
    process_count = args.process_count
    fraction_of_mnist = args.fraction_of_mnist

    # Load trained MulticlassClassifier.
    print('Loading MulticlassClassifier')
    load_filepath = 'multicc_mnist.dill.xz'
    if os.path.isfile(load_filepath):
        with lzma.open(load_filepath, 'rb') as multicc_file:
            multicc = dill.load(multicc_file)
    else:
        raise FileNotFoundError('Unable to locate ' + load_filepath)

    # Set other binary classifier predict arguments.
    # We set option so the voted perceptron predict method returns a real-valued confidence score
    # as opposed to a label. This is required by MulticlassClassifier.
    other_bc_predict_args = tuple(['score'])
    other_bc_predict_kwargs = {}

    # Get data and labels for testing.
    print('Loading MNIST data')
    data, labels = get_data_and_labels('test', fraction_of_mnist)

    # Calculate error rate on test data.
    print('Computing error rate')
    error_rate = multicc.error_rate(data, labels,
                                    other_bc_predict_args, other_bc_predict_kwargs,
                                    process_count)

    return error_rate

def main():
    """ Define and parse command line arguments.

    """
    # Create the top-level parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--process_count',
                        help='number of worker processes to use',
                        type=int,
                        choices=range(1, os.cpu_count() + 1),
                        default=os.cpu_count())
    parser.add_argument('-f', '--fraction_of_mnist',
                        help='fraction of MNIST data to use (range 0 to 1)',
                        type=float,
                        default=1)
    subparsers = parser.add_subparsers(help='sub-command help')

    # Create the parser for the train command.
    parser_train = subparsers.add_parser('train',
                                         help='create and train a MulticlassClassifier')
    parser_train.add_argument('strategy',
                              help='strategy',
                              choices=['OVA', 'OVO'],
                              default='OVA')
    parser_train.add_argument('error_threshold',
                              help='error_threshold',
                              type=float,
                              default=0)
    parser_train.add_argument('max_epochs',
                              help='max_epochs',
                              type=int,
                              default=1)
    parser_train.set_defaults(func=train_mnist)

    # Create the parser for the test command.
    parser_test = subparsers.add_parser('test',
                                        help='test trained MulticlassClassifier')
    parser_test.set_defaults(func=test_mnist)

    # Parse arguments and call appropriate function (train or test).
    args = parser.parse_args()
    print(args.func(args))

if __name__ == '__main__':
    main()
