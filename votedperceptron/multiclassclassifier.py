# Copyright 2015 B. Gee. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Multiclass Classifier.

A multiclass classifier based on multiple binary classifiers.

"""
from itertools import combinations
import numpy as np
from multiprocess import Pool

class MulticlassClassifier(object):
    """ Multiclass Classifier constructed from binary classifiers.

    Parameters
    ----------
    strategy : str
        Strategy to use to transform multiclass classification
        problem to multiple binary classification problems.
        Valid choices are:
            OVA: One-vs-All
            OVO: One-vs-One
        If there are N possible_labels then OVA will require building
        N binary classifiers (one for every possible label) and OVO will
        require building (N choose 2) (i.e. N*(N-1)/2) binary classifiers
        (one for every possible pair of labels).
    possible_labels : tuple
        Tuple of the possible classification labels.
    BinaryClassifier : class
        Class to build the binary classifiers.
        Class should implement the following methods:
            train(data : ndarray, labels : ndarray, *)
            predict(input_vector : ndarray, *) -- See Notes.
    bc_args : tuple
        Positional arguments to BinaryClassifier.__init__
    bc_kwargs : dict
        Keyword arguments to BinaryClassifier.__init__

    Attributes
    ----------
    strategy : str
        Same as parameter of same name.

    possible_labels : tuple
        Same as parameter of same name.

    Raises
    ------
    NotImplementedError
        If strategy not in ('OVA', 'OVO')

    Notes
    -----
    The predict method of the underlying Binary Classifiers should be set to return
    a real-valued confidence score for its decision, rather than just a class label.
    """
    def __init__(self, strategy, possible_labels, BinaryClassifier, bc_args, bc_kwargs):
        self.strategy = strategy
        self.possible_labels = possible_labels

        # self.binary_classifiers is a dictionary of the underlying binary classifiers keyed
        # by a 2-tuple of the label(s) they are used to classify.
        # In the OVA case the keys are (x, None) since the binary classifiers are used
        # to predict whether an input vector is of classification label x or not.
        # In the OVO case the keys are (x, y) since the binary classifiers are used to
        # predict whether an input vector is of classification label x or label y with
        # positive scores indicating label x and negative scores indicating label y.
        if self.strategy == 'OVA':
            self.binary_classifiers = {(x, None):
                                       BinaryClassifier(*bc_args, **bc_kwargs)
                                       for x in self.possible_labels}
        elif self.strategy == 'OVO':
            self.binary_classifiers = {(x, y):
                                       BinaryClassifier(*bc_args, **bc_kwargs)
                                       for x, y in combinations(self.possible_labels, 2)}
        else:
            raise NotImplementedError(self.strategy)

    def train(self, data, labels, other_bc_train_args, other_bc_train_kwargs, process_count):
        """ Train the Multiclass Classifier.
            (i.e. by training the underlying binary classifiers).

        Parameters
        ----------
        data : ndarray
            An ndarray where each row is a training case consisting of the
            state of visible units.
        labels : ndarray
            An ndarray where each element is the label/classification of a
            training case in data for binary classification.
            Valid label values are -1 and 1.
        other_bc_train_args : tuple
            Positional arguments to BinaryClassifier.train not including data and labels.
        other_bc_train_kwargs : dict
            Keyword arguments to BinaryClassifier.train not including data and labels.
        process_count : int
            The number of worker processes to use when training the binary classifiers.

        Notes
        -----
        The elements in data must correspond in sequence to the
        elements in labels.
        """
        def strategized_data_and_labels(strategy, label_1, label_2, data, labels):
            """ Generate data and labels appropriate for the given strategy and
                binary classifier defined by label_1 and label_2.

            Parameters
            ----------
            strategy : str
                Strategy to use to transform multiclass classification
                problem to multiple binary classification problems.
                Valid choices are:
                    OVA: One-vs-All
                    OVO: One-vs-One
                If there are N possible_labels then OVA will require building
                N binary classifiers (one for every possible label) and OVO will
                require building (N choose 2) (i.e. N*(N-1)/2) binary classifiers
                (one for every possible pair of labels).
            label_1, label_2 : str
                The key of the binary classifier to be trained.
            data : ndarray
                An ndarray where each row is a training case consisting of the
                state of visible units.
            labels : ndarray
                An ndarray where each element is the label/classification of a
                training case in data for binary classification.
                Valid label values are -1 and 1.

            Returns
            -------
            tuple
                2 element tuple where first element is strategized_data and
                the second element is strategized_labels.

            Raises
            ------
            NotImplementedError
                If strategy not in ('OVA', 'OVO')
            """
            if strategy == 'OVA':
                # strategized_labels entries are 1 if label_1, -1 otherwise.
                strategized_labels = np.where(np.in1d(labels, label_1),
                                              np.ones(labels.shape, np.int8),
                                              -np.ones(labels.shape, np.int8))
                strategized_data = data
            elif strategy == 'OVO':
                # strategized labels and data only containing entries
                # corresponding to label_1 and label_2 and strategized_labels
                # entries being 1 if label_1, -1 if label_2.
                strategized_condition = np.in1d(labels, (label_1, label_2))
                strategized_prelabels = labels[strategized_condition]
                strategized_labels = np.where(np.in1d(strategized_prelabels, label_1),
                                              np.ones(strategized_prelabels.shape, np.int8),
                                              -np.ones(strategized_prelabels.shape, np.int8))
                strategized_data = data[strategized_condition]
            else:
                raise NotImplementedError(strategy)

            return (strategized_data, strategized_labels)

        def managed_binary_classifier_train(binary_classifier,
                                            strategized_data, strategized_labels,
                                            other_bc_args, other_bc_kwargs):
            """ Train a specific binary classifier.

            Parameters
            ----------
            binary_classifier : binary classifier
                The binary classifier to train.
            strategized_data, strategized_labels : ndarray
                data and labels relevant for the training of the
                binary classifier.
            other_bc_train_args : tuple
                Positional arguments to BinaryClassifier.train not including data and labels.
            other_bc_train_kwargs : dict
                Keyword arguments to BinaryClassifier.train not including data and labels.

            Returns
            -------
            binary classifier
                The trained binary classifier.
            """
            binary_classifier.train(strategized_data, strategized_labels,
                                    *other_bc_args, **other_bc_kwargs)

            return binary_classifier

        def error_callback(exc):
            """ Callback used by pool.apply_async when an error occurs.

            Parameters
            ----------
            exc : Exception
                Exception thrown by the process pool.apply_async was running in.

            """
            print(exc.__cause__)

        if process_count is not None and process_count > 1:
            trained_bc_results = {}
            with Pool(processes=process_count) as pool:
                # Use the process pool to initiate training of the
                # binary classifiers.
                for (label_1, label_2), binary_classifier in self.binary_classifiers.items():
                    (strategized_data, strategized_labels) = (
                        strategized_data_and_labels(self.strategy, label_1, label_2, data, labels)
                    )

                    trained_bc_results[(label_1, label_2)] = (
                        pool.apply_async(func=managed_binary_classifier_train,
                                         args=(binary_classifier,
                                               strategized_data, strategized_labels,
                                               other_bc_train_args, other_bc_train_kwargs),
                                         error_callback=error_callback)
                    )

                # Retrieve the trained binary classifiers back from the process pool and
                # replace the untrained instances in self.binary_classifiers with
                # their trained counterparts.
                for (label_1, label_2) in self.binary_classifiers.keys():
                    self.binary_classifiers[(label_1, label_2)] = (
                        trained_bc_results[(label_1, label_2)].get()
                    )

        else:
            # Train each binary classifier.
            for (label_1, label_2), binary_classifier in self.binary_classifiers.items():
                (strategized_data, strategized_labels) = (
                    strategized_data_and_labels(self.strategy, label_1, label_2, data, labels)
                )

                binary_classifier.train(strategized_data, strategized_labels,
                                        *other_bc_train_args, **other_bc_train_kwargs)

    def predict(self, input_vector, other_bc_predict_args, other_bc_predict_kwargs):
        """ Output predicted label of multiclass classifier and given input vector.

        Based on output of underlying binary classifiers for given input vector.

        Parameters
        ----------
        input_vector : ndarray
            A given state of visible units.
        other_bc_predict_args : tuple
            Positional arguments to BinaryClassifier.predict not including input_vector.
        other_bc_predict_kwargs : dict
            Keyword arguments to BinaryClassifier.predict not including input_vector.

        Returns
        -------
        str
            The label the multiclass classifer predicts for the given input_vector.
        """
        bc_scores = {(label_1, label_2): binary_classifier.predict(input_vector,
                                                                   *other_bc_predict_args,
                                                                   **other_bc_predict_kwargs)
                     for (label_1, label_2), binary_classifier in self.binary_classifiers.items()}

        if self.strategy in ('OVA', 'OVO'):
            # Compute a confidence score for each label and set the predicted label to be
            # the one with the highest score.
            # For OVA it will just be the score output by the single corresponding binary classifier
            # for the label (i.e. the binary classifier keyed by (label, None)).
            # For OVO it will be the sum of appropriate scores output by all the binary
            # classifiers classifying that label
            # (i.e. binary classifiers with the label in any position in the key tuple).
            # For OVO what we mean by appropriate scores is if the label is in the second position
            # of the key tuple then we take the negative of the score output by the binary
            # classifier since positive scores predict the label in the first position of the
            # key tuple and negative scores indicates predict the label in the second position
            # of the key tuple.
            # The following dictionary comprehension works for both cases.
            label_scores = {label: sum(score if label == label_1 else -score
                                       for (label_1, label_2), score in bc_scores.items()
                                       if label in (label_1, label_2))
                            for label in self.possible_labels}
            predicted_label = max(label_scores, key=label_scores.get)
        else:
            raise NotImplementedError(self.strategy)

        return predicted_label

    def error_rate(self, data, labels,
                   other_bc_predict_args, other_bc_predict_kwargs,
                   process_count):
        """ Outputs the error rate of multiclass classifier for the given data and labels.

        Parameters
        ----------
        data : ndarray
            An ndarray where each row is a input vector consisting of the
            state of the visible units.
        labels : ndarray
            An ndarray where each element is the label/classification of a
            input vector in data for binary classification.
            Valid label values are -1 and 1.
        other_bc_predict_args : tuple
            Positional arguments to BinaryClassifier.predict not including input_vector.
        other_bc_predict_kwargs : dict
            Keyword arguments to BinaryClassifier.predict not including input_vector.
        process_count : int
            The number of worker processes to use when generating the predictions.

        Note the elements in data must correspond in sequence to the
        elements in labels.

        Returns
        -------
        float
            The error rate of the multiclass classifier for the given data
            and labels.
        """
        def binary_classifier_predict(binary_classifier, data,
                                      other_bc_predict_args, other_bc_predict_kwargs):
            """ Generate predictions for a specific binary classifier.

            Parameters
            ----------
            binary_classifier : binary classifier
                The binary classifier to generate predictions.
            data : ndarray
                Data to generate predictions for.
            other_bc_predict_args : tuple
                Positional arguments to BinaryClassifier.predict not including input_vector.
            other_bc_predict_kwargs : dict
                Keyword arguments to BinaryClassifier.predict not including input_vector.

            Returns
            -------
            list
                The predictions of the given binary classifier.
            """
            bc_data_scores = [binary_classifier.predict(input_vector,
                                                        *other_bc_predict_args,
                                                        **other_bc_predict_kwargs)
                              for input_vector in data]

            return bc_data_scores

        def error_callback(exc):
            """ Callback used by pool.apply_async when an error occurs.

            Parameters
            ----------
            exc : Exception
                Exception thrown by the process pool.apply_async was running in.
            """
            print(exc.__cause__)

        if process_count is not None and process_count > 1:
            # Unfortunately we cannot just use self.predict directly
            # (e.g. predictions = pool.map(self.predict, data)).
            # Instead must partially repeat what self.predict does here.
            binary_classifier_results = {}
            binary_classifier_scores = {}
            with Pool(processes=process_count) as pool:
                # Use the process pool to compute predictions of the binary classifiers.
                for (label_1, label_2), binary_classifier in self.binary_classifiers.items():
                    binary_classifier_results[(label_1, label_2)] = (
                        pool.apply_async(func=binary_classifier_predict,
                                         args=(binary_classifier, data,
                                               other_bc_predict_args,
                                               other_bc_predict_kwargs),
                                         error_callback=error_callback)
                    )

                # Retrieve the binary classifier scores from the process pool.
                for (label_1, label_2) in self.binary_classifiers.keys():
                    binary_classifier_scores[(label_1, label_2)] = (
                        binary_classifier_results[(label_1, label_2)].get()
                    )

            # Generate list of predictions for each data element based on the predictions of
            # the underlying binary classifiers.
            predictions = []
            if self.strategy in ('OVA', 'OVO'):
                # Compute a confidence score for each label and set the predicted label to be
                # the one with the highest score.
                # Same technique as in self.predict with the difference here being
                # binary_classifier_scores values are not a single value and instead are an
                # iterable of scores for each data element.
                for i in range(len(data)):
                    label_scores = {label: sum(scores[i] if label == label_1 else -scores[i]
                                               for (label_1, label_2), scores
                                               in binary_classifier_scores.items()
                                               if label in (label_1, label_2))
                                    for label in self.possible_labels}
                    predicted_label = max(label_scores, key=label_scores.get)
                    predictions.append(predicted_label)
            else:
                raise NotImplementedError(self.strategy)

            predictions = np.asarray(predictions, dtype=labels.dtype)

        else:
            # Generate list of predictions for each data element using self.predict.
            predictions = np.asarray(
                [self.predict(input_vector, other_bc_predict_args, other_bc_predict_kwargs)
                 for input_vector in data], dtype=labels.dtype
            )

        # Gather the results of the predictions; prediction_results is an ndarray corresponding
        # to the predictions and the labels for the data with True meaning the prediction matched
        # the label and False meaning it did not.
        prediction_results = (predictions == labels)
        # Note the number of incorrect prediction results
        # (i.e. the number of False entries in prediction_results).
        num_incorrect_prediction_results = np.sum(~prediction_results)
        # Note the number of results.
        num_prediction_results = prediction_results.shape[0]
        # Compute the error rate.
        error_rate = num_incorrect_prediction_results / num_prediction_results

        return error_rate
