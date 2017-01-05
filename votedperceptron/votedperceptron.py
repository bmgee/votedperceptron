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
"""
An implementation of the voted perceptron algorithm
described in the publication below:

%0 Journal Article
%D 1999
%@ 0885-6125
%J Machine Learning
%V 37
%N 3
%R 10.1023/A:1007662407062
%T Large Margin Classification Using the Perceptron Algorithm
%U http://dx.doi.org/10.1023/A%3A1007662407062
%I Kluwer Academic Publishers
%8 1999-12-01
%A Freund, Yoav
%A Schapire, RobertE.
%P 277-296
%G English
"""
from math import copysign, exp
from itertools import accumulate
import numpy as np

class VotedPerceptron(object):
    """ Voted Perceptron.

    Parameters
    ----------
    kernel_parameters : dict
        Kernel parameters appropriate for the desired kernel.
        kernel_type : str
            The desired kernel {'linear'|'polynomial'|'gaussian_rbf'}
        degree : int
            Used by kernel_type: polynomial
        gamma : float
            Used by kernel_type: polynomial, rbf
        coef0 : float
            Used by kernel_type: linear, polynomial

    Attributes
    ----------
    kernel : function
        The kernel function defined by kernel_parameters.

    prediction_vector_term_coeffs : list
        The label components of the prediction_vectors.

    prediction_vector_terms : list
        The training case components of the prediction vectors.

    prediction_vector_votes : list
        The votes for the prediction vectors.

    Notes
    -----
    Each prediction vector can be written in an implicit form based
    on its contruction via training.
    A prediction vector is calculated by adding the training label times the
    training input to the previous prediction vector and since the initial
    prediction vector is the zero vector we have
    :math:`v_k = v_{k-1} + y_i x_i` for some label :math:`y_i` and training case :math:`x_i`.
    From this recurrence we see
    :math:`v_k = \sum_{j=1}^{k-1}{y_{i_j} \vec{x}_{i_j}}` for appropriate indices :math:`i_j`.

    Then the kth prediction vector is the kth partial sum of the
    element-wise product of prediction_vector_term_coeffs
    and prediction_vector_terms.
    Specifically the kth prediction_vector is the sum i=1 to i=k of
    prediction_vector_term_coeffs[i] * prediction_vector_terms[i]
    Note: To get an iterable of the prediction vectors explicitly we can do:
    accumulate(pvtc * pvt
                for pvtc, pvt in
                zip(self.prediction_vector_term_coeffs,
                self.prediction_vector_terms))

    Working with prediction vectors in their implicit form allows us to
    apply the kernel method to the voted perceptron algorithm.
    """
    def __init__(self, kernel_parameters):
        # Set the kernel function
        self.kernel = self.kernel_function(kernel_parameters)

        # Initialize structures that will store the prediction vectors in their
        # implicit form.
        self.prediction_vector_term_coeffs = []
        self.prediction_vector_terms = []

        # Prediction vector votes generated during training.
        self.prediction_vector_votes = []

    def train(self, data, labels, error_threshold=0, max_epochs=1):
        """ Train the Voted Perceptron.

        Parameters
        ----------
        data : ndarray
            An ndarray where each row is a training case consisting of the
            state of visible units.
        labels : ndarray
            An ndarray where each element is the label/classification of a
            training case in data for binary classification.
            Valid label values are -1 and 1.
        error_threshold : float
            Training is stopped if the error_rate for the last epoch is
            below error_threshold.
        max_epochs : int
            The maximum number of epochs to train.

        Notes
        -----
        The elements in data must correspond in sequence to the
        elements in labels.
        """
        # Ensure the data dtype is allowed.
        self.check_data_dtype(data)

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis=1)

        # If needed set the initial prediction vector.
        # Initialize the initial prediction_vector to all zeros.
        if len(self.prediction_vector_terms) == 0:
            initial_prediction_vector = np.zeros(data.shape[1], dtype=data.dtype)
            self.prediction_vector_terms.append(initial_prediction_vector.copy())
            self.prediction_vector_term_coeffs.append(1)

        # shape is tuple of dimensions of array (num_rows, num_columns); so shape[0] is
        # be the total number of input vectors.
        num_training_cases = data.shape[0]

        # Set the starting prediction_vector_vote for training to be the last
        # prediction vector vote defined by self.prediction_vector_votes.
        # Note: self.prediction_vector_votes is empty if we have not done any training yet.
        prediction_vector_vote = (self.prediction_vector_votes[-1]
                                  if len(self.prediction_vector_votes) > 0
                                  else 0)

        for _ in range(max_epochs):
            num_epoch_errors = 0
            for training_case, training_label in zip(data, labels):
                pre_activation = sum(pvtc * self.kernel(pvt, training_case)
                                     for pvtc, pvt
                                     in zip(self.prediction_vector_term_coeffs,
                                            self.prediction_vector_terms))
                result = copysign(1, pre_activation)

                if result == training_label:
                    prediction_vector_vote += 1
                else:
                    num_epoch_errors += 1

                    # Save the prediction vector vote.
                    self.prediction_vector_votes.append(prediction_vector_vote)

                    # Save new prediction vector term and term coefficient.
                    self.prediction_vector_term_coeffs.append(training_label)
                    self.prediction_vector_terms.append(training_case.copy())

                    # Reset prediction_vector_vote.
                    prediction_vector_vote = 1

            epoch_error = num_epoch_errors / num_training_cases

            if epoch_error <= error_threshold:
                # Error for epoch is under the error threshold.
                break

        # Training complete.
        # Save the last prediction_vector_vote.
        self.prediction_vector_votes.append(prediction_vector_vote)

    def predict(self, input_vector, output_type):
        """ Output of voted perceptron and given input vector.

        Parameters
        ----------
        input_vector : ndarray
            A given state of visible units.
        output_type : str
            Determines output, either 'classification' or 'score'.
                'classification': The label the voted perceptron predicts
                    for the given input_vector.
                'score': The pre_activation value the voted perceptron
                    calculates for the given input_vector.

        Returns
        -------
        float
            If output_type is 'classification' then output the classification
            the voted perceptron predicts for the given input_vector: 1 or -1.
            If output_type is 'score' then output the pre_activation value the
            voted perceptron calculates for the given input_vector.
        """
        # Insert a bias unit of 1.
        input_vector = np.insert(input_vector, 0, 1, axis=0)

        pv_pre_activations = accumulate(pvtc * self.kernel(pvt, input_vector)
                                        for pvtc, pvt
                                        in zip(self.prediction_vector_term_coeffs,
                                               self.prediction_vector_terms))

        pre_activation = sum(
            pvv
            * copysign(1, pvpa)
            for pvv, pvpa
            in zip(self.prediction_vector_votes, pv_pre_activations)
        )

        if output_type == 'score':
            result = pre_activation
        elif output_type == 'classification':
            result = copysign(1, pre_activation)

        return result

    def error_rate(self, data, labels):
        """ Outputs the error rate for the given data and labels.

        Parameters
        ----------
        data : ndarray
            An ndarray where each row is a input vector consisting of the
            state of the visible units.
        labels : ndarray
            An ndarray where each element is the label/classification of a
            input vector in data for binary classification.
            Valid label values are -1 and 1.

        Notes
        -----
        The elements in data must correspond in sequence to the
        elements in labels.

        Returns
        -------
        float
            The error rate of the voted perceptron for the given data
            and labels.
        """
        # Ensure the data dtype is allowed.
        self.check_data_dtype(data)

        # Generate the VotedPerceptron output/classification for each
        # input and save as a numpy array.
        predictions = np.asarray(
            [self.predict(d, 'classification') for d in data], dtype=labels.dtype
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

    @staticmethod
    def kernel_function(kernel_parameters):
        """ Output the chosen kernel function given the name and parameters.

        Parameters
        ----------
        kernel_parameters : dict
            Kernel parameters appropriate for the desired kernel.
            kernel_type : str
                The desired kernel {'linear'|'polynomial'|'gaussian_rbf'}
            degree : int
                Used by kernel_type: polynomial
            gamma : float
                Used by kernel_type: polynomial, gaussian_rbf
            coef0 : float
                Used by kernel_type: linear, polynomial

        Returns
        -------
        function
            The chosen kernel function with the appropriate parameters set.

        Raises
        ------
        NotImplementedError
            If strategy not in ('OVA', 'OVO')
        """
        def linear(vector_1, vector_2):
            """
            Linear Kernel
            """
            coef0 = kernel_parameters["coef0"]
            output = np.dot(vector_1, vector_2) + coef0
            return output
        def polynomial(vector_1, vector_2):
            """
            Polynomial Kernel
            """
            gamma = kernel_parameters["gamma"]
            coef0 = kernel_parameters["coef0"]
            degree = kernel_parameters["degree"]
            output = (gamma * np.dot(vector_1, vector_2) + coef0) ** degree
            return output
        def gaussian_rbf(vector_1, vector_2):
            """
            Gaussian Radial Basis Function Kernel
            """
            gamma = kernel_parameters["gamma"]
            vector_difference = vector_1 - vector_2
            output = exp(-gamma * np.dot(vector_difference, vector_difference))
            return output

        kernel_choices = {'linear': linear,
                          'polynomial': polynomial,
                          'gaussian_rbf': gaussian_rbf}

        kernel_type = kernel_parameters['kernel_type']

        if kernel_type not in kernel_choices:
            raise NotImplementedError(kernel_type)

        kernel_choice = kernel_choices[kernel_type]

        return kernel_choice

    @staticmethod
    def check_data_dtype(data):
        """ Check to see if the data dtype is a valid predesignated type.

        Parameters
        ----------
        data : ndarray
            An ndarray where each row is a input vector consisting of the
            state of the visible units.

        Raises
        ------
        TypeError
            If data.dtype is not a valid predesignated type.

        Notes
        -----
        We require data.dtype to be float32 or float64. When numpy is built
        with an accelerated BLAS the dtypes eligible for accelerated operations
        are float32, float64, complex64, complex128 with the latter 2 not relevant
        for the voted perceptron implementation here.
        Also by restricting to floating point types we minimize the possibility of
        any unanticipated overflow issues with regards to np.dot.
        By having this check a user cannot pass in say mnist data as uint8.
        While uint8 does fit mnist data it leads to unexpected np.dot calculations.
        More specifically np.dot does not upcast or warn when integer overflow occurs
        (numpy bugs 4126, 6753).
        e.g. a = np.asarray([1,128], dtype=uint8)
             b = np.asarray([0,2], dtype=uint8)
             np.dot(a,b) would return 0.
        """
        if data.dtype not in (np.float32, np.float64):
            raise TypeError('data dtype required to be float32 or float64')

    @staticmethod
    def validate_inputs(input_vector_size, data, labels):
        """ Validate inputs used for the train method of the voted perceptron.

        Parameters
        ----------
        input_vector_size: The number of visible units.
        data: An ndarray where each row is a input vector consisting of the
              state of the visible units.
        labels: An ndarray where each element is the label/classification of a
                input vector in data for binary classification.
                Valid label values are -1 and 1.

        Note the elements in data must correspond in sequence to the
        elements in labels.

        Raises
        ------
        ValueError
            If the given combination of input_vector_size, data, labels is not
            valid.

        Notes
        -----
        To be used by instantiator to double check that their input data has been
        properly preprocessed.
        """
        # Ensure the number of data items matches the number of labels.
        if len(data) != len(labels):
            raise ValueError("Number of data items does not match"
                             + " the number of labels.")

        # Ensure self.input_vector_size matches size of each item in data.
        if any(input_vector_size != len(data_item) for data_item in data):
            raise ValueError("A data item size does not match"
                             + " input_vector_size.")

        # Ensure set of label values in [-1, 1].
        if not np.all(np.in1d(labels, [-1, 1])):
            raise ValueError("Valid label values are -1 and 1;"
                             + " adjust labels accordingly when calling"
                             + " this function.")
