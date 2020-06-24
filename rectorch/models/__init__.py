"""This module includes the implementation of many recommender systems.

TODO
"""

__all__ = ['baseline', 'nn', 'RecSysModel']

class RecSysModel():
    r"""Abstract base class that any Recommendation model must inherit from.
    """
    def train(self, train_data, **kwargs):
        r"""Training procedure.

        This method is meant to execute all the training phase. Once the method ends, the
        model should be ready to be queried for predictions.

        Parameters
        ----------
        train_data : :class:`rectorch.samplers.Sampler` or :class:`scipy.sparse.csr_matrix` or\
            :class:`torch.Tensor`
            This object represents the training data. If the training procedure is based on
            mini-batches, then ``train_data`` should be a :class:`rectorch.samplers.Sampler`.
        **kargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for performing the
            training.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError()

    def predict(self, x, *args, **kwargs):
        r"""Perform the prediction using a trained model.

        The prediction is preformed over a generic input ``x`` and the method should be callable
        after the training procedure (:meth:`RecSysModel.train`).

        Parameters
        ----------
        x : :class:`rectorch.samplers.Sampler` or :class:`scipy.sparse.csr_matrix` or\
            :class:`torch.Tensor`
            The input for which the prediction has to be computed.
        *args : :obj:`list` [optional]
            These are the potential additional parameters useful to the model for performing the
            prediction.
        **kwargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for performing the
            prediction.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError()

    def save_model(self, filepath, *args, **kwargs):
        r"""Save the model to file.

        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file to save the model.
        *args : :obj:`list` [optional]
            These are the potential additional parameters useful to the model for performing the
            prediction.
        **kwargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for performing the
            prediction.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError()

    def load_model(self, filepath, *args, **kwargs):
        r"""Load the model from file.

        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file where the model is saved.
        *args : :obj:`list` [optional]
            These are the potential additional parameters useful to the model for performing the
            prediction.
        **kwargs : :obj:`dict` [optional]
            These are the potential keyword parameters useful to the model for performing the
            prediction.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError()
