"""This module includes the implementation of many recommender systems.
"""
# AUTHORSHIP
__version__ = "0.9.0dev"
__author__ = "Mirko Polato"
__copyright__ = "Copyright 2020, rectorch"
__license__ = "MIT"
__maintainer__ = "Mirko Polato"
__email__ = "mak1788@gmail.com"
__status__ = "Development"
#

__all__ = ['baseline', 'nn', 'RecSysModel']

class RecSysModel():
    r"""Abstract base class that any Recommendation model must inherit from.
    """

    def train(self, data_sampler, **kwargs):
        r"""Training procedure.

        This method is meant to execute all the training phase. Once the method ends, the
        model should be ready to be queried for predictions.

        Parameters
        ----------
        data_sampler : :class:`rectorch.samplers.Sampler` or :class:`scipy.sparse.csr_matrix` or\
            :class:`torch.Tensor`
            This object represents the training data. If the training procedure is based on
            mini-batches, then ``data_sampler`` should be a :class:`rectorch.samplers.Sampler`.
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

    @classmethod
    def load_model(cls, filepath, *args, **kwargs):
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

        Returns
        -------
        :class:`rectorch.models.RecSysModel`
            A recommendation model.

        Raises
        ------
        :class:`NotImplementedError`
            Raised when not implemeneted in the sub-class.
        """
        raise NotImplementedError()

    def __str__(self):
        s = self.__class__.__name__ + "(\n"
        for k, v in self.__dict__.items():
            sv = "\n".join(["  "+line for line in str(str(v)).split("\n")])[2:]
            s += "  %s = %s,\n" % (k, sv)
        s = s[:-2] + "\n)"
        return s

    def __repr__(self):
        return str(self)
