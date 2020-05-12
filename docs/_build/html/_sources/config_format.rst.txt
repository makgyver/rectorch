.. _config-format:

Configuration files format
==========================

In **rectorch**, te configuration of the data set pre-processing, and the models' training/test 
is performed via `.json <https://www.json.org/json-en.html>`_ files.


Data configuration file
-----------------------

The data configuration file defines how the data set must be pre-processed.
The pre-processing comprehends the reading, clean up, and the partition of the data set.
The `.json <https://www.json.org/json-en.html>`_ data configuration file must have the following key-value pairs:

* ``data_path``: string representing the path to the data set `.csv <https://it.wikipedia.org/wiki/Comma-separated_values>`_ file;
* ``proc_path``: string representing the folder to save the pre-processed files,
* ``separator``: string delimiter used in the `.csv <https://it.wikipedia.org/wiki/Comma-separated_values>`_ file;
* ``seed``: integer random seed used for both the training/validation/test division as well as for shuffling the data;
* ``threshold``: float cut-off value for converting explicit feedback to implicit feedback;
* ``u_min``: integer minimum number of items for a user to be kept in the data set;
* ``i_min``: integer minimum number of users for an item to be kept in the data set;
* ``heldout``: integer heldout size (in number of users) for both the validation and test set;
* ``test_prop``: float in the range (0,1) which represents the proportion of items of the test users that are considered as test items (optional, default 0.2);
* ``topn``: binary integer value which states if the dataset should be pre-processed for performing top-N recommendation (=1) or rating prediction (optional, by default 0).

This is an example of a valid data configuration file:

.. code-block:: json

    {
        "data_path": "datasets/ml-100k/u.data",
        "proc_path": "datasets/ml-100k/preproc",
        "seed": 98765,
        "threshold": 3.5,
        "separator": "\t",
        "u_min": 3,
        "i_min": 0,
        "heldout": 100,
        "test_prop": 0.2,
        "topn": 1
    }

The example above is a valid configuration for the `Movielens 100k dataset <https://grouplens.org/datasets/movielens/100k/>`_
where ratings less than 3.5 stars are discarded as well as users with less than 4 ratings.
The heldout set has size 100 users (100 for validation set and 100 for the test set).
The portion of ratings kept in the testinng part of each users is 20%. Top-N is the task so the
remaining positive ratings are set equal to 1.

Model configuration file
------------------------

The model configuration file defines the model and training hyper-parameters.
The `.json <https://www.json.org/json-en.html>`_ model configuration file must have the following key-value pairs:

* ``model``: dictionary (:obj:`str`: :obj:`dict`) with the values of model hyper-parameters. The name of the hyper-parameter must match the signature of the model as in :mod:`models`;
* ``train``: dictionary (:obj:`str`: :obj:`dict`) with the training parameters such as, number of epochs or the validation metrics. The name of the parameters must match the signature of the model's training method (see :mod:`models`);
* ``test``: dictionary (:obj:`str`: :obj:`dict`) with the test parameters. Up to now "metrics" is the only parameters to which a list of strings has to be associated. Metric names must follow the convention as defined in :mod:`metrics`;
* ``sampler``: dictionary (:obj:`str`: :obj:`dict`) with sampler parameters. The name of the parameters must match the signature of the sampler (see :mod:`sampler`);

.. code-block:: json

    {
        "model": {
            "beta" : 0.2,
            "anneal_steps" : 100000,
            "learning_rate": 0.001
        },
        "train": {
            "num_epochs": 200,
            "verbose": 1,
            "best_path": "chkpt_best.pth",
            "valid_metric": "ndcg@100"
        },
        "test":{
            "metrics": ["ndcg@100", "ndcg@10", "recall@20", "recall@50"]
        },
        "sampler": {
            "batch_size": 250
        }
    }

