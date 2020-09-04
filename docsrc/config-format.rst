.. _config-format:

Configuration files format
==========================

In **rectorch**, the configuration of the data set pre-processing, and the models' training/test 
is performed via `.json <https://www.json.org/json-en.html>`_ files.


Data configuration file
-----------------------

The data configuration file defines how the data set must be pre-processed.
The pre-processing comprehends the reading, clean up, and the partition of the data set.
The `.json <https://www.json.org/json-en.html>`_ data configuration file must have the following key-value pairs:

* ``processing``: dictionary with the pre-processing configurations;
* ``splitting``: dictionary with the splitting configurations.

The ``processing`` options are the following:

* ``data_path``: string representing the path to the data set `.csv <https://it.wikipedia.org/wiki/Comma-separated_values>`_ file;
* ``threshold``: float cut-off value for converting explicit feedback to implicit feedback;
* ``separator``: string delimiter used in the `.csv <https://it.wikipedia.org/wiki/Comma-separated_values>`_ file;
* ``header``: number of rows of the header (if no header set to ``None`` or ``null`` if in *json*);
* ``u_min``: integer minimum number of items for a user to be kept in the data set;
* ``i_min``: integer minimum number of users for an item to be kept in the data set;

The ``splitting`` options are the following:

* ``split_type``: string in the set {``vertical``, ``horizontal``} that indicates the type of splitting. Vertical splitting means that the heldout (validation and test) set considers users that are not in the training set. Horizontal splitting instead uses part of the ratings of all users as training and the rest as validation/test set/s;
* ``sort_by``: string that indicates the column to use for sorting the ratings. If the header is missing use the column index. If no sorting is required set to ``None`` or``null`` if in *json*;
* ``seed``: integer random seed used for both the training/validation/test division as well as for shuffling the data;
* ``shuffle``:  binary integer value which states whether to shuffle the ratings or not;
* ``valid_size``: if float is considered as the portion of ratings (if horizontal split) or users (if vertical split) to consider in the validation set. In the case of vertical splitting the value can be also an integer that indicates the number of users in the validation set;
* ``test_size``: if float is considered as the portion of ratings (if horizontal split) or users (if vertical split) to consider in the test set. In the case of vertical splitting the value can be also an integer that indicates the number of users in the test set;
* ``test_prop``: (used only in the case of vertical splitting) float in the range (0,1) which represents the proportion of items of the test users that are considered as test items (optional, default 0.2).
* ``cv``: whether to split the dataset in a cross-validationn fashion. If 'cv' is provided and it is > 1 then the splitting procedure returns a list of datasets. Note that in this case when the horizontal splitting is performed the default values of both 'shuffle' and 'sort_by' are used.

This is an example of a valid data configuration file:

.. code-block:: json

    {
        "processing": {
            "data_path": "./ml-100k/u.data",
            "threshold": 3.5,
            "separator": ",",
            "header": null,
            "u_min": 3,
            "i_min": 0
        },
        "splitting": {
            "split_type": "vertical", 
            "sort_by": null, 
            "seed": 98765,
            "shuffle": 1,
            "valid_size": 100,
            "test_size": 100,
            "test_prop": 0.2,
            "cv": 1
        }
    }



The example above is a valid configuration for the `Movielens 100k dataset <https://grouplens.org/datasets/movielens/100k/>`_
where ratings less than 3.5 stars are discarded as well as users with less than 3 ratings.
The splitting type is vertical and the heldout set has size 100 users (100 for validation set and 100 for the test set).
Ratings are not sorted but shuffled. The portion of ratings kept in the testing part of each users is 20%. Top-N is the task so the
remaining positive ratings are set equal to 1. Some examples of data configuration files are
available in `GitHub <https://github.com/makgyver/rectorch/tree/master/config>`_.

Model configuration file
------------------------

The model configuration file defines the model and training hyper-parameters.
The `.json <https://www.json.org/json-en.html>`_ model configuration file must have the following key-value pairs:

* ``model``: dictionary with the values of model hyper-parameters. The name of the hyper-parameter must match the signature of the model as in :mod:`models`;
* ``train``: dictionary with the training parameters such as, number of epochs or the validation metrics. The name of the parameters must match the signature of the model's training method (see :mod:`models`);
* ``test``: dictionary with the test parameters. Up to now "metrics" is the only parameters to which a list of strings has to be associated. Metric names must follow the convention as defined in :mod:`metrics`;
* ``sampler``: dictionary with sampler parameters. The name of the parameters must match the signature of the sampler (see :mod:`sampler`);

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

Some examples of model configuration files are available in
`GitHub <https://github.com/makgyver/rectorch/tree/master/config>`_.