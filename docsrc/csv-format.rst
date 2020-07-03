.. _csv-format:

Data sets CSV format
====================

Data sets file are assumed of being CSV where each row is a user-item rating.
Each row must be formed of at least two columns where the first column represents the user raw id
and the second column the raw item id.

For example:

.. code-block::
   
    user_id,item_id
    u1,i1
    u1,i2
    u2,i1
    u2,i3

The example shows a data set CSV with 4 ratings, 2 users (u1, u2) and 3 items (i1, i2, i3). The CSV
uses colon as delimiter and it has a header which must be configured correctly in the data
configuration file (see :ref:`config-format`).

In the case of data sets for rating predition, a third column must be provided containing the
rating value. For example:

.. code-block::
   
    user_id,item_id,rating
    u1,i1,3.5
    u1,i2,2
    u2,i1,4
    u2,i3,5

which is a data set with the same users as the one above, but with explicit ratings. 
