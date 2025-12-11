
Systematics Uncertainties
=========================

NMMA currently uses ``--error-budget`` to specify the constant systematic uncertainties to be added to the likelihood quadrature.

However, it is now possible to use systematic error (:math:`\sigma_{sys}`) prior  in form of a freely sampled parameter, time dependent and/or filter dependent systematic error. This can done by specifying the file path using the ``--systematics-file`` in ``lightcurve-analysis`` command.

For more information on systematics error, please refer to the `paper <https://arxiv.org/abs/2410.21978>`__.

The following are the examples of the systematics file:

Example 1: Freely sampled (time independent) systematic error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this case the systematic error is freely sampled and is not dependent on time or filter.

.. code:: yaml

  config:
    withTime:
      value: false
      filters:
        - null
      time_nodes: 4
      type: Uniform
      minimum: 0
      maximum: 2
    withoutTime:
      value: true
      type: Uniform
      minimum: 0
      maximum: 2

Example 2: Time dependent systematic error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this configuration, a single systematic error is applied across all filters.

.. code:: yaml

    config:
      withTime:
        value: true
        filters:
          - null
        time_nodes: 4
        type: Uniform
        minimum: 0
        maximum: 2
      withoutTime:
        value: false
        type: Uniform
        minimum: 0
        maximum: 2


Example 3: Time and filter dependent systematic error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this configuration, the ``sdssu`` and ``ztfr`` filters are sampled together for systematic errors, while the ``2massks`` filter is sampled independently. All other filters are grouped and sampled together

.. code:: yaml

    config:
      withTime:
        value: true
        filters:
          - [sdssu, ztfr]
          - [2massks]
          - null
        time_nodes: 4
        type: Uniform
        minimum: 0
        maximum: 2
      withoutTime:
        value: false
        type: Uniform
        minimum: 0
        maximum: 2


Distribution types
==================

Distribution can be of any of the ``analytical`` prior from `bilby <https://git.ligo.org/lscsoft/bilby>`__.
Please refer to bilby documentation for more information on the available distribution type and their usage. Only positional arguments are required for any of the distrbutions.
