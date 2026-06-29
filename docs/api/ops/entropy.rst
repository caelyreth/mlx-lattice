Entropy coding
==============

Entropy helpers expose normalized-CDF range coding and probability-row rANS
coding through the native extension. They operate on dense MLX arrays rather
than sparse tensors.

.. warning::

   Entropy coding helpers are provisional in this release line. The Python
   signatures are public, but encoded byte-stream compatibility is not yet a
   stable cross-version storage contract. Use them for experiments and
   controlled pipelines, not long-lived file formats.

Probability inputs are row-major ``(N, S)`` arrays where ``S`` is the symbol
alphabet size. Symbol inputs are one-dimensional integer rows. Encoders return
Python ``bytes``; decoders reconstruct an MLX integer array using the model
shape implied by the probability or CDF rows.

Related pages
-------------

* Project caveats and boundaries: :doc:`../../project/caveats`
* Native diagnostics: :doc:`../native`

.. automodule:: mlx_lattice.ops.entropy
   :members:
