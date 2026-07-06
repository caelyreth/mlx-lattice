Artifact API
============

``mlx_lattice.artifact`` loads and saves legacy lattice model artifacts. The
deployment entry point is :func:`mlx_lattice.artifact.load_lattice_model`, which
reads a manifest and ``safetensors`` weight file, validates the graph, and
returns an in-memory :class:`mlx_lattice.artifact.LatticeModel`.

The public package root intentionally exposes only loading, saving, and the
runtime model. JSON graph builders and registries are legacy submodule tools;
future producer work should target the MLIR lattice dialect rather than growing
the JSON graph authoring surface.

The artifact implementation has three layers:

* :mod:`mlx_lattice.artifact.io` handles the on-disk artifact directory;
* :mod:`mlx_lattice.artifact.model` executes a validated in-memory manifest;
* :mod:`mlx_lattice.artifact.registry` maps manifest operations and module
  annotations to approved public ``mlx_lattice.ops`` calls.

Module artifact helpers under :mod:`mlx_lattice.artifact.builder` still build
legacy manifests and weight dictionaries from approved sparse NN modules. They
are kept for tests, local fixtures, and compatibility with the current JSON
runner, but they are not the long-term cross-framework IR API.

The artifact runner honors manifest ``dtype_policy`` for floating dense arrays and
sparse feature matrices while preserving coordinates, integer arrays, byte
streams, and packed quantized payloads.

Artifact manifests also carry runtime compatibility metadata. The loader
accepts artifacts targeted at ``mlx-lattice`` whose version specifier matches
the installed native mlx-lattice package, and rejects incompatible runtime metadata names or version
windows before dispatching any graph node.

.. automodule:: mlx_lattice.artifact
   :members:

Artifact I/O
------------

.. automodule:: mlx_lattice.artifact.io
   :members:

Artifact model
--------------

.. automodule:: mlx_lattice.artifact.model
   :members:

Legacy internals
----------------

.. automodule:: mlx_lattice.artifact.registry
   :members:

.. automodule:: mlx_lattice.artifact.bindings
   :members:

.. automodule:: mlx_lattice.artifact.builder
   :members:
