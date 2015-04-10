from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.space import IndexSpace, VectorSpace, CompositeSpace
from pylearn2.utils import contains_nan, safe_zip
import numpy as np
from volumetric_space import Conv3DSpace
import warnings

class VolumetricDenseDesignMatrix(DenseDesignMatrix):
   
    def set_topological_view(self, V, axes=('b', 0, 1, 2, 'c')):
        """
        Sets the dataset to represent V, where V is a batch
        of topological views of examples.

        .. todo::

            Why is this parameter named 'V'?

        Parameters
        ----------
        V : ndarray
            An array containing a design matrix representation of
            training examples.
        axes : tuple, optional
            The axes ordering of the provided topo_view. Must be some
            permutation of ('b', 0, 1, 2, 'c') where 'b' indicates the axis
            indexing examples, 0, 1 and indicate the row/cols/time dimensions and
            'c' indicates the axis indexing color channels.
        """
        if len(V.shape) != len(axes):
            raise ValueError("The topological view must have exactly 5 "
                             "dimensions, corresponding to %s" % str(axes))
        assert not contains_nan(V)
        rows = V.shape[axes.index(0)]
        cols = V.shape[axes.index(1)]
        frames = V.shape[axes.index(2)]
        channels = V.shape[axes.index('c')]
        self.view_converter = Default3dViewConverter([rows, cols, frames, 
                                                      channels], axes=axes)
        self.X = self.view_converter.topo_view_to_design_mat(V)
        # self.X_topo_space stores a "default" topological space that
        # will be used only when self.iterator is called without a
        # data_specs, and with "topo=True", which is deprecated.
        self.X_topo_space = self.view_converter.topo_space
        assert not contains_nan(self.X)

        # Update data specs
        X_space = VectorSpace(dim=self.X.shape[1])
        X_source = 'features'
        if self.y is None:
            space = X_space
            source = X_source
        else:
            if self.y.ndim == 1:
                dim = 1
            else:
                dim = self.y.shape[-1]
            # This is to support old pickled models
            if getattr(self, 'y_labels', None) is not None:
                y_space = IndexSpace(dim=dim, max_labels=self.y_labels)
            elif getattr(self, 'max_labels', None) is not None:
                y_space = IndexSpace(dim=dim, max_labels=self.max_labels)
            else:
                y_space = VectorSpace(dim=dim)
            y_source = 'targets'
            space = CompositeSpace((X_space, y_space))
            source = (X_source, y_source)

        self.data_specs = (space, source)
        self.X_space = X_space
        self._iter_data_specs = (X_space, X_source)
        

class Default3dViewConverter(object):

    """
    .. todo::

        WRITEME

    Parameters
    ----------
    shape : list
      [num_rows, num_cols, num_frames, channels]
    axes : tuple
      The axis ordering to use in topological views of the data. Must be some
      permutation of ('b', 0, 1, 2, 'c'). Default: ('b', 0, 1, 2, 'c')
    """

    def __init__(self, shape, axes=('b', 0, 1, 2, 'c')):
        self.shape = shape
        self.pixels_per_channel = 1
        for dim in self.shape[:-1]:
            self.pixels_per_channel *= dim
        self.axes = axes
        self._update_topo_space()

    def view_shape(self):
        """
        .. todo::

            WRITEME
        """
        return self.shape

    def weights_view_shape(self):
        """
        .. todo::

            WRITEME
        """
        return self.shape

    def design_mat_to_topo_view(self, design_matrix):
        """
        Returns a topological view/copy of design matrix.

        Parameters
        ----------
        design_matrix: numpy.ndarray
          A design matrix with data in rows. Data is assumed to be laid out in
          memory according to the axis order ('b', 'c', 0, 1)

        returns: numpy.ndarray
          A matrix with axis order given by self.axes and batch shape given by
          self.shape (if you reordered self.shape to match self.axes, as
          self.shape is always in 'c', 0, 1 order).

          This will try to return
          a view into design_matrix if possible; otherwise it will allocate a
          new ndarray.
        """
        if len(design_matrix.shape) != 2:
            raise ValueError("design_matrix must have 2 dimensions, but shape "
                             "was %s." % str(design_matrix.shape))

        expected_row_size = np.prod(self.shape)
        if design_matrix.shape[1] != expected_row_size:
            raise ValueError("This DefaultViewConverter's self.shape = %s, "
                             "for a total size of %d, but the design_matrix's "
                             "row size was different (%d)." %
                             (str(self.shape),
                              expected_row_size,
                              design_matrix.shape[1]))

        bc01_shape = tuple([design_matrix.shape[0], ] +  # num. batches
                           # Maps the (0, 1, 'c') of self.shape to ('c', 0, 1)
                           [self.shape[i] for i in (3, 0, 1, 2)])
        topo_array_bc01 = design_matrix.reshape(bc01_shape)
        axis_order = [('b', 'c', 0, 1, 2).index(axis) for axis in self.axes]
        return topo_array_bc01.transpose(*axis_order)

    def design_mat_to_weights_view(self, X):
        """
        .. todo::

            WRITEME
        """
        rval = self.design_mat_to_topo_view(X)

        # weights view is always for display
        rval = np.transpose(rval, tuple(self.axes.index(axis)
                                        for axis in ('b', 0, 1, 2, 'c')))

        return rval

    def topo_view_to_design_mat(self, topo_array):
        """
        Returns a design matrix view/copy of topological matrix.

        Parameters
        ----------
        topo_array: numpy.ndarray
          An N-D array with axis order given by self.axes. Non-batch axes'
          dimension sizes must agree with corresponding sizes in self.shape.

        returns: numpy.ndarray
          A design matrix with data in rows. Data, is laid out in memory
          according to the default axis order ('b', 'c', 0, 1). This will
          try to return a view into topo_array if possible; otherwise it will
          allocate a new ndarray.
        """
        for shape_elem, axis in safe_zip(self.shape, (0, 1, 2, 'c')):
            if topo_array.shape[self.axes.index(axis)] != shape_elem:
                raise ValueError(
                    "topo_array's %s axis has a different size "
                    "(%d) from the corresponding size (%d) in "
                    "self.shape.\n"
                    "  self.shape:       %s (uses standard axis order: 0, 1, "
                    "'c')\n"
                    "  self.axes:        %s\n"
                    "  topo_array.shape: %s (should be in self.axes' order)")

        topo_array_bc01 = topo_array.transpose([self.axes.index(ax)
                                                for ax in ('b', 'c', 0, 1, 2)])

        return topo_array_bc01.reshape((topo_array_bc01.shape[0],
                                        np.prod(topo_array_bc01.shape[1:])))

    def get_formatted_batch(self, batch, dspace):
        """
        .. todo::

            WRITEME properly

        Reformat batch from the internal storage format into dspace.
        """
        if isinstance(dspace, VectorSpace):
            # If a VectorSpace is requested, batch should already be in that
            # space. We call np_format_as anyway, in case the batch needs to be
            # cast to dspace.dtype. This also validates the batch shape, to
            # check that it's a valid batch in dspace.
            return dspace.np_format_as(batch, dspace)
        elif isinstance(dspace, Conv3DSpace):
            # design_mat_to_topo_view will return a batch formatted
            # in a Conv3DSpace, but not necessarily the right one.
            topo_batch = self.design_mat_to_topo_view(batch)
            if self.topo_space.axes != self.axes:
                warnings.warn("It looks like %s.axes has been changed "
                              "directly, please use the set_axes() method "
                              "instead." % self.__class__.__name__)
                self._update_topo_space()

            return self.topo_space.np_format_as(topo_batch, dspace)
        else:
            raise ValueError("%s does not know how to format a batch into "
                             "%s of type %s."
                             % (self.__class__.__name__, dspace, type(dspace)))

    def __setstate__(self, d):
        """
        .. todo::

            WRITEME
        """
        # Patch old pickle files that don't have the axes attribute.
        if 'axes' not in d:
            d['axes'] = ['b', 0, 1, 2, 'c']
        self.__dict__.update(d)

        # Same for topo_space
        if 'topo_space' not in self.__dict__:
            self._update_topo_space()

    def _update_topo_space(self):
        """Update self.topo_space from self.shape and self.axes"""
        rows, cols, frames, channels = self.shape
        self.topo_space = Conv3DSpace(shape=(rows, cols, frames),
                                      num_channels=channels,
                                      axes=self.axes)

    def set_axes(self, axes):
        """
        .. todo::

            WRITEME
        """
        self.axes = axes
        self._update_topo_space()