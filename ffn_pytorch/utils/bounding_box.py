"""BoundingBox compatibility layer.

Tries to import from connectomics.common; falls back to ffn.utils.bounding_box
with a to_slice3d shim.
"""

try:
    from connectomics.common.bounding_box import BoundingBox
except ImportError:
    from ffn.utils.bounding_box import BoundingBox as _BBox

    class BoundingBox(_BBox):
        """BoundingBox with to_slice3d compatibility."""

        def to_slice3d(self):
            """Returns slice in C-order (ZYX), same as to_slice."""
            return self.to_slice()

        def intersection(self, other):
            from ffn.utils import bounding_box as bb_mod
            return bb_mod.intersection(self, other)
