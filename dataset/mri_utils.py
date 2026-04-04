"""
MRI Utilities for DRIFT
========================

Orientation detection and clinical protocol constants for
through-plane MRI super-resolution.

Provides:
- Orientation-to-slice-axis mapping for NIfTI volumes
- Default thickness/gap ranges based on clinical protocols
"""


# ============================================================================
# Default Protocol Ranges (Evidence-based)
# ============================================================================

# Training range: Thr to 6.0 mm (see Sec. 4.1 in the paper)
DEFAULT_THICKNESS_RANGE = (3.0, 6.0)   # mm
DEFAULT_GAP_RANGE = (0.0, 2.0)         # mm

# Target thickness range (typically native resolution)
DEFAULT_TARGET_THICKNESS_RANGE = (0.5, 2.0)  # mm
DEFAULT_TARGET_GAP_RANGE = (0.0, 0.0)        # mm


# ============================================================================
# Orientation to Slice Axis Mapping
# ============================================================================
#
# Standard MRI orientation codes:
#   L/R: Left/Right (Sagittal plane normal)
#   A/P: Anterior/Posterior (Coronal plane normal)
#   S/I: Superior/Inferior (Axial plane normal)
#
# For each 3-letter orientation code (e.g., 'LAS'), the position of each
# letter indicates which array axis corresponds to that anatomical direction.
# ============================================================================

def get_slice_axis_from_orientation(orientation: tuple, scan_type: str = 'axial') -> int:
    """
    Get slice axis index from orientation code and scan type.

    Args:
        orientation: Tuple of 3 orientation codes, e.g., ('L', 'A', 'S')
        scan_type: 'axial' (S/I), 'coronal' (A/P), or 'sagittal' (L/R)

    Returns:
        Axis index (0, 1, or 2) corresponding to slice direction
    """
    scan_type_to_codes = {
        'axial': ('S', 'I'),
        'coronal': ('A', 'P'),
        'sagittal': ('L', 'R'),
    }

    target_codes = scan_type_to_codes.get(scan_type, ('S', 'I'))

    for axis, code in enumerate(orientation):
        if code in target_codes:
            return axis

    return 2  # Fallback: axis 2 (typically axial)


# Pre-computed mapping for common orientations
ORIENTATION_SLICE_AXIS_MAP = {
    # Standard radiological orientations
    'LAS': {'axial': 2, 'coronal': 1, 'sagittal': 0},
    'RAS': {'axial': 2, 'coronal': 1, 'sagittal': 0},
    'LPS': {'axial': 2, 'coronal': 1, 'sagittal': 0},
    'RPS': {'axial': 2, 'coronal': 1, 'sagittal': 0},
    'LAI': {'axial': 2, 'coronal': 1, 'sagittal': 0},
    'RAI': {'axial': 2, 'coronal': 1, 'sagittal': 0},
    'LPI': {'axial': 2, 'coronal': 1, 'sagittal': 0},
    'RPI': {'axial': 2, 'coronal': 1, 'sagittal': 0},
    # Permuted orientations
    'ASL': {'axial': 1, 'coronal': 0, 'sagittal': 2},
    'ASR': {'axial': 1, 'coronal': 0, 'sagittal': 2},
    'PSL': {'axial': 1, 'coronal': 0, 'sagittal': 2},
    'PSR': {'axial': 1, 'coronal': 0, 'sagittal': 2},
    'AIL': {'axial': 1, 'coronal': 0, 'sagittal': 2},
    'AIR': {'axial': 1, 'coronal': 0, 'sagittal': 2},
    'PIL': {'axial': 1, 'coronal': 0, 'sagittal': 2},
    'PIR': {'axial': 1, 'coronal': 0, 'sagittal': 2},
    'SLA': {'axial': 0, 'coronal': 2, 'sagittal': 1},
    'SRA': {'axial': 0, 'coronal': 2, 'sagittal': 1},
    'SLP': {'axial': 0, 'coronal': 2, 'sagittal': 1},
    'SRP': {'axial': 0, 'coronal': 2, 'sagittal': 1},
    'ILA': {'axial': 0, 'coronal': 2, 'sagittal': 1},
    'IRA': {'axial': 0, 'coronal': 2, 'sagittal': 1},
    'ILP': {'axial': 0, 'coronal': 2, 'sagittal': 1},
    'IRP': {'axial': 0, 'coronal': 2, 'sagittal': 1},
    'SAL': {'axial': 0, 'coronal': 1, 'sagittal': 2},
    'SAR': {'axial': 0, 'coronal': 1, 'sagittal': 2},
    'SPL': {'axial': 0, 'coronal': 1, 'sagittal': 2},
    'SPR': {'axial': 0, 'coronal': 1, 'sagittal': 2},
    'IAL': {'axial': 0, 'coronal': 1, 'sagittal': 2},
    'IAR': {'axial': 0, 'coronal': 1, 'sagittal': 2},
    'IPL': {'axial': 0, 'coronal': 1, 'sagittal': 2},
    'IPR': {'axial': 0, 'coronal': 1, 'sagittal': 2},
    'ALS': {'axial': 2, 'coronal': 0, 'sagittal': 1},
    'ARS': {'axial': 2, 'coronal': 0, 'sagittal': 1},
    'PLS': {'axial': 2, 'coronal': 0, 'sagittal': 1},
    'PRS': {'axial': 2, 'coronal': 0, 'sagittal': 1},
    'ALI': {'axial': 2, 'coronal': 0, 'sagittal': 1},
    'ARI': {'axial': 2, 'coronal': 0, 'sagittal': 1},
    'PLI': {'axial': 2, 'coronal': 0, 'sagittal': 1},
    'PRI': {'axial': 2, 'coronal': 0, 'sagittal': 1},
    'LSA': {'axial': 1, 'coronal': 2, 'sagittal': 0},
    'RSA': {'axial': 1, 'coronal': 2, 'sagittal': 0},
    'LIA': {'axial': 1, 'coronal': 2, 'sagittal': 0},
    'RIA': {'axial': 1, 'coronal': 2, 'sagittal': 0},
    'LSP': {'axial': 1, 'coronal': 2, 'sagittal': 0},
    'RSP': {'axial': 1, 'coronal': 2, 'sagittal': 0},
    'LIP': {'axial': 1, 'coronal': 2, 'sagittal': 0},
    'RIP': {'axial': 1, 'coronal': 2, 'sagittal': 0},
}

# All possible scan types for random axis selection during training
SCAN_TYPES = ['axial', 'coronal', 'sagittal']
