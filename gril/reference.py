"""Identity of the frozen ROS-GRIL migration reference."""

from __future__ import annotations

REFERENCE = {
    "name": "gril_ros_reference",
    "repository": "https://github.com/Taeyoung96/GRIL-Calib.git",
    "revision": "c09b01a05ec83bc0a361941acf897109aaecf0a6",
    "source_archive_sha256": (
        "f246d579953f145380327c4add2414b7431bb8ab74cf96c0bd4d27bd063487d9"
    ),
    "validation_patch": (
        ".agents/skills/gril-calib-validation/patches/gril-validation.patch"
    ),
    "validation_patch_sha256": (
        "db07052e910eda485db2f6b10f3fb745d868352d24f1051a1bc738a5a0b6c15b"
    ),
    "batch_trace_patch": "gril/reference_patches/gril-batch-trace-v1.patch",
    "batch_trace_patch_sha256": (
        "a89a22f22c6690b7e5f037664e3d0380794d25b23d3ec4968381fb7d1e12e155"
    ),
    "preprocess_trace_patch": ("gril/reference_patches/gril-preprocess-trace-v1.patch"),
    "preprocess_trace_patch_sha256": (
        "5af0cb5b7b201be76c8f4c63c94c8d62d5e812bd69c3583108d70a817f762a25"
    ),
    "frontend_cv_trace_patch": (
        "gril/reference_patches/gril-frontend-cv-trace-v1.patch"
    ),
    "frontend_cv_trace_patch_sha256": (
        "bd0109efc61c59008461d90b9824de78dc90b858f10980483adbf8f1461a38e2"
    ),
    "ground_trace_patch": "gril/reference_patches/gril-ground-trace-v1.patch",
    "ground_trace_patch_sha256": (
        "c935f32659c630ba92fb3ba6f72d0f076f67c1d9cb48c1fc5582830f9cfd3930"
    ),
    "full_frontend_reference_trace_patch": (
        "gril/reference_patches/gril-full-frontend-reference-trace-v2.patch"
    ),
    "full_frontend_reference_trace_patch_sha256": (
        "e12357469144a9b37cfbf34432857e54f578452514eb7c850e6613d48f511148"
    ),
    "deterministic_frontend": True,
    "runtime": "ROS Noetic reference container with optional trace export",
    "release_role": "migration_reference_only",
    "license_review": {
        "status": "required_before_distribution",
        "reason": (
            "package.xml declares BSD, while the source tree includes the GPLv2 "
            "LI-Init license and describes GRIL as derived from LI-Init"
        ),
    },
}
