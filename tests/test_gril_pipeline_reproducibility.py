from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

PIPELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents/skills/gril-calib-validation/scripts/gril_pipeline.py"
)
PIPELINE_SPEC = importlib.util.spec_from_file_location(
    "gril_validation_pipeline", PIPELINE_PATH
)
assert PIPELINE_SPEC is not None and PIPELINE_SPEC.loader is not None
pipeline = importlib.util.module_from_spec(PIPELINE_SPEC)
PIPELINE_SPEC.loader.exec_module(pipeline)


class ReferenceSourceReproducibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.resource_dir = self.root / "skill/resources"
        self.resource_dir.mkdir(parents=True)
        (self.repository / "config").mkdir()
        (self.repository / "source.txt").write_text("baseline\n")
        (self.repository / "config/velodyne16.yaml").write_text("old config\n")
        (self.resource_dir / "vanjeelidar16.yaml").write_text("effective config\n")
        self.patch = self.root / "source.patch"
        self.patch.write_text(
            "--- a/source.txt\n"
            "+++ b/source.txt\n"
            "@@ -1 +1 @@\n"
            "-baseline\n"
            "+patched\n"
        )
        self.git(["init", "-q"])
        self.git(["add", "."])
        self.git(
            [
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                "base",
            ]
        )
        self.revision = self.git(["rev-parse", "HEAD"], text=True).strip()
        pipeline.SKILL_DIR = self.root / "skill"
        pipeline.PINNED_REVISION = self.revision
        pipeline.REFERENCE_PATCHES = (self.patch,)
        self.git(["apply", str(self.patch)])
        (self.repository / "config/velodyne16.yaml").write_text("effective config\n")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def git(self, arguments: list[str], *, text: bool = False):
        return subprocess.check_output(
            ["git", "-C", str(self.repository), *arguments], text=text
        )

    def test_exact_patches_and_effective_config_are_accepted(self) -> None:
        evidence = pipeline.verify_reference_source(self.repository)
        self.assertEqual(evidence["revision"], self.revision)
        self.assertEqual(len(evidence["worktree_diff_sha256"]), 64)
        self.assertEqual(len(evidence["config_sha256"]), 64)

    def test_unrecorded_tracked_change_is_rejected(self) -> None:
        (self.repository / "source.txt").write_text("unexpected\n")
        with self.assertRaisesRegex(RuntimeError, "unrecorded tracked changes"):
            pipeline.verify_reference_source(self.repository)

    def test_unrecorded_untracked_source_is_rejected(self) -> None:
        (self.repository / "src").mkdir()
        (self.repository / "src/extra.cpp").write_text("untracked\n")
        with self.assertRaisesRegex(RuntimeError, "untracked files"):
            pipeline.verify_reference_source(self.repository)

    def test_catkin_tree_hash_changes_with_build_output(self) -> None:
        build_output = self.root / "catkin/devel"
        build_output.mkdir(parents=True)
        artifact = build_output / "lib/gril_calib"
        artifact.parent.mkdir()
        artifact.write_bytes(b"first build")
        initial_hash = pipeline.sha256_tree(build_output)
        artifact.write_bytes(b"changed build")
        self.assertNotEqual(initial_hash, pipeline.sha256_tree(build_output))
