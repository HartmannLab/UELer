"""Unit tests for the BioImage Archive streaming loader (issue #110).

Network access is fully mocked; these tests exercise study resolution,
directory-index parsing, layout classification, the folder-mode channel/mask
plumbing (reusing the local loaders) and the ``run_viewer_bia`` wiring.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ueler.bia_loader as bia_loader
from ueler.bia_loader import (
    BIADataSource,
    BIALayout,
    BIAStudyIndex,
    _auto_detect_layout,
    _layout_from_descriptor,
    _parse_autoindex,
    _to_https,
)
from ueler.runner import run_viewer_bia


# --- fakes -----------------------------------------------------------------
class FakeIndex:
    """In-memory stand-in for :class:`BIAStudyIndex` (no network)."""

    def __init__(self, tree, base_url="https://example.org/study"):
        self.base_url = base_url
        self.tree = tree  # rel_path -> list[(name, is_dir)]

    def url_for(self, rel_path):
        return f"{self.base_url}/{rel_path.strip('/')}"

    def list_dir(self, rel_path=""):
        return self.tree.get(rel_path, [])

    def list_subdirs(self, rel_path=""):
        return sorted(n for n, is_dir in self.list_dir(rel_path) if is_dir)

    def list_files(self, rel_path=""):
        return sorted(n for n, is_dir in self.list_dir(rel_path) if not is_dir)


def _folder_tree():
    """A trimmed S-BIAD2557-style folder-per-FOV MIBI tree."""
    return {
        "Files/img": [("fov1", True), ("fov2", True)],
        "Files/img/fov1": [("CD3.tiff", False), ("CD4.tiff", False)],
        "Files/img/fov2": [("CD3.tiff", False)],
        "Files/masks": [
            ("fov1_cleaned_mask.tiff", False),
            ("fov2_cleaned_mask.tiff", False),
        ],
    }


def _folder_descriptor():
    return {
        "mode": "folder",
        "base": "Files/img",
        "mask_dir": "Files/masks",
        "mask_glob": "{fov}_*.tiff",
    }


def _make_folder_source(cache_dir):
    with patch.object(BIAStudyIndex, "from_source", return_value=FakeIndex(_folder_tree())):
        ds = BIADataSource(
            "https://example.org/study",
            cache_dir=cache_dir,
            descriptor=_folder_descriptor(),
        )
    ds.index = FakeIndex(_folder_tree())
    return ds


# --- URL / parsing ---------------------------------------------------------
class HelperTest(unittest.TestCase):
    def test_to_https_rewrites_ftp(self):
        self.assertEqual(
            _to_https("ftp://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/557/S-BIAD2557"),
            "https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/557/S-BIAD2557",
        )
        self.assertEqual(_to_https("https://x/y"), "https://x/y")

    def test_parse_autoindex_filters_sort_and_parent_links(self):
        html = (
            '<a href="?C=N;O=D">Name</a>'
            '<a href="/pub/parent/">Parent</a>'
            '<a href="image_data/">image_data/</a>'
            '<a href="CD3.tiff">CD3.tiff</a>'
            '<a href="https://other/host">external</a>'
        )
        entries = _parse_autoindex(html)
        self.assertIn(("image_data", True), entries)
        self.assertIn(("CD3.tiff", False), entries)
        self.assertTrue(all(name not in ("", "..") for name, _ in entries))
        self.assertEqual(len(entries), 2)


class StudyResolutionTest(unittest.TestCase):
    def test_from_source_accepts_direct_url(self):
        index = BIAStudyIndex.from_source("https://ftp.ebi.ac.uk/x/y/")
        self.assertEqual(index.base_url, "https://ftp.ebi.ac.uk/x/y")
        self.assertIsNone(index.accession)

    def test_from_source_resolves_accession_via_api(self):
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "ftpLink": "ftp://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/557/S-BIAD2557"
        }
        fake_resp.raise_for_status.return_value = None
        fake_requests = SimpleNamespace(get=MagicMock(return_value=fake_resp))

        with patch.object(bia_loader, "_ensure_requests", return_value=fake_requests):
            index = BIAStudyIndex.from_source("S-BIAD2557")

        self.assertEqual(
            index.base_url,
            "https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/557/S-BIAD2557",
        )
        self.assertEqual(index.accession, "S-BIAD2557")
        fake_requests.get.assert_called_once()

    def test_from_source_rejects_garbage(self):
        with self.assertRaises(ValueError):
            BIAStudyIndex.from_source("not-an-accession")


# --- layout classification -------------------------------------------------
class LayoutTest(unittest.TestCase):
    def test_descriptor_folder_mode(self):
        layout = _layout_from_descriptor(_folder_descriptor())
        self.assertEqual(layout.mode, "folder")
        self.assertEqual(layout.base, "Files/img")
        self.assertEqual(layout.mask_dir, "Files/masks")

    def test_descriptor_ome_mode_splits_glob(self):
        layout = _layout_from_descriptor(
            {"mode": "ome-tiff", "fov_glob": "Files/images/*.ome.tiff"}
        )
        self.assertEqual(layout.mode, "ome-tiff")
        self.assertEqual(layout.fov_dir, "Files/images")
        self.assertEqual(layout.fov_glob, "*.ome.tiff")

    def test_auto_detect_folder_layout(self):
        # A crawlable tree (with the intermediate directory levels a real
        # Apache autoindex would expose).
        tree = {
            "": [("Files", True)],
            "Files": [("img", True), ("masks", True)],
            "Files/img": [("fov1", True), ("fov2", True)],
            "Files/img/fov1": [("CD3.tiff", False), ("CD4.tiff", False)],
            "Files/img/fov2": [("CD3.tiff", False)],
            "Files/masks": [("fov1_cleaned_mask.tiff", False)],
        }
        layout = _auto_detect_layout(FakeIndex(tree))
        self.assertEqual(layout.mode, "folder")
        self.assertEqual(layout.base, "Files/img")

    def test_auto_detect_ome_layout(self):
        tree = {"": [("Files", True)], "Files": [("fovA.ome.tiff", False)]}
        layout = _auto_detect_layout(FakeIndex(tree))
        self.assertEqual(layout.mode, "ome-tiff")
        self.assertEqual(layout.fov_dir, "Files")

    def test_auto_detect_raises_when_unrecognised(self):
        tree = {"": [("readme.txt", False), ("data.csv", False)]}
        with self.assertRaises(ValueError):
            _auto_detect_layout(FakeIndex(tree))


# --- data source (folder mode) --------------------------------------------
class FolderDataSourceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.cache = self._tmp.name
        self.ds = _make_folder_source(self.cache)

    def tearDown(self):
        self._tmp.cleanup()

    def test_fov_mode_and_list(self):
        self.assertEqual(self.ds.fov_mode, "folder")
        self.assertEqual(self.ds.list_fovs(), ["fov1", "fov2"])

    def test_channels_for_maps_stems_to_urls(self):
        channels = self.ds._channels_for("fov1")
        self.assertEqual(sorted(channels), ["CD3", "CD4"])
        self.assertTrue(channels["CD3"].endswith("Files/img/fov1/CD3.tiff"))

    def test_open_fov_returns_channel_stub_dict(self):
        obj = self.ds.open_fov("fov1", ds_factor=1)
        self.assertEqual(obj, {"CD3": None, "CD4": None})

    def test_load_channel_downloads_then_reads_via_local_loader(self):
        with patch.object(bia_loader, "_download") as dl, \
                patch.object(bia_loader, "load_one_channel_fov", return_value="ARR") as read:
            result = self.ds.load_channel("fov1", "CD3", {}, compute_stats=False)

        self.assertEqual(result, "ARR")
        dl.assert_called_once()
        called_url, called_dest = dl.call_args[0]
        self.assertTrue(called_url.endswith("Files/img/fov1/CD3.tiff"))
        self.assertTrue(str(called_dest).endswith("channels/fov1/CD3.tiff"))
        # local loader is pointed at the channel cache root, not base_folder
        args, kwargs = read.call_args
        self.assertEqual(args[0], "fov1")
        self.assertTrue(args[1].endswith("channels"))
        self.assertEqual(args[3], {"CD3"})

    def test_load_channel_unknown_returns_none(self):
        self.assertIsNone(self.ds.load_channel("fov1", "NOPE", {}))

    def test_has_masks_and_prefetch(self):
        self.assertTrue(self.ds.has_masks)
        with patch.object(bia_loader, "_download") as dl:
            self.ds.prefetch_masks("fov1")
        dl.assert_called_once()
        url, dest = dl.call_args[0]
        self.assertTrue(url.endswith("Files/masks/fov1_cleaned_mask.tiff"))
        self.assertTrue(str(dest).endswith("masks_flat/fov1_cleaned_mask.tiff"))

    def test_prefetch_masks_only_matching_fov(self):
        with patch.object(bia_loader, "_download") as dl:
            self.ds.prefetch_masks("fov2")
        self.assertEqual(dl.call_count, 1)
        url, _ = dl.call_args[0]
        self.assertIn("fov2_cleaned_mask.tiff", url)


# --- data source (ome mode stream vs cache) --------------------------------
class OmeDataSourceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        tree = {"Files": [("fovA.ome.tiff", False)]}
        with patch.object(BIAStudyIndex, "from_source", return_value=FakeIndex(tree)):
            self.ds = BIADataSource(
                "https://example.org/study",
                cache_dir=self._tmp.name,
                descriptor={"mode": "ome-tiff", "fov_glob": "Files/*.ome.tiff"},
            )
        self.ds.index = FakeIndex(tree)

    def tearDown(self):
        self._tmp.cleanup()

    def test_list_fovs_strips_ome_suffix(self):
        self.assertEqual(self.ds.list_fovs(), ["fovA"])

    def test_open_fov_streams_when_pyramidal(self):
        with patch.object(bia_loader, "_is_streamable", return_value=True), \
                patch.object(bia_loader, "OMEFovWrapper") as wrapper, \
                patch.object(bia_loader, "_download") as dl:
            self.ds.open_fov("fovA", ds_factor=2)
        dl.assert_not_called()
        _, kwargs = wrapper.call_args
        self.assertIs(kwargs["opener"], bia_loader._open_remote)

    def test_open_fov_caches_when_not_streamable(self):
        with patch.object(bia_loader, "_is_streamable", return_value=False), \
                patch.object(bia_loader, "OMEFovWrapper") as wrapper, \
                patch.object(bia_loader, "_download", return_value=Path("/x/fovA.ome.tiff")) as dl:
            self.ds.open_fov("fovA", ds_factor=2)
        dl.assert_called_once()
        _, kwargs = wrapper.call_args
        self.assertNotIn("opener", kwargs)


# --- runner wiring ---------------------------------------------------------
class RunViewerBiaTest(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.workspace = Path(self._tmp.name) / "ws"

    def tearDown(self):
        self._tmp.cleanup()

    def test_run_viewer_bia_wires_data_source_and_workspace(self):
        made = {}

        def fake_ds_factory(source, *, cache_dir, descriptor):
            made["source"] = source
            made["cache_dir"] = cache_dir
            made["descriptor"] = descriptor
            return SimpleNamespace(tag="ds")

        def fake_viewer_factory(base_folder, *, data_source=None, **kwargs):
            made["base"] = base_folder
            made["data_source"] = data_source
            made["kwargs"] = kwargs
            return MagicMock()

        with patch("ueler.runner.ensure_aliases_loaded"), \
                patch("ueler.runner._normalise_directory") as normalise, \
                patch("ueler.runner._load_display_helpers") as load_display:
            load_display.return_value = (MagicMock(), MagicMock())
            run_viewer_bia(
                "S-BIAD2557",
                descriptor=_folder_descriptor(),
                local_dir=self.workspace,
                viewer_factory=fake_viewer_factory,
                data_source_factory=fake_ds_factory,
            )

        normalise.assert_not_called()
        self.assertEqual(made["source"], "S-BIAD2557")
        self.assertEqual(made["base"], str(self.workspace))
        self.assertEqual(made["data_source"].tag, "ds")
        self.assertEqual(made["cache_dir"], str(self.workspace / "cache"))
        self.assertTrue((self.workspace / "cache").is_dir())

    def test_run_viewer_bia_loads_descriptor_from_json_file(self):
        import json

        descriptor_path = Path(self._tmp.name) / "desc.json"
        descriptor_path.write_text(json.dumps(_folder_descriptor()), encoding="utf-8")

        seen = {}

        def fake_ds_factory(source, *, cache_dir, descriptor):
            seen["descriptor"] = descriptor
            return SimpleNamespace()

        with patch("ueler.runner.ensure_aliases_loaded"), \
                patch("ueler.runner._load_display_helpers") as load_display:
            load_display.return_value = (MagicMock(), MagicMock())
            run_viewer_bia(
                "https://example.org/study",
                descriptor=descriptor_path,
                local_dir=self.workspace,
                viewer_factory=lambda base, *, data_source=None, **k: MagicMock(),
                data_source_factory=fake_ds_factory,
                auto_display=False,
                after_plugins=False,
            )

        self.assertEqual(seen["descriptor"], _folder_descriptor())


if __name__ == "__main__":
    unittest.main()
