# **Implementation Plan: Add OME-TIFF Support (with dask_image + ds_factor)**

## 1. Preserve existing layout and behavior

* Keep the current **folder-per-FOV + per-channel TIFF** layout exactly as-is:

  * `base_folder/<FOV>/<channel>.tiff`
  * Existing functions must continue to work unchanged:

    * `load_channel_struct_fov`
    * `load_one_channel_fov`
    * `image_cache[fov_name]` as dict-of-channels (`str -> dask.array`)
    * mask & annotation loading
    * map mode
    * LRU cache & eviction
* OME-TIFF is a **new, additional format**, not a replacement.

---

## 2. Add layout detection (folder vs OME-TIFF)

In the top-level dataset loader (where `base_folder` is processed):

1. **Scan `base_folder`:**

   * If it contains **subdirectories** → use *existing* folder layout.
   * If it contains **`.ome.tif` / `.ome.tiff` files directly** → activate **OME-TIFF mode**.

2. Store mode on the viewer instance:

   ```python
   self._fov_mode = "folder"  # existing
   self._fov_mode = "ome-tiff"  # new
   ```

3. Build FOV list in OME mode:

   * Each `.ome.tif(f)` file is a FOV.
   * FOV name = filename without extension (used in GUI and as `fov_key`).

---

## 3. Channel metadata: use `tifffile` + OME-XML

Implement a helper function that **only reads metadata**, not pixel data:

```python
from tifffile import TiffFile
import xml.etree.ElementTree as ET

def extract_ome_channel_names(path):
    with TiffFile(path) as tif:
        xml_str = tif.ome_metadata

    root = ET.fromstring(xml_str)
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

    channels = root.findall(".//ome:Channel", ns)
    channel_names = [ch.attrib.get("Name") for ch in channels]

    return channel_names
```

* This is how OME-TIFF **channel names** are obtained.
* No pixel data is loaded here, so no memory issues.

You can extend this later to also fetch pixel sizes, etc., if needed.

---

## 4. Load OME-TIFF lazily using `dask_image.imread`

Use **`dask_image.imread`** to create a lazy Dask array for each OME-TIFF file:

```python
from dask_image.imread import imread

def open_ome_tiff_as_dask(path):
    # imread returns a lazy dask array; shape/axes depend on the file
    arr = imread(path)
    return arr
```

* This must be used instead of loading via `tifffile` into a full in-memory array.
* The array remains lazy; **no `.compute()`** should be called in the viewer code path.

---

## 5. Respect `ds_factor`: never read full-size images

UELEr already computes a **downsampling factor** `ds_factor`.

**Rule for OME-TIFF mode:**

> Never work on full-resolution slices.
> Always use spatial subsampling via `::ds_factor`.

For a given FOV:

```python
# omi_tiff: lazy dask array from dask_image.imread
# ds_factor: provided by UELer

# Example for a single channel index ch_idx
downsampled_channel = omi_tiff[ch_idx, ::ds_factor, ::ds_factor]
```

* This is the *only* way channel rasters should be sliced for OME-TIFF in the viewer.
* No reading of the full `[ch_idx, :, :]` slice without `::ds_factor`.

---

## 6. Introduce `OMEFovWrapper` to encapsulate OME logic

Create a new wrapper class that:

* Holds the **path**, **channel names**, **lazy Dask array**, and **ds_factor**.
* Provides a **dict-like interface** compatible with existing `image_cache[fov_name][channel_name]`.

Example sketch:

```python
class OMEFovWrapper:
    def __init__(self, path, ds_factor):
        self.path = path
        self.ds_factor = ds_factor

        # Lazy image data
        self._dask_arr = open_ome_tiff_as_dask(path)

        # Metadata
        self.channel_names = extract_ome_channel_names(path)
        self._name_to_index = {
            name: idx for idx, name in enumerate(self.channel_names)
        }

    def get_channel_names(self):
        return self.channel_names

    def __getitem__(self, channel_name):
        idx = self._name_to_index[channel_name]
        # Critically: only downsampled slice is exposed
        return self._dask_arr[idx, ::self.ds_factor, ::self.ds_factor]
```

* From outside, this behaves like a `dict[channel_name] -> dask.array`.
* Internally, it guarantees:

  * **`dask_image.imread`** is used
  * Only **downsampled** views `::ds_factor` are accessed
  * No full-size images are read into memory

---

## 7. Integrate `OMEFovWrapper` into `load_fov`

In `ImageMaskViewer.load_fov` (or equivalent):

1. If `self._fov_mode == "folder"`:

   * Use existing path unchanged.

2. If `self._fov_mode == "ome-tiff"`:

   * Compute or obtain `ds_factor` from UELer (already available in your pipeline).
   * Instantiate wrapper for that FOV:

     ```python
     wrapper = OMEFovWrapper(path_to_ome_tiff, ds_factor=self.ds_factor)
     image_cache[fov_name] = wrapper
     ```
   * Initialize `channel_max_values` by sampling the **downsampled** arrays:

     ```python
     for ch_name in wrapper.get_channel_names():
         arr = wrapper[ch_name]             # dask (downsampled)
         # trigger reduction on a downsampled view only:
         ch_max = arr.max().compute()
         channel_max_values[ch_name] = ch_max
     ```

* This preserves:

  * Lazy loading
  * No full-resolution read
  * Existing consumer code that expects `image_cache[fov][channel_name]` to return a dask array.

---

## 8. Keep rendering, masks, and annotations unchanged

* Rendering pipeline should see no difference:

  * `render_image` / `_compose_fov_image` can keep doing:

    ```python
    img = image_cache[fov_name][channel_name]
    ```
* Because `OMEFovWrapper.__getitem__` returns a 2D dask array (already downsampled), it slots into the same code path.
* Mask & annotation loading:

  * Continue to use existing loaders and caches (`mask_cache`, `annotation_cache`, etc.).
  * They remain layout-agnostic, so no OME-specific logic is required there.

---

## 9. Cache and eviction

* In OME-TIFF mode:

  * `image_cache[fov_name]` holds a single `OMEFovWrapper`.
* LRU eviction:

  * When a FOV is evicted, the wrapper is discarded as a single object.
  * No special handling is needed, since the wrapper only holds lazy dask arrays and channel metadata.

---

## 10. Summary of key constraints (for Copilot to respect)

* **Do not** modify the folder-based layout behavior.
* **Add** an OME-TIFF mode with:

  * Detection based on `.ome.tif(f)` files in `base_folder`.
  * Channel names extracted from `tifffile.TiffFile(...).ome_metadata` via XML:

    ```python
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    channels = root.findall(".//ome:Channel", ns)
    channel_names = [ch.attrib.get("Name") for ch in channels]
    ```
  * Image loading using `dask_image.imread.imread`.
  * **Never** reading full-resolution arrays; always using:

    ```python
    omi_tiff[ch_idx, ::ds_factor, ::ds_factor]
    ```
* Wrap OME-TIFF FOVs in an `OMEFovWrapper` that behaves like a `dict[channel_name] -> dask.array`.
