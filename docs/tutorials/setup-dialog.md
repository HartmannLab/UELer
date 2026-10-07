# Guided Setup

The first time you open a dataset, UELer puts a short dialog in front of the viewer and asks for the handful of settings that decide whether what you see is correct. It exists because these settings are easy to miss and expensive to get wrong: a wrong pixel size silently mis-scales the scale bar in every figure you export, and a wrong column name makes the viewer find no cells at all — with nothing in the log to say why.

You can always dismiss it. Everything it asks for also lives in the left panel under **Advanced Settings**, and the dialog edits those same fields rather than copies of them.

---

## What it asks, and when

The steps are built from what you actually loaded, so you are never asked about data you do not have.

| Step | Appears when | Fields |
|---|---|---|
| **Main viewer settings** | always | **Cache Size:**, **Pixel Size (nm):**, **Downsample** |
| **Mask mapping** | you passed a masks folder | **Mask key:** |
| **Cell table mapping** | a cell table is loaded | **X key:**, **Y key:**, **Label key:**, **Fov key:** |

**Next** moves on, **Back** returns, and the last step's button reads **Done**. **Skip setup** dismisses the whole dialog, not just the step you are on.

!!! warning "Pixel Size (nm) is the one to read twice"
    It defaults to **390 nm**, the MIBI detector pitch, and it is applied to every dataset without comment. If your instrument is something else, every scale bar UELer draws — on screen and in every batch-exported image — is wrong by that ratio. This is the kind of error that surfaces when a figure is already in review, which is why the dialog names it instead of leaving it in an accordion to be found.

---

## The key fields are suggestion lists, not menus

The five key fields — **X key:**, **Y key:**, **Label key:**, **Mask key:**, **Fov key:** — are the names that link your cell table and your masks to the images. Each one is a text field with a list attached:

- **The list is built from your own data.** The four cell-table keys offer the columns your table actually has (**X key:** and **Y key:** offer the numeric ones); **Mask key:** offers the mask suffixes found in your masks folder. So in the normal case you pick rather than spell, and a typo is not a way to fail.
- **You can still type a name that is not in the list.** This matters more than it sounds. Discovery can come up short — a masks folder on storage too slow to finish scanning, a column whose name no heuristic would guess — and a field that only offered what it had found would leave you with one wrong value and no way to correct it. Typing always works.
- **UELer pre-fills what it recognises.** If your table uses `x`/`y` rather than `centroid-1`/`centroid-0`, or `cell_label` rather than `label`, the right column is already selected when the dialog opens, and the step is a confirmation rather than a decision.
- **It never silently changes an answer you gave.** If you set a key and it is still a real column, it stays set. If it is *not* a column of the table you just loaded, it is kept and shown anyway rather than swapped for a plausible-looking neighbour — a visibly wrong key is something you can fix, a silently retargeted one is not.

!!! tip "If the viewer shows no cells at all"
    Check these four first. The overlay is empty both when the keys are wrong and when the dataset genuinely has nothing to show, and the two look identical. Open **Advanced Settings → Data mapping** and compare each key against your table's actual column names.

---

## What UELer remembers

Answered steps are recorded per dataset, in `setup_dialog.json` inside the dataset's `.UELer` settings folder. The record is **per step**, not per session, which is what makes the two ways of loading a cell table behave sensibly:

```python
viewer = ueler.run_viewer(base_folder, masks_folder=masks_folder)   # asks: viewer settings, mask key
load_cell_table(viewer, cell_table=table)                           # asks: the four columns only
```

Loading the table later in the same notebook asks only about the new columns instead of repeating the whole wizard. Closing counts as answered whether you pressed **Done** or **Skip setup** — a question you deliberately dismissed should not come back on every load.

**To see the dialog again**, delete `setup_dialog.json` from the dataset's `.UELer` folder and reload. You rarely need to: every field it asks for stays editable under **Advanced Settings** for the life of the session.

!!! note "A read-only settings folder costs you a repeated dialog, not a failed session"
    If UELer cannot write the record — a dataset folder you only have read access to — the dialog simply asks again next time. Nothing else is affected.

---

## Where these settings live afterwards

Everything above is in the left panel, in the **Advanced Settings** accordion, split across two tabs:

- **Data mapping** — the five key fields.
- **Advanced Settings** — **Cache Size:**, **Pixel Size (nm):**, **Downsample**.

See [User Interface](user-interface.md#advanced-settings) for the control-by-control map, and [Display Settings](display-settings.md) for the reasoning behind each value.
