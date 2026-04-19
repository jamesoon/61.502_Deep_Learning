"""Execute specific cells of the MCAT evaluation notebook to refresh outputs.
Only runs cells that read CSVs and generate visualizations (skips slow cells)."""
import nbformat
from nbclient import NotebookClient

nb = nbformat.read("MCAT evaluation.ipynb", as_version=4)

# Cell map:
#  0: markdown intro
#  1: code - dataset builder (SLOW - rebuilds from DOCX, SKIP)
#  2: markdown - local server setup
#  3: code - evaluation config + functions (NEEDED - defines helpers)
#  4: code - model resolution (needs server, SKIP)
#  5: markdown - eval run description
#  6: code - evaluation run (VERY SLOW, SKIP)
#  7: markdown - performance comparison intro
#  8: code - performance comparison table + token viz (RUN)
#  9: markdown - visualization intro
# 10: code - main visualization cell (RUN)

# Only execute cells 3, 8, 10
cells_to_execute = {3, 8, 10}
cells_to_skip_with_msg = {1, 4, 6}

print(f"Notebook has {len(nb.cells)} cells")
for i, cell in enumerate(nb.cells):
    ct = cell.cell_type
    preview = cell.source[:80].replace('\n', ' ') if cell.source else ''
    tag = " [EXEC]" if i in cells_to_execute else (" [SKIP]" if i in cells_to_skip_with_msg else "")
    print(f"  Cell {i} ({ct}): {preview}...{tag}")

# Clear all code cell outputs
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None

# Replace non-executed code cells with pass statements temporarily
saved_sources = {}
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and i not in cells_to_execute:
        saved_sources[i] = cell.source
        if i in cells_to_skip_with_msg:
            cell.source = f"print('Cell {i} skipped — run separately')"
        else:
            cell.source = "pass"

print("\nExecuting notebook (cells 3, 8, 10 only)...")
client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()
print("Execution complete!")

# Restore all original cell sources
for idx, source in saved_sources.items():
    nb.cells[idx].source = source

nbformat.write(nb, "MCAT evaluation.ipynb")
print("Notebook saved with fresh outputs for all 7 test sets!")
