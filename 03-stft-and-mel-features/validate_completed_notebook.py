"""Execute every code cell in experiment.ipynb as one notebook session."""

import json
import os
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")

notebook_path = Path(__file__).with_name("experiment.ipynb")
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
namespace: dict[str, object] = {}
executed_cells = 0

for cell_index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue

    source = "".join(cell["source"])
    try:
        exec(compile(source, f"{notebook_path.name}:cell-{cell_index}", "exec"), namespace)
    except Exception as error:
        raise RuntimeError(f"Code cell {cell_index} failed") from error
    executed_cells += 1

print(f"VALIDATION PASSED: executed {executed_cells} code cells")
print(f"STFT shape: {tuple(namespace['complex_stft'].shape)}")
print(f"Mel shape: {tuple(namespace['mel_spectrogram'].shape)}")
print(f"Feature mask shape: {tuple(namespace['feature_mask'].shape)}")
