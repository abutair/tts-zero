"""Execute every code cell in the Chapter 05 notebook in one namespace."""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.stdout.reconfigure(encoding="utf-8")
notebook_path = Path(__file__).with_name("experiment.ipynb")
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
namespace: dict[str, object] = {}
executed = 0

for cell_index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    try:
        exec(compile(source, f"{notebook_path.name}:cell-{cell_index}", "exec"), namespace)
    except Exception as error:
        raise RuntimeError(f"Code cell {cell_index} failed") from error
    executed += 1

print(f"VALIDATION PASSED: executed {executed} code cells")
print(f"Encoded text shape: {tuple(namespace['x_full'].shape)}")
print(f"Prior mean shape: {tuple(namespace['m_p'].shape)}")
print(f"Prior log-scale shape: {tuple(namespace['logs_p'].shape)}")
