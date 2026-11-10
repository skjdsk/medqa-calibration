"""Execute the local training pilot notebook with the current Python; persist outputs per cell.

Run from repository root: python -m train.dpo.run_local_training
The notebook owns all local pilot parameters. Runner status lives in
outputs/dpo_local_pilot_execution; data and stage logs use the notebook OUTPUT.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ctypes
import json
import os
import sys
import traceback

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true', help='Refresh saved results without retraining.')
    options = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    notebook_path = repo / 'notebooks/dpo_train_local.ipynb'
    output = repo / 'outputs/dpo_local_pilot_execution'
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'notebook_runner.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()))
    notebook = None
    started = datetime.now(timezone.utc).isoformat()

    def status(state, **extra):
        value = dict(state=state, pid=os.getpid(), started_utc=started,
                     updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        tmp = output / 'run_status.json.tmp'
        tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
        os.replace(tmp, output / 'run_status.json')
        print(json.dumps(value), flush=True)

    def save():
        if notebook is not None:
            tmp = notebook_path.with_suffix('.ipynb.tmp')
            nbformat.write(notebook, tmp)
            os.replace(tmp, notebook_path)

    class SavingClient(NotebookClient):
        async def async_execute_cell(self, cell, cell_index, *args, **kwargs):
            if options.report_only and cell.cell_type == 'code':
                source = cell.source
                if not (source.startswith('from pathlib import Path') or source.startswith('SOURCE =') or source.startswith('import pandas as pd\nfrom IPython.display import display\nsummary_file')):
                    return cell
            status('running', cell_index=cell_index, cell_type=cell.cell_type)
            try:
                return await super().async_execute_cell(cell, cell_index, *args, **kwargs)
            finally:
                save()

    try:
        if os.name == 'nt':
            # Keep the system awake during execution; release on exit.
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        kernels = output / 'runtime/kernels'
        spec_dir = kernels / 'dl_final'
        spec_dir.mkdir(parents=True, exist_ok=True)
        (spec_dir / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Python (DL_Final)', 'language': 'python',
            'env': {'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'},
        }), encoding='utf-8')
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        km = KernelManager(kernel_name='dl_final',
                           kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        client = SavingClient(notebook, km=km, timeout=None, allow_errors=False,
                              resources={'metadata': {'path': str(repo)}})
        client.execute()
        status('completed', notebook=str(notebook_path))
    except BaseException as exc:
        status('failed', error=str(exc))
        traceback.print_exc()
        raise
    finally:
        save()
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
