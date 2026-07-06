
from pathlib import Path
import nbformat
from nbclient import NotebookClient

notebook_path = Path(r"E:\Thesis Code\new3.ipynb")
progress_path = Path(r"E:\Thesis Code\training_outputs\run_monitor\new3_progress.log")
progress_path.parent.mkdir(parents=True, exist_ok=True)
progress_path.write_text('', encoding='utf-8')

def log(msg):
    with progress_path.open('a', encoding='utf-8') as f:
        f.write(msg + '\n')
    print(msg, flush=True)

log(f"Notebook: {notebook_path}")
nb = nbformat.read(notebook_path.open('r', encoding='utf-8'), as_version=4)
client = NotebookClient(nb, timeout=None, kernel_name='meatlens-gpu', allow_errors=False)

try:
    client.km = client.create_kernel_manager()
    client.start_new_kernel()
    client.start_new_kernel_client()
    log('Kernel started: meatlens-gpu')
    for idx, cell in enumerate(nb.cells):
        if cell.get('cell_type') != 'code':
            continue
        source = ''.join(cell.get('source', []))
        first_line = source.strip().splitlines()[0] if source.strip() else '<empty>'
        log(f"[START] cell {idx}: {first_line[:120]}")
        client.execute_cell(cell, idx)
        nbformat.write(nb, notebook_path.open('w', encoding='utf-8'))
        log(f"[DONE] cell {idx}")
    log('Notebook execution completed successfully.')
except Exception as e:
    nbformat.write(nb, notebook_path.open('w', encoding='utf-8'))
    log(f"[ERROR] {type(e).__name__}: {e}")
    raise
finally:
    try:
        client.shutdown_kernel()
    except Exception:
        pass
