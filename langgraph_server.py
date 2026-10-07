import importlib.util
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TARGET_FOLDER = os.getenv("LANGGRAPH_FOLDER", "Examples").strip()
DEFAULT_TARGET = "command_example.py"


def _load_graph_from_target():
    target = os.getenv("LANGGRAPH_TARGET", DEFAULT_TARGET).strip()
    target_path = (BASE_DIR / TARGET_FOLDER / target).resolve()

    if not target_path.exists():
        raise FileNotFoundError(
            f"LANGGRAPH_TARGET points to '{target}', but that file does not exist in "
            f"{BASE_DIR / TARGET_FOLDER}."
        )

    if target_path.suffix != ".py":
        raise ValueError("LANGGRAPH_TARGET must point to a Python file ending in .py")

    spec = importlib.util.spec_from_file_location("langgraph_dynamic_target", str(target_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {target_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    loaded_graph = getattr(module, "graph", None)
    if loaded_graph is None:
        raise AttributeError(
            f"Target file '{target}' does not define a top-level variable named 'graph'."
        )

    return loaded_graph


# LangGraph dev server imports this symbol.
graph = _load_graph_from_target()
