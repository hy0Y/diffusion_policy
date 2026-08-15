import ast
from pathlib import Path


FORBIDDEN_ROOTS = {
    "diffusion_policy",
    "jump_module",
    "gr00t",
    "diffusers",
    "wandb",
    "sem_bif_experiment_runtime",
}


def test_canonical_package_does_not_import_policy_hosts():
    root = Path(__file__).resolve().parents[1] / "robocasa_milestones"
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", maxsplit=1)[0] in FORBIDDEN_ROOTS:
                    violations.append((str(path.relative_to(root)), name))
    assert violations == []
