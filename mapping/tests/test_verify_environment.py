import builtins
import types

from mapping.pc import verify_environment


def test_neural_runtime_dependencies_are_checked_before_cuda_gate(monkeypatch, capsys):
    imported = set()
    real_import = builtins.__import__
    fake_torch = types.SimpleNamespace(
        __version__="test",
        version=types.SimpleNamespace(cuda=None),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )

    def tracking_import(name, *args, **kwargs):
        imported.add(name)
        if name == "torch":
            return fake_torch
        if name in {"gsplat", "open3d", "gtsam", "nerfstudio", "transformers", "safetensors"}:
            module = types.ModuleType(name)
            module.__version__ = "test"
            return module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    assert verify_environment.main() == 2
    assert {"transformers", "safetensors"}.issubset(imported)
    assert '"cuda_available": false' in capsys.readouterr().out
