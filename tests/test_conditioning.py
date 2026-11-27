"""Regression tests for saved checkpoints and the complete conditioning path.

Run from the repository root: python -m unittest discover -s tests -v
The NPZ fixture contains nonzero weights and outputs captured before the fixes.
"""
import ast
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

from scripts.inference.inference_loadiff import (
    build_user_conditioned, load_model_from_ckpt, run_sampling,
    main,
)
from src.helpers.conditioning import conditioning_class_count, complete_unknown_labels
from src.helpers.loadiff_inference import build_user_labels
from src.loadit.diffusion import create_diffusion
from src.loadit.models import DiT, LabelEmbedder, MultiLabelEmbedder

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests/fixtures/legacy_conditioning.npz"
torch.set_num_threads(1)


def small_model(**kwargs):
    return DiT(input_size=(4, 2), patch_size=(1, 2), hidden_size=16, depth=1,
               num_heads=8, n_exo_var=4, **kwargs)


def training_config(num_slots=3, multilabels=True, temperature=False, **extra):
    names = ["cooker", "dishwasher", "water_heater"][:num_slots]
    return OmegaConf.create({
        "data": {"dataset": "smach", "bool_col_names": names, "value_scale": 10000},
        "ditmodelargs": {"input_size": [4, 2], "patch_size": [1, 2], "hidden_size": 16,
                         "depth": 1, "in_channels": 1, "n_exo_var": 4,
                         "temperature": temperature, "multilabels": multilabels, **extra},
        "training": {"diffusion_steps": 100},
        "valid": {"gen_sample_start_date": "2021-01-01", "gen_sample_days": 4},
    })


def checkpoint_from_fixture(directory, case, *, ema=False):
    config = training_config(0 if case == "unconditioned" else 1 if case == "binary_legacy" else 3,
                             multilabels=case != "binary_legacy", temperature=case.endswith("temperature"))
    with np.load(FIXTURES) as fixture:
        state = {k.split("/state/", 1)[1]: torch.from_numpy(fixture[k].copy())
                 for k in fixture.files if k.startswith(f"{case}/state/")}
    path = Path(directory) / f"{case}.pt"
    torch.save({"args": config, "ema" if ema else "model": state}, path)
    return path


class CheckpointCompatibility(unittest.TestCase):
    def test_original_checkpoint_predictions_are_unchanged(self):
        for case in ["ternary", "ternary_temperature", "unconditioned", "binary_legacy", "multi_hot_legacy"]:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory, np.load(FIXTURES) as fixture:
                path = checkpoint_from_fixture(directory, case)
                model, _, _ = load_model_from_ckpt(str(path), torch.device("cpu"))
                inputs = {name: torch.from_numpy(fixture[f"{case}/{name}"].copy()) for name in ["x", "t", "exog"]}
                key = f"{case}/labels"
                inputs["y"] = torch.from_numpy(fixture[key].copy()) if key in fixture else None
                with torch.no_grad(): actual = model(**inputs)
                torch.testing.assert_close(actual, torch.from_numpy(fixture[f"{case}/expected"]), rtol=0, atol=0)
                saved = torch.load(path, weights_only=False)["model"]
                self.assertEqual(set(model.state_dict()), set(saved))
                for name, value in saved.items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    def test_ema_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint_from_fixture(directory, "ternary", ema=True)
            _, _, which = load_model_from_ckpt(str(path), torch.device("cpu"), prefer_ema=False)
            self.assertEqual(which, "ema")
            payload = torch.load(path, weights_only=False)
            payload["model"] = {k: v + 1 for k, v in payload["ema"].items()}
            torch.save(payload, path)
            model, _, which = load_model_from_ckpt(str(path), torch.device("cpu"))
            self.assertEqual(which, "ema")
            torch.testing.assert_close(model.state_dict()["t_embedder.mlp.0.weight"], payload["ema"]["t_embedder.mlp.0.weight"])

    def test_new_binary_checkpoint_has_distinct_present_and_null_states(self):
        model = small_model(num_classes=2, multilabels=False, class_dropout_prob=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binary.pt"
            cfg = training_config(1, False, conditioning_version=2, num_classes=2, class_dropout_prob=0)
            torch.save({"args": cfg, "model": model.state_dict()}, path)
            loaded, _, _ = load_model_from_ckpt(str(path), torch.device("cpu"))
            self.assertEqual(loaded.num_classes, 2)
            self.assertEqual(loaded.y_embedder.embedding_table.num_embeddings, 2)
        model = small_model(num_classes=2, multilabels=False)
        labels = torch.tensor([[0], [1]])
        present = model.y_embedder(labels, False)[1]
        null = model.y_embedder(labels, False, force_drop_ids=torch.ones(2))[1]
        self.assertFalse(torch.equal(present, null))


class ConditioningSemantics(unittest.TestCase):
    def test_all_ternary_conditions_preserve_slot_identity(self):
        import itertools
        embedder = MultiLabelEmbedder(3, 10, p_uncond=0)
        with torch.no_grad(): embedder.embedding_table.weight.copy_(torch.eye(10))
        labels = torch.tensor(list(itertools.product([-1, 0, 1], repeat=3)))
        self.assertEqual(torch.unique(embedder(labels, False), dim=0).shape[0], 27)

    def test_partial_dropout_trains_unknown_states_through_diffusion_loss(self):
        torch.manual_seed(91)
        model = small_model(num_classes=3, class_dropout_prob=0, partial_label_dropout_prob=1)
        diffusion = create_diffusion("", noise_schedule="squaredcos_cap_v2", diffusion_steps=12)
        diffusion.set_exog(torch.randn(4, 4, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(4):
            optimizer.zero_grad()
            loss = diffusion.training_losses(model, torch.randn(4, 1, 4, 2), torch.tensor([1, 2, 3, 4]),
                                             {"y": torch.randint(0, 2, (4, 3))})["loss"].mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
        gradient = model.y_embedder.embedding_table.weight.grad
        self.assertTrue((gradient[[0, 3, 6]].abs().sum(dim=1) > 0).all())

    def test_unconditional_forward_uses_the_trained_null_vector(self):
        model = small_model(num_classes=3).eval()
        captured = []
        handle = model.blocks[0].register_forward_pre_hook(lambda module, args: captured.append(args[1]))
        x, t, exog = torch.randn(2, 1, 4, 2), torch.tensor([1, 2]), torch.randn(2, 4, 4)
        with torch.no_grad(): model(x, t, exog=exog)
        expected = model.t_embedder(t) + model.y_embedder.embedding_table.weight[9]
        torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
        handle.remove()

    def test_guidance_has_correct_batch_shape_and_distinct_label_branches(self):
        model = small_model(num_classes=3).eval()
        captured = []
        handle = model.blocks[0].register_forward_pre_hook(lambda module, args: captured.append(args[1]))
        x, t, exog, y = torch.randn(2, 1, 4, 2), torch.tensor([1, 2]), torch.randn(4, 4), torch.ones(2, 3, dtype=torch.long)
        with torch.no_grad(): output = model.forward_with_cfg(x, t, exog, y, 3, guide_labels=True, guide_temperature=False)
        self.assertEqual(output.shape, (2, 2, 4, 2))
        self.assertFalse(torch.equal(captured[0][:2], captured[0][2:]))
        handle.remove()
        diffusion = create_diffusion("ddim2", diffusion_steps=100)
        diffusion.set_exog(exog)
        result = diffusion.p_sample_loop(model.forward_with_cfg, x.shape,
            model_kwargs={"y": y, "cfg_scale": 2, "guide_labels": True, "guide_temperature": False}, device="cpu")
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.isfinite(result).all())

    def test_binary_class_count_and_training_builder(self):
        self.assertEqual(conditioning_class_count(["cooker"], False), 2)
        self.assertEqual(conditioning_class_count(["cooker"], True), 1)
        self.assertEqual(conditioning_class_count([], False), 0)
        with self.assertRaises(ValueError): conditioning_class_count(["cooker", "heater"], False)
        # Execute the actual builder without importing its unrelated reporting dependencies.
        tree = ast.parse((REPO / "scripts/training/train_loadiff.py").read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"resolve_conditioning", "build_model"}]
        def requires_grad(model, value):
            for parameter in model.parameters(): parameter.requires_grad_(value)
        namespace = {"OmegaConf": OmegaConf, "DiT": DiT, "deepcopy": deepcopy,
                     "requires_grad": requires_grad, "create_diffusion": create_diffusion,
                     "conditioning_class_count": conditioning_class_count}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "train_loadiff.py", "exec"), namespace)
        cfg = training_config(1, False)
        _, count, _, _ = namespace["resolve_conditioning"](cfg)
        model, _, _ = namespace["build_model"](cfg, count, torch.device("cpu"))
        self.assertEqual(cfg.ditmodelargs.conditioning_version, 2)
        self.assertEqual(cfg.ditmodelargs.num_classes, 2)
        self.assertEqual(model.y_embedder(torch.tensor([[0], [1]]), False).shape, (2, 16))

    def test_invalid_fractional_and_nan_states_are_rejected(self):
        for bad in [0.9, float("nan"), float("inf"), 2]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MultiLabelEmbedder(1, 4)(torch.tensor([[bad]]), False)
            with self.subTest(user_bad=bad), self.assertRaises((ValueError, OverflowError)):
                build_user_labels({"values": {"cooker": bad}, "num_samples_per_combination": 1}, ["cooker"])

    def test_embedding_dtype_is_preserved(self):
        embedder = MultiLabelEmbedder(2, 4).double()
        self.assertEqual(embedder(torch.tensor([[0, 1]]), False).dtype, torch.float64)


class PartialInference(unittest.TestCase):
    def test_completion_is_reproducible_and_does_not_change_the_global_rng(self):
        requested = torch.tensor([[1, -1, -1]] * 16)
        state = torch.random.get_rng_state().clone()
        realized, sampled = complete_unknown_labels(requested, [False] * 3, seed=7)
        torch.testing.assert_close(torch.random.get_rng_state(), state)
        self.assertTrue((realized[:, 0] == 1).all())
        self.assertTrue(((realized == 0) | (realized == 1)).all())
        self.assertTrue(torch.equal(realized, complete_unknown_labels(requested, [False] * 3, seed=7)[0]))
        self.assertEqual(int(sampled.sum()), 32)

    def test_trained_unknown_slots_and_strict_policy(self):
        requested = torch.tensor([[-1, -1, 1]])
        realized, _ = complete_unknown_labels(requested, [True, False, False])
        self.assertEqual(int(realized[0, 0]), -1)
        self.assertIn(int(realized[0, 1]), [0, 1])
        with self.assertRaises(ValueError): complete_unknown_labels(requested, [False] * 3, policy="require_trained")
        realized, _ = complete_unknown_labels(requested, [True] * 3, policy="require_trained")
        torch.testing.assert_close(realized, requested)

    def test_user_batch_saves_requested_and_realized_labels(self):
        cfg = OmegaConf.create({"inference": {"seed": 7, "n_samples": 16,
            "gen_sample_start_date": None, "gen_sample_days": None,
            "conditioning": {"values": {"cooker": 1}, "num_samples_per_combination": 16}}})
        batch = build_user_conditioned(cfg, training_config(), torch.device("cpu"))
        self.assertTrue((batch.y[:, 0] == 1).all())
        for i, row in enumerate(batch.row_meta):
            self.assertEqual(row["requested_dishwasher"], -1)
            self.assertEqual(row["dishwasher"], int(batch.y[i, 1]))

    def test_no_dropout_checkpoint_can_sample_without_explicit_labels(self):
        model = small_model(num_classes=3, class_dropout_prob=0).eval()
        diffusion = create_diffusion("ddim2", diffusion_steps=100)
        result = run_sampling(model, diffusion, exog=torch.randn(4, 4), y=None,
            n_samples=3, n_days=4, patch_length=2, value_scale=1, batch_size=2, device=torch.device("cpu"))
        self.assertEqual(result.shape, (3, 8))
        self.assertTrue(np.isfinite(result).all())

    def test_real_inference_entrypoint_with_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = checkpoint_from_fixture(directory, "ternary")
            for mode in ["user_conditioned", "unconditional"]:
                run_dir = Path(directory) / mode
                overrides = [f"inference.ckpt_path={checkpoint}",
                    "inference.device=cpu", "inference.n_samples=4", "inference.batch_size=2",
                    "inference.timestep_respacing=ddim2", "inference.evaluate=false",
                    "postprocessing.enabled=false", f"hydra.run.dir={run_dir}"]
                if mode == "user_conditioned":
                    overrides.append("inference.conditioning.combinations=[{values:{cooker:1},num_samples:4}]")
                with initialize_config_dir(version_base=None, config_dir=str(REPO / "configs")):
                    cfg = compose(config_name=f"inference_loadiff_{mode}", overrides=overrides, return_hydra_config=True)
                    cfg.hydra.runtime.output_dir = str(run_dir)
                    HydraConfig.instance().set_config(cfg)
                    application_cfg = OmegaConf.masked_copy(cfg, [key for key in cfg if key != "hydra"])
                    main.__wrapped__(application_cfg)
                samples = np.load(run_dir / "loadit_samples.npy")
                self.assertEqual(samples.shape, (4, 8))
                self.assertTrue(np.isfinite(samples).all())
                if mode == "user_conditioned":
                    labels = np.load(run_dir / "y.npy")
                    metadata = pd.read_csv(run_dir / "metadata.csv")
                    np.testing.assert_array_equal(metadata[["cooker", "dishwasher", "water_heater"]], labels)
                    self.assertTrue((metadata.requested_dishwasher == -1).all())
                self.assertTrue((run_dir / "run_info.json").is_file())


if __name__ == "__main__":
    unittest.main()
