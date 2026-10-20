import argparse
import unittest

import torch
import torch.nn as nn

from utils.sam_node_audit import SamNodeAuditConfig


class FakeBackbone(nn.Module):
    def forward(self, image):
        base = image.mean(dim=1, keepdim=True)
        stage = base.repeat(1, 4, 1, 1)
        neck = base.repeat(1, 3, 1, 1)
        return base, base, stage, base, neck


class RaisingFeatureProvider(nn.Module):
    def forward(self, image_a, image_b):
        raise AssertionError("fallback backbone must not run when cached features are supplied")


class TestGraphStage1TrainingRuntime(unittest.TestCase):
    def test_cli_arguments_are_disabled_by_default_and_configurable(self):
        from utils.graph_stage1_training import add_graph_stage1_arguments
        parser = argparse.ArgumentParser()
        add_graph_stage1_arguments(parser)
        defaults = parser.parse_args([])
        enabled = parser.parse_args([
            "--use-sam-graph-stage1",
            "--sam-cache-root", "/cache",
            "--sam-node-min-area-ratio", "0.0025",
            "--sam-node-max-area-ratio", "0.10",
            "--lambda-graph-gt", "0.2",
            "--lambda-edge-gt", "0.3",
        ])
        self.assertFalse(defaults.use_sam_graph_stage1)
        self.assertAlmostEqual(defaults.sam_node_min_area_ratio, 0.001)
        self.assertAlmostEqual(defaults.sam_node_max_area_ratio, 0.05)
        self.assertTrue(enabled.use_sam_graph_stage1)
        self.assertEqual(enabled.sam_cache_root, "/cache")
        self.assertAlmostEqual(enabled.sam_node_min_area_ratio, 0.0025)
        self.assertAlmostEqual(enabled.sam_node_max_area_ratio, 0.10)
        self.assertAlmostEqual(enabled.lambda_graph_gt, 0.2)
        self.assertAlmostEqual(enabled.lambda_edge_gt, 0.3)

    def test_stage2_cli_arguments_are_disabled_by_default_and_configurable(self):
        from utils.graph_stage1_training import add_graph_stage1_arguments
        parser = argparse.ArgumentParser()
        add_graph_stage1_arguments(parser)
        defaults = parser.parse_args([])
        enabled = parser.parse_args([
            "--use-sam-graph-stage2",
            "--graph-stage2-start-epoch", "55",
            "--graph-local-topk", "6",
            "--graph-edge-threshold", "0.7",
            "--graph-pixel-pseudo-weight", "0.05",
        ])
        self.assertFalse(defaults.use_sam_graph_stage2)
        self.assertTrue(enabled.use_sam_graph_stage2)
        self.assertEqual(enabled.graph_stage2_start_epoch, 55)
        self.assertEqual(enabled.graph_local_topk, 6)
        self.assertAlmostEqual(enabled.graph_edge_threshold, 0.7)
        self.assertAlmostEqual(enabled.graph_pixel_pseudo_weight, 0.05)

    def test_build_sam_cache_config_uses_training_area_range(self):
        from utils.graph_stage1_training import (
            add_graph_stage1_arguments, build_sam_cache_config)
        parser = argparse.ArgumentParser()
        add_graph_stage1_arguments(parser)
        options = parser.parse_args([
            "--sam-node-min-area-ratio", "0.001",
            "--sam-node-max-area-ratio", "0.05",
        ])

        config = build_sam_cache_config(options)

        self.assertAlmostEqual(config.min_area_ratio, 0.001)
        self.assertAlmostEqual(config.max_area_ratio, 0.05)

    def _runtime(self):
        from utils.graph_stage1_training import GraphStage1TrainingRuntime
        return GraphStage1TrainingRuntime(
            backbone=FakeBackbone(),
            num_classes=3,
            stage_channels=4,
            neck_channels=3,
            node_dim=8,
            learning_rate=1e-3,
            audit_config=SamNodeAuditConfig(
                tau_gt=0.5, tau_purity=0.5, tau_u=0.9,
                min_mask_pixels=1, min_changed_pixels=0),
            max_pairs=16,
        )

    def _stage2_runtime(self):
        from utils.graph_stage1_training import GraphStage1TrainingRuntime
        return GraphStage1TrainingRuntime(
            backbone=FakeBackbone(), num_classes=3,
            stage_channels=4, neck_channels=3, node_dim=8,
            learning_rate=1e-3, use_stage2=True,
            graph_gat_heads=2, graph_gat_dropout=0.0,
            audit_config=SamNodeAuditConfig(
                tau_gt=0.5, tau_purity=0.5, tau_u=0.9,
                min_mask_pixels=1, min_changed_pixels=0),
            max_pairs=16,
        )

    def _batch(self):
        image_a = torch.ones(1, 3, 4, 4)
        image_b = torch.full((1, 3, 4, 4), 2.0)
        label_a = torch.ones(1, 4, 4, dtype=torch.long)
        label_b = torch.full((1, 4, 4), 2, dtype=torch.long)
        masks = torch.ones(1, 4, 4, dtype=torch.bool)
        graph_data = [{"sample_id": "one", "shared_masks": masks}]
        return image_a, image_b, label_a, label_b, graph_data

    def test_compute_is_finite_and_backpropagates_graph_parameters(self):
        runtime = self._runtime()
        result = runtime.compute(*self._batch())
        loss = result.loss_graph_gt + result.loss_edge_gt
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in runtime.parameters()
            if parameter.requires_grad))

    def test_compute_uses_supplied_backbone_features_without_rerunning_backbone(self):
        runtime = self._runtime()
        runtime.feature_provider = RaisingFeatureProvider()
        cached_features = (
            torch.ones(1, 4, 4, 4),
            torch.ones(1, 3, 4, 4),
            torch.full((1, 4, 4, 4), 2.0),
            torch.full((1, 3, 4, 4), 2.0),
        )

        result = runtime.compute(
            *self._batch(), backbone_features=cached_features)
        loss = result.loss_graph_gt + result.loss_edge_gt
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in runtime.parameters()
            if parameter.requires_grad))

    def test_checkpoint_round_trip_restores_queue_and_optimizer(self):
        runtime = self._runtime()
        result = runtime.compute(*self._batch())
        (result.loss_graph_gt + result.loss_edge_gt).backward()
        runtime.optimizer.step()
        state = runtime.state_dict()

        restored = self._runtime()
        self.assertTrue(restored.load_checkpoint_payload({"sam_graph_stage1": state}))
        nodes, classes = restored.stage1.queue.snapshot("cpu")
        self.assertEqual(nodes.shape, (2, 8))
        self.assertEqual(classes.tolist(), [1, 2])

    def test_old_checkpoint_starts_with_fresh_graph_state(self):
        runtime = self._runtime()
        self.assertFalse(runtime.load_checkpoint_payload({"format_version": 1}))

    def test_stage1_v1_checkpoint_loads_into_stage2_with_fresh_gat(self):
        stage1_runtime = self._runtime()
        result = stage1_runtime.compute(*self._batch())
        (result.loss_graph_gt + result.loss_edge_gt).backward()
        stage1_runtime.optimizer.step()
        checkpoint = {"sam_graph_stage1": stage1_runtime.state_dict()}

        restored = self._stage2_runtime()
        gat_before = {
            key: value.clone() for key, value in restored.stage2.state_dict().items()}
        self.assertTrue(restored.load_checkpoint_payload(checkpoint))

        self.assertEqual(restored.loaded_format_version, 1)
        self.assertTrue(all(torch.equal(gat_before[key], value)
                            for key, value in restored.stage2.state_dict().items()))
        self.assertGreater(len(restored.optimizer.state), 0)

    def test_stage2_checkpoint_round_trip_restores_gat_and_optimizer(self):
        runtime = self._stage2_runtime()
        result = runtime.compute(*self._batch(), stage2_enabled=True)
        total = (result.loss_graph_gt + result.loss_edge_gt
                 + result.loss_stage2_graph_gt + result.loss_graph_temp)
        total.backward()
        runtime.optimizer.step()
        state = runtime.state_dict()
        self.assertEqual(state["format_version"], 2)

        restored = self._stage2_runtime()
        restored.load_state_dict(state)
        for key, value in runtime.stage2.state_dict().items():
            self.assertTrue(torch.equal(value, restored.stage2.state_dict()[key]))
        self.assertEqual(len(runtime.optimizer.state), len(restored.optimizer.state))

    def test_stage2_disabled_compute_keeps_original_result_shape(self):
        stage1_only = self._runtime()
        runtime = self._stage2_runtime()
        runtime.projector.load_state_dict(stage1_only.projector.state_dict())
        runtime.stage1.load_state_dict(stage1_only.stage1.state_dict())
        torch.manual_seed(17)
        expected = stage1_only.compute(*self._batch())
        torch.manual_seed(17)
        result = runtime.compute(*self._batch(), stage2_enabled=False)
        self.assertIsNone(result.stage2_result)
        self.assertEqual(float(result.loss_stage2_graph_gt), 0.0)
        self.assertEqual(float(result.loss_graph_temp), 0.0)
        self.assertTrue(torch.equal(expected.loss_graph_gt, result.loss_graph_gt))
        self.assertTrue(torch.equal(expected.loss_edge_gt, result.loss_edge_gt))
        self.assertEqual(expected.gt_node_count, result.gt_node_count)


if __name__ == "__main__":
    unittest.main()
