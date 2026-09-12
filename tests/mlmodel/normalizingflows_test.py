import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock

try:
    import torch
    from nflows.distributions import StandardNormal
    from nflows.flows import Flow
    from nflows.transforms import CompositeTransform, RandomPermutation
    from nflows.transforms.autoregressive import (
        MaskedAffineAutoregressiveTransform,
    )
    from torch import nn

    from nmma.mlmodel import normalizingflows
    from nmma.mlmodel.embedding import SimilarityEmbedding

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is the optional neuralnet extra
    TORCH_AVAILABLE = False

# The stub below subclasses torch's module base, which only exists once the
# optional extra is installed; without it the class is never instantiated.
StubFlowBase = nn.Module if TORCH_AVAILABLE else object

needs_torch = unittest.skipUnless(
    TORCH_AVAILABLE, "torch or nflows is not installed; install the neuralnet extra"
)


@needs_torch
class TestModuleConstants(unittest.TestCase):
    """The flow infers three ejecta parameters from a seven-dimensional
    context produced by the embedding."""

    def test_the_context_width_matches_the_representation_width(self):
        self.assertEqual(normalizingflows.num_dim, 7)
        self.assertEqual(normalizingflows.context_features, normalizingflows.num_dim)

    def test_the_light_curve_length_matches_the_padding_target(self):
        from nmma.mlmodel import dataprocessing

        self.assertEqual(normalizingflows.num_points, dataprocessing.num_points)

    def test_a_module_level_embedding_is_built_at_import(self):
        self.assertIsInstance(
            normalizingflows.similarity_embedding, SimilarityEmbedding
        )

    def test_the_device_is_resolved_at_import(self):
        self.assertIsInstance(normalizingflows.device, torch.device)


@needs_torch
class TestEmbeddingNet(unittest.TestCase):
    """Wraps the trained embedding so the flow sees a context vector, and
    freezes the parts of it that only existed for contrastive training."""

    def build(self):
        return normalizingflows.EmbeddingNet(
            SimilarityEmbedding(num_dim=7, num_dim_final=5)
        )

    def test_the_light_curve_is_reduced_to_the_context_vector(self):
        network = self.build().eval()
        self.assertEqual(tuple(network(torch.randn(4, 3, 121)).shape), (4, 7))

    def test_the_batch_size_is_taken_from_the_input(self):
        network = self.build().eval()
        self.assertEqual(network(torch.randn(9, 3, 121)).shape[0], 9)

    def test_the_wrapped_embedding_is_kept_as_a_submodule(self):
        network = self.build()
        self.assertIsInstance(network.representation_net, SimilarityEmbedding)

    def test_the_expander_is_frozen_because_it_is_unused(self):
        # Only the representation feeds the flow, so the expander head is
        # left out of the gradient computation.
        network = self.build()
        for name, parameter in network.representation_net.named_parameters():
            if "expander_layer" in name or "final_layer" in name:
                self.assertFalse(parameter.requires_grad, msg=name)

    def test_the_expander_hidden_layers_are_frozen(self):
        network = self.build()
        for name, parameter in network.representation_net.named_parameters():
            if "layers_h" in name:
                self.assertFalse(parameter.requires_grad, msg=name)

    def test_the_backbone_stays_trainable_so_the_flow_can_tune_it(self):
        network = self.build()
        trainable = [
            name
            for name, parameter in network.representation_net.named_parameters()
            if "layers_f" in name and parameter.requires_grad
        ]
        self.assertTrue(trainable)

    def test_the_contraction_layer_stays_trainable(self):
        network = self.build()
        for name, parameter in network.representation_net.named_parameters():
            if "contraction_layer" in name:
                self.assertTrue(parameter.requires_grad, msg=name)

    def test_a_context_layer_is_added_on_top(self):
        network = self.build()
        self.assertIsInstance(network.context_layer, nn.Sequential)
        self.assertEqual(network.context_layer[0].in_features, 7)
        self.assertEqual(network.context_layer[-1].out_features, 7)

    def test_the_output_is_differentiable_through_the_context_layer(self):
        network = self.build()
        network(torch.randn(2, 3, 121)).sum().backward()
        self.assertIsNotNone(network.context_layer[0].weight.grad)

    def test_an_embedding_of_the_wrong_width_cannot_be_reshaped(self):
        # The reshape to the module-level context width is unconditional, so
        # the wrapped embedding has to have been built with num_dim=7.
        network = normalizingflows.EmbeddingNet(
            SimilarityEmbedding(num_dim=3, num_dim_final=5)
        ).eval()
        with self.assertRaises(RuntimeError):
            network(torch.randn(4, 3, 121))


@needs_torch
class TestNormflowParams(unittest.TestCase):
    """Builds the three pieces the flow is assembled from: the transform
    stack, the base distribution and the context network."""

    def build(self, num_transforms=2, num_blocks=2, hidden_features=16):
        return normalizingflows.normflow_params(
            SimilarityEmbedding(num_dim=7, num_dim_final=5),
            num_transforms,
            num_blocks,
            hidden_features,
            context_features=7,
            num_dim=7,
        )

    def test_three_pieces_are_returned(self):
        transform, base_dist, embedding_net = self.build()
        self.assertIsInstance(transform, CompositeTransform)
        self.assertIsInstance(base_dist, StandardNormal)
        self.assertIsInstance(embedding_net, normalizingflows.EmbeddingNet)

    def test_the_base_distribution_is_over_the_three_ejecta_parameters(self):
        _, base_dist, _ = self.build()
        self.assertEqual(tuple(base_dist._shape), (3,))

    def test_each_transform_is_paired_with_a_permutation(self):
        # A permutation after every autoregressive layer lets each parameter
        # be conditioned on the others.
        transform, _, _ = self.build(num_transforms=3)
        self.assertEqual(len(transform._transforms), 6)

    def test_the_layers_alternate_between_autoregression_and_permutation(self):
        transform, _, _ = self.build(num_transforms=2)
        kinds = [type(layer) for layer in transform._transforms]
        self.assertEqual(
            kinds,
            [
                MaskedAffineAutoregressiveTransform,
                RandomPermutation,
                MaskedAffineAutoregressiveTransform,
                RandomPermutation,
            ],
        )

    def test_no_transforms_gives_an_empty_stack(self):
        transform, _, _ = self.build(num_transforms=0)
        self.assertEqual(len(transform._transforms), 0)

    def test_the_number_of_sampled_parameters_is_fixed_at_three(self):
        # The final num_dim argument names the context width, not the
        # sampled dimension, which is hard-coded to the three ejecta
        # parameters the released weights were trained on.
        transform, base_dist, _ = normalizingflows.normflow_params(
            SimilarityEmbedding(num_dim=7, num_dim_final=5),
            1,
            2,
            16,
            context_features=7,
            num_dim=99,
        )
        self.assertEqual(tuple(base_dist._shape), (3,))

    def test_the_pieces_assemble_into_a_working_flow(self):
        transform, base_dist, embedding_net = self.build()
        flow = Flow(transform, base_dist, embedding_net)
        samples = flow.sample(5, context=torch.randn(2, 3, 121))
        self.assertEqual(tuple(samples.shape), (2, 5, 3))

    def test_the_assembled_flow_scores_the_ejecta_parameters(self):
        transform, base_dist, embedding_net = self.build()
        flow = Flow(transform, base_dist, embedding_net)
        log_prob = flow.log_prob(torch.randn(4, 3), context=torch.randn(4, 3, 121))
        self.assertEqual(tuple(log_prob.shape), (4,))

    def test_the_released_weights_load_into_the_released_architecture(self):
        # em.analysis builds the flow with exactly these settings before
        # loading the frozen weights, so the shapes have to match.
        from pathlib import Path

        import nmma

        embedding_network = SimilarityEmbedding(
            num_dim=7,
            num_hidden_layers_f=1,
            num_hidden_layers_h=1,
            num_blocks=4,
            kernel_size=5,
            num_dim_final=5,
        )
        transform, base_dist, embedding_net = normalizingflows.normflow_params(
            embedding_network, 9, 5, 90, context_features=7, num_dim=7
        )
        flow = Flow(transform, base_dist, embedding_net)
        weights = Path(nmma.__file__).parent / "mlmodel" / "frozen-flow-weights.pth"
        flow.load_state_dict(torch.load(weights, map_location="cpu"))


@needs_torch
class StubFlow(StubFlowBase):
    """Stands in for the flow during the loop tests. It holds one parameter
    so the backward pass has something to reach, and records what it saw."""

    def __init__(self, loss_value=2.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.loss_value = loss_value
        self.calls = []

    def log_prob(self, inputs, context=None):
        self.calls.append((inputs.shape, context.shape))
        return -self.loss_value * self.scale * torch.ones(inputs.shape[0])


@needs_torch
class FlowLoopMixin:
    """Both loops slice the first three columns off the parameters, flatten
    away the repeat axis and reshape the light curves into channels."""

    repeats = 2
    points = 121

    def setUp(self):
        self.writer = MagicMock()
        self.flow = StubFlow()
        # A zero step keeps the stub's loss constant across the epoch, so the
        # reported average is checkable; the update test raises it.
        self.optimizer = torch.optim.SGD(self.flow.parameters(), lr=0.0)

    def batches(self, count):
        parameters = torch.randn(1, self.repeats, 5)
        light_curves = torch.randn(1, self.repeats, 3, self.points)
        return [
            (parameters, parameters.clone(), light_curves, light_curves.clone())
            for _ in range(count)
        ]


@needs_torch
class TestTrainOneEpoch(FlowLoopMixin, unittest.TestCase):
    def run_epoch(self, count, flatten_dim=1):
        return normalizingflows.train_one_epoch(
            0, self.writer, self.batches(count), self.flow, self.optimizer, flatten_dim
        )

    def test_a_loss_is_reported_once_ten_batches_have_been_seen(self):
        self.assertAlmostEqual(self.run_epoch(10), 2.0, places=3)

    def test_fewer_than_ten_batches_report_no_loss_at_all(self):
        # The running loss is only flushed every tenth batch, so a short
        # epoch returns the initial zero rather than its real loss.
        self.assertEqual(self.run_epoch(9), 0.0)

    def test_only_the_three_ejecta_parameters_are_scored(self):
        self.run_epoch(1)
        scored_shape, _ = self.flow.calls[0]
        self.assertEqual(scored_shape[-1], 3)

    def test_the_repeat_axis_is_flattened_into_the_batch(self):
        self.run_epoch(1)
        scored_shape, _ = self.flow.calls[0]
        self.assertEqual(tuple(scored_shape), (self.repeats, 3))

    def test_the_light_curves_are_reshaped_into_bands_and_points(self):
        self.run_epoch(1)
        _, context_shape = self.flow.calls[0]
        self.assertEqual(tuple(context_shape), (self.repeats, 3, self.points))

    def test_the_loss_is_written_to_tensorboard(self):
        self.run_epoch(10)
        self.writer.add_scalar.assert_called_once()
        self.assertEqual(self.writer.add_scalar.call_args.args[0], "Flow Loss/train")

    def test_the_writer_is_flushed_so_the_epoch_is_visible(self):
        self.run_epoch(10)
        self.assertTrue(self.writer.flush.called)

    def test_nothing_is_written_for_a_short_epoch(self):
        self.run_epoch(9)
        self.writer.add_scalar.assert_not_called()

    def test_the_weights_are_updated(self):
        self.optimizer = torch.optim.SGD(self.flow.parameters(), lr=1e-3)
        before = float(self.flow.scale)
        self.run_epoch(10)
        self.assertNotEqual(before, float(self.flow.scale))

    def test_the_tensorboard_step_counts_across_epochs(self):
        normalizingflows.train_one_epoch(
            3, self.writer, self.batches(10), self.flow, self.optimizer, 1
        )
        self.assertEqual(self.writer.add_scalar.call_args.args[2], 3 * 10 + 10)

    def test_the_running_loss_is_reported_per_batch_not_accumulated(self):
        self.flow = StubFlow(loss_value=5.0)
        self.optimizer = torch.optim.SGD(self.flow.parameters(), lr=0.0)
        self.assertAlmostEqual(self.run_epoch(10), 5.0, places=3)

    def test_the_loss_drifts_as_the_weights_are_trained(self):
        self.optimizer = torch.optim.SGD(self.flow.parameters(), lr=1e-2)
        self.assertNotAlmostEqual(self.run_epoch(10), 2.0, places=3)

    def test_the_progress_is_printed(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.run_epoch(10)
        self.assertIn("train loss/batch", stream.getvalue())


@needs_torch
class TestValidateOneEpoch(FlowLoopMixin, unittest.TestCase):
    def run_epoch(self, count, flatten_dim=1):
        return normalizingflows.val_one_epoch(
            0, self.writer, self.batches(count), self.flow, flatten_dim
        )

    def test_a_loss_is_reported_after_every_batch(self):
        # Validation flushes on every batch, unlike training.
        self.assertAlmostEqual(self.run_epoch(1), 2.0, places=3)

    def test_every_batch_is_written_to_tensorboard(self):
        self.run_epoch(3)
        self.assertEqual(self.writer.add_scalar.call_count, 3)
        self.assertEqual(self.writer.add_scalar.call_args.args[0], "Flow Loss/val")

    def test_the_weights_are_left_untouched(self):
        before = float(self.flow.scale)
        self.run_epoch(3)
        self.assertEqual(before, float(self.flow.scale))

    def test_the_same_reshaping_is_applied_as_in_training(self):
        self.run_epoch(1)
        scored_shape, context_shape = self.flow.calls[0]
        self.assertEqual(tuple(scored_shape), (self.repeats, 3))
        self.assertEqual(tuple(context_shape), (self.repeats, 3, self.points))

    def test_an_empty_loader_reports_no_loss(self):
        self.assertEqual(self.run_epoch(0), 0.0)

    def test_the_writer_is_flushed_even_for_an_empty_loader(self):
        self.run_epoch(0)
        self.assertTrue(self.writer.flush.called)


if __name__ == "__main__":
    unittest.main()
