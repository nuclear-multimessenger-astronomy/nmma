import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock

try:
    import torch
    from torch import nn

    from nmma.mlmodel import embedding
    from nmma.mlmodel.resnet import ResNet

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is the optional neuralnet extra
    TORCH_AVAILABLE = False

needs_torch = unittest.skipUnless(
    TORCH_AVAILABLE, "torch is not installed; install the neuralnet extra"
)


@needs_torch
class TestModuleConstants(unittest.TestCase):
    """The embedding is trained on three ZTF bands padded to a fixed length,
    and those numbers are baked in as module constants."""

    def test_the_three_ztf_bands_are_the_channels(self):
        self.assertEqual(embedding.bands, ["ztfg", "ztfr", "ztfi"])
        self.assertEqual(embedding.num_channels, len(embedding.bands))

    def test_the_light_curve_length_matches_the_padding_target(self):
        from nmma.mlmodel import dataprocessing

        self.assertEqual(embedding.num_points, dataprocessing.num_points)

    def test_the_detection_limit_is_the_padding_value(self):
        from nmma.mlmodel import dataprocessing

        self.assertEqual(embedding.detection_limit, dataprocessing.detection_limit)


@needs_torch
class TestVICRegLossOffDiagonal(unittest.TestCase):
    """The covariance term only penalises correlations between different
    features, so the diagonal has to be dropped from each batch matrix."""

    def setUp(self):
        self.loss = embedding.VICRegLoss()

    def test_the_diagonal_is_dropped_from_each_matrix(self):
        covariance = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
        self.assertEqual(
            self.loss.off_diagonal(covariance).tolist(), [1.0, 2.0, 3.0, 5.0, 6.0, 7.0]
        )

    def test_every_matrix_in_the_batch_contributes(self):
        covariance = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
        self.assertEqual(len(self.loss.off_diagonal(covariance)), 12)

    def test_the_element_count_is_the_batch_times_the_off_diagonal_size(self):
        covariance = torch.randn(4, 5, 5)
        self.assertEqual(len(self.loss.off_diagonal(covariance)), 4 * (25 - 5))

    def test_a_non_square_matrix_is_refused(self):
        with self.assertRaises(AssertionError):
            self.loss.off_diagonal(torch.randn(1, 3, 4))

    def test_a_diagonal_matrix_has_no_off_diagonal_content(self):
        covariance = torch.eye(4).unsqueeze(0)
        self.assertEqual(float(self.loss.off_diagonal(covariance).abs().sum()), 0.0)


@needs_torch
class TestVICRegLoss(unittest.TestCase):
    """Three terms keep the two embeddings of the same light curve close
    without letting the representation collapse to a constant."""

    def setUp(self):
        self.loss = embedding.VICRegLoss()

    def test_four_values_are_returned(self):
        total, repr_loss, cov_loss, std_loss = self.loss(
            torch.randn(8, 5), torch.randn(8, 5)
        )
        for value in [total, repr_loss, cov_loss, std_loss]:
            self.assertEqual(value.ndim, 0)

    def test_identical_embeddings_have_no_invariance_penalty(self):
        embedded = torch.randn(8, 5)
        _, repr_loss, _, _ = self.loss(embedded, embedded.clone())
        self.assertAlmostEqual(float(repr_loss), 0.0, places=6)

    def test_differing_embeddings_are_penalised(self):
        _, repr_loss, _, _ = self.loss(torch.zeros(8, 5), torch.ones(8, 5))
        self.assertAlmostEqual(float(repr_loss), 1.0, places=5)

    def test_a_collapsed_representation_is_penalised_by_the_variance_term(self):
        # Every sample embedding identically is the failure mode the
        # variance term exists to prevent.
        collapsed = torch.zeros(8, 5)
        _, _, _, std_loss = self.loss(collapsed, collapsed)
        self.assertGreater(float(std_loss), 0.0)

    def test_a_well_spread_representation_has_little_variance_penalty(self):
        spread = torch.randn(512, 5) * 5
        _, _, _, std_loss = self.loss(spread, spread.clone())
        self.assertLess(float(std_loss), 1e-3)

    def test_the_total_is_the_weighted_sum_of_the_three_terms(self):
        first, second = torch.randn(8, 5), torch.randn(8, 5)
        total, repr_loss, cov_loss, std_loss = self.loss(first, second)
        torch.testing.assert_close(total, repr_loss + cov_loss + std_loss)

    def test_each_term_can_be_reweighted(self):
        first, second = torch.randn(8, 5), torch.randn(8, 5)
        total, repr_loss, cov_loss, std_loss = self.loss(
            first, second, wt_repr=2.0, wt_cov=3.0, wt_std=4.0
        )
        torch.testing.assert_close(
            total, 2.0 * repr_loss + 3.0 * cov_loss + 4.0 * std_loss
        )

    def test_a_zero_weight_removes_a_term(self):
        first, second = torch.randn(8, 5), torch.randn(8, 5)
        total, repr_loss, _, std_loss = self.loss(first, second, wt_cov=0.0)
        torch.testing.assert_close(total, repr_loss + std_loss)

    def test_the_loss_is_differentiable(self):
        first = torch.randn(8, 5, requires_grad=True)
        total, _, _, _ = self.loss(first, torch.randn(8, 5))
        total.backward()
        self.assertIsNotNone(first.grad)

    def test_the_covariance_term_is_never_negative(self):
        _, _, cov_loss, _ = self.loss(torch.randn(16, 5), torch.randn(16, 5))
        self.assertGreaterEqual(float(cov_loss), 0.0)


@needs_torch
class TestConvResidualBlock(unittest.TestCase):
    def test_the_shape_is_preserved(self):
        block = embedding.ConvResidualBlock(channels=8, kernel_size=5).eval()
        self.assertEqual(tuple(block(torch.randn(2, 8, 32)).shape), (2, 8, 32))

    def test_two_convolutions_are_used(self):
        block = embedding.ConvResidualBlock(channels=8, kernel_size=5)
        self.assertEqual(len(block.conv_layers), 2)

    def test_batch_normalisation_is_on_by_default(self):
        block = embedding.ConvResidualBlock(channels=8, kernel_size=5)
        self.assertTrue(block.use_batch_norm)
        self.assertEqual(len(block.batch_norm_layers), 2)

    def test_batch_normalisation_can_be_switched_off(self):
        block = embedding.ConvResidualBlock(
            channels=8, kernel_size=5, use_batch_norm=False
        )
        self.assertFalse(hasattr(block, "batch_norm_layers"))

    def test_the_last_convolution_starts_near_zero_so_the_block_is_an_identity(self):
        # A near-identity block at initialisation keeps a deep stack from
        # destroying the input before training starts.
        block = embedding.ConvResidualBlock(channels=4, kernel_size=3)
        self.assertLess(float(block.conv_layers[-1].weight.abs().max()), 1e-3)

    def test_the_initialisation_can_be_left_to_torch(self):
        block = embedding.ConvResidualBlock(
            channels=4, kernel_size=3, zero_initialization=False
        )
        self.assertGreater(float(block.conv_layers[-1].weight.abs().max()), 1e-3)

    def test_the_input_is_added_back_as_a_residual(self):
        block = embedding.ConvResidualBlock(
            channels=4, kernel_size=3, use_batch_norm=False
        ).eval()
        nn.init.zeros_(block.conv_layers[0].weight)
        nn.init.zeros_(block.conv_layers[0].bias)
        nn.init.zeros_(block.conv_layers[1].weight)
        nn.init.zeros_(block.conv_layers[1].bias)
        inputs = torch.randn(2, 4, 16)
        torch.testing.assert_close(block(inputs), inputs)

    def test_dropout_is_applied_during_training_only(self):
        block = embedding.ConvResidualBlock(
            channels=4, kernel_size=3, dropout_probability=0.9
        )
        self.assertAlmostEqual(block.dropout.p, 0.9)


@needs_torch
class TestConvResidualNet(unittest.TestCase):
    def build(self, **kwargs):
        settings = dict(
            in_channels=3,
            out_channels=1,
            hidden_channels=16,
            num_blocks=2,
            kernel_size=5,
        )
        settings.update(kwargs)
        return embedding.ConvResidualNet(**settings).eval()

    def test_the_channels_are_remapped_while_the_time_axis_survives(self):
        network = self.build()
        self.assertEqual(tuple(network(torch.randn(2, 3, 121)).shape), (2, 1, 121))

    def test_one_block_is_built_per_request(self):
        self.assertEqual(len(self.build(num_blocks=4).blocks), 4)

    def test_no_blocks_still_gives_a_working_network(self):
        network = self.build(num_blocks=0)
        self.assertEqual(tuple(network(torch.randn(2, 3, 32)).shape), (2, 1, 32))

    def test_the_hidden_channels_are_used_between_the_outer_layers(self):
        network = self.build(hidden_channels=20)
        self.assertEqual(network.initial_layer.out_channels, 20)
        self.assertEqual(network.final_layer.in_channels, 20)

    def test_the_output_channel_count_is_honoured(self):
        network = self.build(out_channels=5)
        self.assertEqual(tuple(network(torch.randn(2, 3, 32)).shape), (2, 5, 32))

    def test_the_time_axis_length_is_only_fixed_at_call_time(self):
        network = self.build()
        for length in [32, 121]:
            self.assertEqual(network(torch.randn(1, 3, length)).shape[-1], length)


@needs_torch
class TestSimilarityEmbedding(unittest.TestCase):
    """The network trained by the contrastive loss. Its forward pass returns
    both the expander output used for training and the representation used
    downstream as the flow's context."""

    def build(self, **kwargs):
        settings = dict(num_dim=7, num_dim_final=5)
        settings.update(kwargs)
        return embedding.SimilarityEmbedding(**settings).eval()

    def test_both_the_expanded_output_and_the_representation_are_returned(self):
        expanded, representation = self.build()(torch.randn(4, 3, 121))
        self.assertEqual(tuple(expanded.shape), (4, 5))
        self.assertEqual(tuple(representation.shape), (4, 7))

    def test_the_representation_width_follows_the_requested_dimension(self):
        _, representation = self.build(num_dim=3)(torch.randn(2, 3, 121))
        self.assertEqual(representation.shape[-1], 3)

    def test_the_expanded_width_follows_the_requested_final_dimension(self):
        expanded, _ = self.build(num_dim_final=10)(torch.randn(2, 3, 121))
        self.assertEqual(expanded.shape[-1], 10)

    def test_the_backbone_is_the_one_dimensional_resnet(self):
        self.assertIsInstance(self.build().layers_f, ResNet)

    def test_the_representation_is_taken_before_the_expander(self):
        # The expander exists only to shape the contrastive loss, so the
        # representation must not carry its transformation.
        network = self.build()
        inputs = torch.randn(2, 3, 121)
        _, representation = network(inputs)
        contracted = network.contraction_layer(network.layers_f(inputs))
        torch.testing.assert_close(representation, contracted)

    def test_the_representation_is_a_copy_not_a_view(self):
        network = self.build()
        _, representation = network(torch.randn(2, 3, 121))
        self.assertFalse(representation._is_view())

    def test_one_hidden_expander_layer_is_built_per_request(self):
        self.assertEqual(len(self.build(num_hidden_layers_h=3).layers_h), 3)

    def test_the_contraction_layer_reads_the_backbone_context(self):
        network = self.build(num_dim=7)
        self.assertEqual(network.contraction_layer.in_features, 100)
        self.assertEqual(network.contraction_layer.out_features, 7)

    def test_the_output_is_differentiable(self):
        network = self.build()
        expanded, _ = network(torch.randn(2, 3, 121))
        expanded.sum().backward()
        self.assertIsNotNone(network.contraction_layer.weight.grad)

    def test_the_weights_shipped_with_the_package_load_into_the_network(self):
        # em.analysis builds the embedding with exactly these settings
        # before loading the released weights, so the shapes have to match.
        from pathlib import Path

        import nmma

        weights = (
            Path(nmma.__file__).parent / "mlmodel" / "similarity_embedding_weights.pth"
        )
        network = embedding.SimilarityEmbedding(
            num_dim=7,
            num_hidden_layers_f=1,
            num_hidden_layers_h=1,
            num_blocks=4,
            kernel_size=5,
            num_dim_final=5,
        )
        network.load_state_dict(torch.load(weights, map_location="cpu"))


@needs_torch
class TrainingLoopMixin:
    """Both loops walk a data loader of four-tuples, reshape away the repeat
    axis and report a running loss to tensorboard."""

    def batches(self, count, repeats=2, channels=3, points=121):
        parameters = torch.randn(1, repeats, 1, 5)
        light_curves = torch.randn(1, repeats, channels, points)
        return [
            (parameters, parameters.clone(), light_curves, light_curves.clone())
            for _ in range(count)
        ]

    def setUp(self):
        self.writer = MagicMock()
        self.network = embedding.SimilarityEmbedding(num_dim=7, num_dim_final=5)
        self.loss = embedding.VICRegLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=1e-4)


@needs_torch
class TestTrainOneEpoch(TrainingLoopMixin, unittest.TestCase):
    def run_epoch(self, count, **kwargs):
        return embedding.train_one_epoch_se(
            0,
            self.writer,
            self.batches(count),
            self.network,
            self.optimizer,
            kwargs.pop("verbose", False),
            self.loss,
            **kwargs,
        )

    def test_a_loss_is_reported_once_ten_batches_have_been_seen(self):
        self.assertGreater(self.run_epoch(10), 0.0)

    def test_fewer_than_ten_batches_report_no_loss_at_all(self):
        # The running loss is only flushed every tenth batch, so a short
        # epoch returns the initial zero rather than its real loss.
        self.assertEqual(self.run_epoch(9), 0.0)

    def test_the_loss_is_written_to_tensorboard(self):
        self.run_epoch(10)
        self.writer.add_scalar.assert_called_once()
        self.assertEqual(self.writer.add_scalar.call_args.args[0], "SimLoss/train")

    def test_nothing_is_written_for_a_short_epoch(self):
        self.run_epoch(9)
        self.writer.add_scalar.assert_not_called()

    def test_the_weights_are_updated(self):
        before = self.network.contraction_layer.weight.detach().clone()
        self.run_epoch(10)
        self.assertFalse(
            torch.equal(before, self.network.contraction_layer.weight.detach())
        )

    def test_the_loss_weights_are_forwarded(self):
        loss = MagicMock(
            return_value=(
                torch.zeros(1, requires_grad=True),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
            )
        )
        embedding.train_one_epoch_se(
            0,
            self.writer,
            self.batches(1),
            self.network,
            self.optimizer,
            False,
            loss,
            wt_cov=2.0,
        )
        self.assertEqual(loss.call_args.kwargs, {"wt_cov": 2.0})

    def test_the_tensorboard_step_counts_across_epochs(self):
        embedding.train_one_epoch_se(
            3,
            self.writer,
            self.batches(10),
            self.network,
            self.optimizer,
            False,
            self.loss,
        )
        self.assertEqual(self.writer.add_scalar.call_args.args[2], 3 * 10 + 10)

    def test_quiet_mode_prints_nothing(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.run_epoch(10)
        self.assertEqual(stream.getvalue(), "")

    def test_verbose_mode_reports_the_three_loss_terms(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.run_epoch(10, verbose=True)
        self.assertIn("train loss/batch", stream.getvalue())


@needs_torch
class TestValidateOneEpoch(TrainingLoopMixin, unittest.TestCase):
    def run_epoch(self, count):
        return embedding.val_one_epoch_se(
            0, self.writer, self.batches(count), self.network, self.loss
        )

    def test_a_loss_is_reported_after_every_batch(self):
        # Validation flushes on every batch, unlike training.
        self.assertGreater(self.run_epoch(1), 0.0)

    def test_every_batch_is_written_to_tensorboard(self):
        self.run_epoch(3)
        self.assertEqual(self.writer.add_scalar.call_count, 3)
        self.assertEqual(self.writer.add_scalar.call_args.args[0], "SimLoss/val")

    def test_the_weights_are_left_untouched(self):
        before = self.network.contraction_layer.weight.detach().clone()
        self.run_epoch(3)
        torch.testing.assert_close(
            before, self.network.contraction_layer.weight.detach()
        )

    def test_the_writer_is_flushed_so_the_epoch_is_visible(self):
        self.run_epoch(1)
        self.assertTrue(self.writer.flush.called)

    def test_an_empty_loader_reports_no_loss(self):
        self.assertEqual(self.run_epoch(0), 0.0)


if __name__ == "__main__":
    unittest.main()
