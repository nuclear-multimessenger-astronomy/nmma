import unittest

try:
    import torch
    from torch import nn

    from nmma.mlmodel import resnet

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is the optional neuralnet extra
    TORCH_AVAILABLE = False

needs_torch = unittest.skipUnless(
    TORCH_AVAILABLE, "torch is not installed; install the neuralnet extra"
)


@needs_torch
class TestChannelNorm(unittest.TestCase):
    """Normalises each channel over the time axis, folding the statistics
    into the learnable scale and shift so the long time axis is only walked
    once."""

    def test_the_group_count_defaults_to_one_group_per_channel(self):
        layer = resnet.ChannelNorm(num_channels=8)
        self.assertEqual(layer.num_groups, 8)
        self.assertEqual(layer.channels_per_group, 1)

    def test_a_group_count_that_does_not_divide_the_channels_is_refused(self):
        with self.assertRaises(ValueError):
            resnet.ChannelNorm(num_channels=8, num_groups=3)

    def test_a_group_count_that_divides_the_channels_is_accepted(self):
        layer = resnet.ChannelNorm(num_channels=8, num_groups=4)
        self.assertEqual(layer.channels_per_group, 2)

    def test_the_scale_and_shift_are_learnable_and_shaped_per_channel(self):
        layer = resnet.ChannelNorm(num_channels=5)
        self.assertEqual(tuple(layer.weight.shape), (5, 1))
        self.assertEqual(tuple(layer.bias.shape), (5, 1))
        self.assertTrue(layer.weight.requires_grad)
        self.assertTrue(layer.bias.requires_grad)

    def test_the_scale_starts_at_one_and_the_shift_at_zero(self):
        layer = resnet.ChannelNorm(num_channels=3)
        torch.testing.assert_close(layer.weight, torch.ones(3, 1))
        torch.testing.assert_close(layer.bias, torch.zeros(3, 1))

    def test_the_shape_is_preserved(self):
        layer = resnet.ChannelNorm(num_channels=4)
        output = layer(torch.randn(2, 4, 16))
        self.assertEqual(tuple(output.shape), (2, 4, 16))

    def test_each_channel_is_centred_on_zero(self):
        layer = resnet.ChannelNorm(num_channels=4)
        output = layer(torch.randn(2, 4, 64))
        torch.testing.assert_close(
            output.mean(-1), torch.zeros(2, 4), atol=1e-5, rtol=0
        )

    def test_each_channel_is_scaled_to_unit_variance(self):
        layer = resnet.ChannelNorm(num_channels=4)
        output = layer(torch.randn(2, 4, 256))
        torch.testing.assert_close(
            output.std(-1, unbiased=False), torch.ones(2, 4), atol=1e-3, rtol=0
        )

    def test_a_constant_channel_does_not_divide_by_zero(self):
        # The epsilon in the variance keeps a flat light curve, which a
        # padded non-detection produces, from blowing up.
        layer = resnet.ChannelNorm(num_channels=2)
        output = layer(torch.ones(1, 2, 10))
        self.assertTrue(torch.isfinite(output).all())

    def test_grouped_normalisation_shares_statistics_within_a_group(self):
        layer = resnet.ChannelNorm(num_channels=4, num_groups=2)
        inputs = torch.randn(3, 4, 32)
        output = layer(inputs)
        self.assertEqual(tuple(output.shape), (3, 4, 32))
        self.assertTrue(torch.isfinite(output).all())

    def test_grouped_normalisation_centres_each_group_not_each_channel(self):
        layer = resnet.ChannelNorm(num_channels=4, num_groups=2)
        output = layer(torch.randn(1, 4, 128))
        group_means = output.reshape(1, 2, 2, 128).mean(dim=(2, 3))
        torch.testing.assert_close(group_means, torch.zeros(1, 2), atol=1e-4, rtol=0)


@needs_torch
class TestGetNormLayer(unittest.TestCase):
    """The network is built from a factory so every block normalises the
    same way, while only needing the channel count at construction."""

    def test_a_class_is_returned_that_only_needs_the_channel_count(self):
        layer_class = resnet.get_norm_layer()
        layer = layer_class(6)
        self.assertIsInstance(layer, resnet.ChannelNorm)

    def test_without_a_group_count_every_channel_gets_its_own_group(self):
        layer = resnet.get_norm_layer()(6)
        self.assertEqual(layer.num_groups, 6)

    def test_a_group_count_is_capped_at_the_channel_count(self):
        # A block with fewer channels than the requested groups would
        # otherwise fail the divisibility check.
        layer = resnet.get_norm_layer(groups=8)(4)
        self.assertEqual(layer.num_groups, 4)

    def test_a_group_count_below_the_channel_count_is_used_as_given(self):
        layer = resnet.get_norm_layer(groups=2)(8)
        self.assertEqual(layer.num_groups, 2)

    def test_each_call_gives_an_independent_class(self):
        self.assertIsNot(resnet.get_norm_layer(2), resnet.get_norm_layer(2))


@needs_torch
class TestConvolutions(unittest.TestCase):
    """The convolution helpers pad so the time axis survives unchanged, and
    carry no bias because a normalisation layer always follows."""

    def test_an_even_kernel_is_refused(self):
        # An even kernel cannot be padded symmetrically, so the time axis
        # would silently shift.
        with self.assertRaises(ValueError):
            resnet.convN(3, 8, kernel_size=4)

    def test_an_odd_kernel_preserves_the_time_axis(self):
        layer = resnet.convN(3, 8, kernel_size=7)
        self.assertEqual(tuple(layer(torch.randn(2, 3, 64)).shape), (2, 8, 64))

    def test_the_padding_follows_the_kernel_size(self):
        self.assertEqual(resnet.convN(3, 8, kernel_size=5).padding, (2,))

    def test_the_padding_grows_with_the_dilation(self):
        layer = resnet.convN(3, 8, kernel_size=5, dilation=3)
        self.assertEqual(layer.padding, (6,))

    def test_a_dilated_convolution_still_preserves_the_time_axis(self):
        layer = resnet.convN(3, 8, kernel_size=5, dilation=3)
        self.assertEqual(tuple(layer(torch.randn(2, 3, 64)).shape), (2, 8, 64))

    def test_a_stride_of_two_halves_the_time_axis(self):
        layer = resnet.convN(3, 8, kernel_size=3, stride=2)
        self.assertEqual(tuple(layer(torch.randn(2, 3, 64)).shape), (2, 8, 32))

    def test_no_bias_is_learned_because_a_norm_layer_follows(self):
        self.assertIsNone(resnet.convN(3, 8).bias)
        self.assertIsNone(resnet.conv1(3, 8).bias)

    def test_the_single_point_convolution_only_remaps_channels(self):
        layer = resnet.conv1(3, 8)
        self.assertEqual(layer.kernel_size, (1,))
        self.assertEqual(tuple(layer(torch.randn(2, 3, 64)).shape), (2, 8, 64))

    def test_the_single_point_convolution_can_downsample(self):
        layer = resnet.conv1(3, 8, stride=2)
        self.assertEqual(tuple(layer(torch.randn(2, 3, 64)).shape), (2, 8, 32))

    def test_grouped_convolutions_are_supported(self):
        layer = resnet.convN(4, 8, kernel_size=3, groups=2)
        self.assertEqual(layer.groups, 2)


@needs_torch
class TestBasicBlock(unittest.TestCase):
    def test_the_block_does_not_change_the_feature_map_count(self):
        self.assertEqual(resnet.BasicBlock.expansion, 1)

    def test_the_shape_is_preserved_without_downsampling(self):
        block = resnet.BasicBlock(8, 8).eval()
        self.assertEqual(tuple(block(torch.randn(2, 8, 32)).shape), (2, 8, 32))

    def test_the_input_is_added_back_as_a_residual(self):
        # With both convolutions zeroed the block has to be the identity,
        # up to the rectifier.
        block = resnet.BasicBlock(4, 4, norm_layer=nn.Identity).eval()
        nn.init.zeros_(block.conv1.weight)
        nn.init.zeros_(block.conv2.weight)
        inputs = torch.rand(2, 4, 16)
        torch.testing.assert_close(block(inputs), inputs)

    def test_a_downsample_path_is_used_for_the_residual(self):
        downsample = nn.Sequential(resnet.conv1(4, 8, 2), nn.Identity())
        block = resnet.BasicBlock(4, 8, stride=2, downsample=downsample).eval()
        self.assertEqual(tuple(block(torch.randn(2, 4, 32)).shape), (2, 8, 16))

    def test_batch_normalisation_is_the_default(self):
        self.assertIsInstance(resnet.BasicBlock(8, 8).bn1, nn.BatchNorm1d)

    def test_a_custom_normalisation_layer_is_used(self):
        block = resnet.BasicBlock(8, 8, norm_layer=resnet.get_norm_layer())
        self.assertIsInstance(block.bn1, resnet.ChannelNorm)

    def test_grouped_convolutions_are_refused(self):
        with self.assertRaises(ValueError):
            resnet.BasicBlock(8, 8, groups=2)

    def test_a_non_default_width_is_refused(self):
        with self.assertRaises(ValueError):
            resnet.BasicBlock(8, 8, base_width=32)

    def test_dilation_is_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            resnet.BasicBlock(8, 8, dilation=2)

    def test_a_dilation_of_one_is_accepted(self):
        self.assertIsInstance(resnet.BasicBlock(8, 8, dilation=1), nn.Module)


@needs_torch
class TestBottleneck(unittest.TestCase):
    """The bottleneck squeezes the feature maps, convolves, then expands
    them back out by a factor of four."""

    def test_the_block_expands_the_feature_maps_fourfold(self):
        self.assertEqual(resnet.Bottleneck.expansion, 4)

    def test_the_output_carries_the_expanded_feature_maps(self):
        downsample = nn.Sequential(resnet.conv1(8, 32), nn.Identity())
        block = resnet.Bottleneck(8, 8, downsample=downsample).eval()
        self.assertEqual(tuple(block(torch.randn(2, 8, 32)).shape), (2, 32, 32))

    def test_the_squeeze_keeps_the_requested_width_by_default(self):
        block = resnet.Bottleneck(32, 8)
        self.assertEqual(block.conv1.out_channels, 8)

    def test_the_width_scales_with_the_group_width(self):
        block = resnet.Bottleneck(32, 8, base_width=128)
        self.assertEqual(block.conv1.out_channels, 16)

    def test_the_expansion_happens_in_a_single_point_convolution(self):
        block = resnet.Bottleneck(8, 8)
        self.assertEqual(block.conv3.kernel_size, (1,))
        self.assertEqual(block.conv3.out_channels, 32)

    def test_the_time_axis_is_downsampled_in_the_middle_convolution(self):
        downsample = nn.Sequential(resnet.conv1(8, 32, 2), nn.Identity())
        block = resnet.Bottleneck(8, 8, stride=2, downsample=downsample).eval()
        self.assertEqual(tuple(block(torch.randn(2, 8, 32)).shape), (2, 32, 16))

    def test_grouped_convolutions_are_allowed_unlike_the_basic_block(self):
        block = resnet.Bottleneck(32, 8, groups=2)
        self.assertEqual(block.conv2.groups, 2)


@needs_torch
class ResNetMixin:
    network_class = None

    def build(self, **kwargs):
        settings = dict(
            num_ifos=[3, None], layers=[2, 2], kernel_size=5, context_dim=10
        )
        settings.update(kwargs)
        return self.network_class(**settings).eval()


@needs_torch
class TestResNet(ResNetMixin, unittest.TestCase):
    """The embedding backbone: a 1D ResNet that maps a multi-band light
    curve onto a fixed-length context vector."""

    network_class = resnet.ResNet if TORCH_AVAILABLE else None

    def test_the_light_curve_is_mapped_onto_the_context_vector(self):
        network = self.build(context_dim=100)
        self.assertEqual(tuple(network(torch.randn(4, 3, 121)).shape), (4, 100))

    def test_the_channel_count_is_taken_from_the_first_detector_entry(self):
        # The second entry exists only because other embeddings are
        # constructed from a number of detectors and a strain length.
        network = self.build(num_ifos=[2, 4096])
        self.assertEqual(network.conv1.in_channels, 2)

    def test_the_time_axis_length_is_only_fixed_at_call_time(self):
        network = self.build()
        for length in [64, 121, 256]:
            self.assertEqual(
                tuple(network(torch.randn(2, 3, length)).shape), (2, 10), msg=length
            )

    def test_one_residual_layer_is_built_per_entry_in_layers(self):
        network = self.build(layers=[2, 2, 2])
        self.assertEqual(len(network.residual_layers), 3)

    def test_each_residual_layer_holds_the_requested_number_of_blocks(self):
        network = self.build(layers=[3, 1])
        self.assertEqual(len(network.residual_layers[0]), 3)
        self.assertEqual(len(network.residual_layers[1]), 1)

    def test_the_feature_maps_double_at_every_layer_after_the_first(self):
        network = self.build(layers=[1, 1, 1])
        self.assertEqual(network.fc.in_features, 64 * 2**2)

    def test_striding_is_the_default_downsampling_for_every_later_layer(self):
        network = self.build(layers=[1, 1, 1])
        self.assertEqual(network.dilation, 1)

    def test_a_stride_type_must_be_given_for_every_layer_after_the_first(self):
        with self.assertRaises(ValueError):
            self.build(layers=[2, 2, 2], stride_type=["stride"])

    def test_a_stride_type_of_the_right_length_is_accepted(self):
        network = self.build(layers=[2, 2, 2], stride_type=["stride", "stride"])
        self.assertEqual(len(network.residual_layers), 3)

    def test_dilation_can_replace_striding_for_single_block_layers(self):
        network = self.build(layers=[1, 1], stride_type=["dilation"])
        self.assertEqual(network.dilation, 2)

    def test_dilation_cannot_be_used_with_multi_block_layers(self):
        # _make_layer passes the updated dilation to every block after the
        # first, and BasicBlock refuses any dilation above one. Dilated
        # downsampling is therefore only reachable here when each layer
        # holds a single block; the bottleneck network has no such limit.
        with self.assertRaises(NotImplementedError):
            self.build(layers=[2, 2], stride_type=["dilation"])

    def test_an_unknown_stride_type_is_refused(self):
        with self.assertRaises(ValueError):
            self.build(layers=[2, 2], stride_type=["shuffle"])

    def test_the_first_layer_does_not_downsample_the_time_axis(self):
        network = self.build(layers=[1, 1])
        first_block = network.residual_layers[0][0]
        self.assertEqual(first_block.stride, 1)

    def test_the_later_layers_downsample_the_time_axis(self):
        network = self.build(layers=[1, 1])
        self.assertEqual(network.residual_layers[1][0].stride, 2)

    def test_the_time_axis_is_pooled_away_before_the_output_layer(self):
        network = self.build()
        self.assertIsInstance(network.avgpool, nn.AdaptiveAvgPool1d)

    def test_the_convolutions_are_initialised_for_the_rectifier(self):
        # Kaiming initialisation keeps the activations from collapsing
        # through a deep stack of rectified layers.
        network = self.build()
        self.assertNotEqual(float(network.conv1.weight.std()), 0.0)

    def test_the_residual_branches_can_start_as_identities(self):
        network = self.build(zero_init_residual=True)
        for block in network.residual_layers[0]:
            torch.testing.assert_close(
                block.bn2.weight, torch.zeros_like(block.bn2.weight)
            )

    def test_the_residual_branches_are_not_zeroed_by_default(self):
        network = self.build()
        block = network.residual_layers[0][0]
        self.assertNotEqual(float(block.bn2.weight.abs().sum()), 0.0)

    def test_the_normalisation_groups_reach_every_block(self):
        network = self.build(norm_groups=2)
        self.assertEqual(network.bn1.num_groups, 2)

    def test_the_network_is_built_from_basic_blocks(self):
        self.assertIs(resnet.ResNet.block, resnet.BasicBlock)

    def test_the_output_is_differentiable(self):
        network = self.build()
        output = network(torch.randn(2, 3, 121))
        output.sum().backward()
        self.assertIsNotNone(network.conv1.weight.grad)


@needs_torch
class TestBottleneckResNet(ResNetMixin, unittest.TestCase):
    network_class = resnet.BottleneckResNet if TORCH_AVAILABLE else None

    def test_the_network_is_built_from_bottleneck_blocks(self):
        self.assertIs(resnet.BottleneckResNet.block, resnet.Bottleneck)

    def test_it_still_maps_onto_the_context_vector(self):
        network = self.build(context_dim=7)
        self.assertEqual(tuple(network(torch.randn(2, 3, 121)).shape), (2, 7))

    def test_the_output_layer_accounts_for_the_fourfold_expansion(self):
        network = self.build(layers=[1, 1])
        self.assertEqual(network.fc.in_features, 64 * 2 * 4)

    def test_grouped_convolutions_are_accepted_unlike_the_basic_network(self):
        network = self.build(groups=2, width_per_group=64)
        self.assertEqual(tuple(network(torch.randn(2, 3, 64)).shape), (2, 10))

    def test_the_residual_branches_can_start_as_identities(self):
        network = self.build(zero_init_residual=True)
        for block in network.residual_layers[0]:
            torch.testing.assert_close(
                block.bn3.weight, torch.zeros_like(block.bn3.weight)
            )

    def test_dilation_works_with_multi_block_layers(self):
        network = self.build(layers=[2, 2], stride_type=["dilation"])
        self.assertEqual(network.dilation, 2)


if __name__ == "__main__":
    unittest.main()
