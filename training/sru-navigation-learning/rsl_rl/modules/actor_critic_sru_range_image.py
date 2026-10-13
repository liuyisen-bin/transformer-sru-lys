"""Actor-Critic with a range-image CNN and SRU memory."""

from __future__ import annotations

from rsl_rl.modules.actor_critic_sru_pointnet import ActorCriticSRUPointNet
from rsl_rl.networks import RangeImageEncoder


class ActorCriticSRURangeImage(ActorCriticSRUPointNet):
    """Range Image CNN -> proprioceptive fusion -> SRU actor-critic."""

    perception_encoder_name = "Range Image CNN"

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_points: int = 656,
        point_dim: int = 3,
        range_image_shape: tuple[int, int] = (16, 41),
        range_image_feature_dim: int = 128,
        range_image_batch_size: int = 256,
        range_image_max_distance: float = 5.0,
        transformer_num_layers: int = 0,
        transformer_num_heads: int = 4,
        transformer_feedforward_dim: int = 512,
        transformer_dropout: float = 0.1,
        **kwargs,
    ):
        # These inherited PointNet fields are present in the base config but do
        # not control the range-image encoder.
        kwargs.pop("pointnet_feature_dim", None)
        kwargs.pop("pointnet_batch_size", None)

        image_height, image_width = range_image_shape
        if image_height * image_width != num_points:
            raise ValueError(
                f"Range image shape {range_image_shape} contains "
                f"{image_height * image_width} points, expected {num_points}."
            )

        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            num_points=num_points,
            point_dim=point_dim,
            pointnet_feature_dim=range_image_feature_dim,
            pointnet_batch_size=range_image_batch_size,
            **kwargs,
        )

        self.actor_pointnet = RangeImageEncoder(
            point_dim=point_dim,
            image_height=image_height,
            image_width=image_width,
            feature_dim=range_image_feature_dim,
            max_distance=range_image_max_distance,
            max_batch_size=range_image_batch_size,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_feedforward_dim=transformer_feedforward_dim,
            transformer_dropout=transformer_dropout,
        )
        self.critic_pointnet = RangeImageEncoder(
            point_dim=point_dim,
            image_height=image_height,
            image_width=image_width,
            feature_dim=range_image_feature_dim,
            max_distance=range_image_max_distance,
            max_batch_size=range_image_batch_size,
            transformer_num_layers=transformer_num_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_feedforward_dim=transformer_feedforward_dim,
            transformer_dropout=transformer_dropout,
        )
        self.range_image_shape = range_image_shape
        self.range_image_max_distance = range_image_max_distance
        self._pointnet_training_stage = "full"

        print(
            "[ActorCriticSRURangeImage] "
            f"image=(XYZ+range)x{image_height}x{image_width}, "
            f"feature={range_image_feature_dim}, transformer_layers={transformer_num_layers}, "
            f"heads={transformer_num_heads}"
        )
