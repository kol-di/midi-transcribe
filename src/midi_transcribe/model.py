import torch
import torch.nn as nn


class TemporalMLP(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 512, output_dim: int = 88) -> None:
        super().__init__()
        self.input_dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop2 = nn.Dropout(p=0.25)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.drop3 = nn.Dropout(p=0.25)

        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> apply MLP per frame by flattening batch*time
        bsz, timesteps, feat = x.shape
        z = x.reshape(bsz * timesteps, feat)

        z = self.input_dropout(z)

        z = self.fc1(z)
        z = self.bn1(z)
        z = self.act(z)
        z = self.drop1(z)

        z = self.fc2(z)
        z = self.bn2(z)
        z = self.act(z)
        z = self.drop2(z)

        z = self.fc3(z)
        z = self.bn3(z)
        z = self.act(z)
        z = self.drop3(z)

        z = self.out(z)
        return z.reshape(bsz, timesteps, -1)  # [B, T, 88]


class ConvContextModel(nn.Module):
    """Frame-wise predictor with local 5-frame context over Mel bins."""

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 88,
        context_frames: int = 5,
        dense_dim: int = 512,
    ) -> None:
        super().__init__()
        if context_frames % 2 == 0:
            raise ValueError("context_frames must be odd to keep centered context.")

        self.context_frames = context_frames
        self.time_pad = context_frames // 2

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.drop1 = nn.Dropout(p=0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.drop2 = nn.Dropout(p=0.25)

        reduced_mels = input_dim // 4
        flattened = 64 * context_frames * reduced_mels
        self.fc1 = nn.Linear(flattened, dense_dim)
        self.drop3 = nn.Dropout(p=0.5)
        self.fc_out = nn.Linear(dense_dim, output_dim)

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        bsz, timesteps, feats = x.shape

        # Build a centered local context for each frame: [B, T, K, F]
        x_pad = x.transpose(1, 2)  # [B, F, T]
        x_pad = nn.functional.pad(x_pad, (self.time_pad, self.time_pad), mode="replicate")
        x_ctx = x_pad.unfold(dimension=2, size=self.context_frames, step=1)  # [B, F, T, K]
        x_ctx = x_ctx.permute(0, 2, 3, 1).contiguous()  # [B, T, K, F]

        # Apply 2D conv stack per center frame: [B*T, 1, K, F]
        z = x_ctx.reshape(bsz * timesteps, 1, self.context_frames, feats)
        z = self.act(self.conv1(z))
        z = self.act(self.bn2(self.conv2(z)))
        z = self.pool1(z)
        z = self.drop1(z)

        z = self.act(self.conv3(z))
        z = self.pool2(z)
        z = self.drop2(z)

        z = z.flatten(start_dim=1)
        z = self.act(self.fc1(z))
        z = self.drop3(z)
        z = self.fc_out(z)
        return z.reshape(bsz, timesteps, -1)  # [B, T, 88]


class CNNFrequencyEncoder(nn.Module):
    """
    Shared CNN encoder:
    input  [B, 1, F, T]
    output [B, C, F_reduced, T]

    Downsamples only frequency axis, preserves temporal resolution.
    """

    def __init__(self, hidden_channels: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(64, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNOnsetFrameBaseline(nn.Module):
    """Compact onset+frame CNN baseline that preserves temporal resolution."""

    def __init__(
        self,
        out_keys: int = 88,
        hidden_channels: int = 96,
        pooled_freq_bands: int = 8,
        head_hidden: int = 256,
        dropout: float = 0.2,
        detach_onset_for_frame: bool = True,
        use_temporal_convs: bool = False,
    ) -> None:
        super().__init__()
        self.pooled_freq_bands = pooled_freq_bands
        self.detach_onset_for_frame = detach_onset_for_frame
        self.use_temporal_convs = use_temporal_convs

        self.encoder = CNNFrequencyEncoder(hidden_channels=hidden_channels)

        self.temporal_convs = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3),
                padding=(0, 1),
                dilation=(1, 1),
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3),
                padding=(0, 2),
                dilation=(1, 2),
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3),
                padding=(0, 4),
                dilation=(1, 4),
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        feature_dim = hidden_channels * pooled_freq_bands
        self.onset_head = nn.Sequential(
            nn.Linear(feature_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_keys),
        )
        self.frame_head = nn.Sequential(
            nn.Linear(feature_dim + out_keys, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_keys),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, T, F] -> [B, 1, F, T]
        x = x.transpose(1, 2).unsqueeze(1)

        h = self.encoder(x)  # [B, C, F_reduced, T]
        if self.use_temporal_convs:
            h = self.temporal_convs(h)
        h = nn.functional.adaptive_avg_pool2d(h, (self.pooled_freq_bands, h.shape[-1]))
        h = h.permute(0, 3, 1, 2).contiguous()  # [B, T, C, F_reduced]
        features = h.flatten(start_dim=2)  # [B, T, D]

        onset_logits = self.onset_head(features)
        onset_probs = torch.sigmoid(onset_logits)
        if self.detach_onset_for_frame:
            onset_probs = onset_probs.detach()

        frame_input = torch.cat([features, onset_probs], dim=-1)
        frame_logits = self.frame_head(frame_input)
        return {
            "onset_logits": onset_logits,
            "frame_logits": frame_logits,
        }


class CRNNOnsetFrameBaseline(nn.Module):
    """
    Compact CRNN onset+frame baseline.

    input [B, T, F] -> {
        "onset_logits": [B, T, 88],
        "frame_logits": [B, T, 88],
    }

    The architecture is intentionally close to CNNOnsetFrameBaseline:
    CNN encoder -> frequency pooling -> features [B, T, D] -> LSTM/GRU -> heads.

    This allows a clean comparison:
    - CNNOnsetFrameBaseline: no recurrent temporal modeling
    - CRNNOnsetFrameBaseline: same CNN features + recurrent temporal modeling
    """

    def __init__(
        self,
        out_keys: int = 88,
        hidden_channels: int = 96,
        pooled_freq_bands: int = 8,
        head_hidden: int = 256,
        dropout: float = 0.2,
        detach_onset_for_frame: bool = True,
        rnn_type: str = "lstm",  # "lstm" or "gru"
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        rnn_bidirectional: bool = True,
    ) -> None:
        super().__init__()

        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got {rnn_type}")

        self.pooled_freq_bands = pooled_freq_bands
        self.detach_onset_for_frame = detach_onset_for_frame
        self.rnn_type = rnn_type
        self.rnn_bidirectional = rnn_bidirectional

        self.encoder = CNNFrequencyEncoder(hidden_channels=hidden_channels)

        feature_dim = hidden_channels * pooled_freq_bands

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=feature_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=rnn_bidirectional,
            dropout=dropout if rnn_num_layers > 1 else 0.0,
        )

        rnn_output_dim = rnn_hidden_size * (2 if rnn_bidirectional else 1)

        self.onset_head = nn.Sequential(
            nn.Linear(rnn_output_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_keys),
        )

        self.frame_head = nn.Sequential(
            nn.Linear(rnn_output_dim + out_keys, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_keys),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, T, F] -> [B, 1, F, T]
        x = x.transpose(1, 2).unsqueeze(1)

        h = self.encoder(x)  # [B, C, F_reduced, T]
        h = nn.functional.adaptive_avg_pool2d(
            h,
            (self.pooled_freq_bands, h.shape[-1]),
        )  # [B, C, pooled_freq_bands, T]
        h = h.permute(0, 3, 1, 2).contiguous()  # [B, T, C, F_reduced]
        features = h.flatten(start_dim=2)  # [B, T, D]

        rnn_features, _ = self.rnn(features)  # [B, T, H] or [B, T, 2H]

        onset_logits = self.onset_head(rnn_features)  # [B, T, 88]
        onset_probs = torch.sigmoid(onset_logits)
        if self.detach_onset_for_frame:
            onset_probs = onset_probs.detach()

        frame_input = torch.cat([rnn_features, onset_probs], dim=-1)
        frame_logits = self.frame_head(frame_input)  # [B, T, 88]

        return {
            "onset_logits": onset_logits,
            "frame_logits": frame_logits,
        }


def create_model(
    model_name: str,
    input_dim: int = 128,
    output_dim: int = 88,
    hidden_dim: int = 512,
    use_temporal_convs: bool = False,
    rnn_type: str = "lstm",
    rnn_hidden_size: int = 128,
    rnn_num_layers: int = 1,
    rnn_bidirectional: bool = True,
) -> nn.Module:
    if model_name == "mlp_baseline":
        return TemporalMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    if model_name == "cnn_context5":
        return ConvContextModel(
            input_dim=input_dim,
            output_dim=output_dim,
            context_frames=5,
            dense_dim=hidden_dim,
        )
    if model_name == "cnn_onsets_frames":
        return CNNOnsetFrameBaseline(
            out_keys=output_dim,
            hidden_channels=96,
            pooled_freq_bands=8,
            head_hidden=hidden_dim,
            dropout=0.2,
            detach_onset_for_frame=True,
            use_temporal_convs=use_temporal_convs,
        )
    if model_name == "crnn_onsets_frames":
        return CRNNOnsetFrameBaseline(
            out_keys=output_dim,
            hidden_channels=96,
            pooled_freq_bands=8,
            head_hidden=hidden_dim,
            dropout=0.2,
            detach_onset_for_frame=True,
            rnn_type=rnn_type,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            rnn_bidirectional=rnn_bidirectional,
        )
    raise ValueError(f"Unknown model_name: {model_name}")
