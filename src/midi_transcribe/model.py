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


def create_model(
    model_name: str,
    input_dim: int = 128,
    output_dim: int = 88,
    hidden_dim: int = 512,
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
    raise ValueError(f"Unknown model_name: {model_name}")
