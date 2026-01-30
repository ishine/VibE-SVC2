import torch
import torchaudio

from vencoder.encoder import SpeechEncoder


class HuBERTLarge(SpeechEncoder):
    def __init__(self, device=None):
        super().__init__()
        bundle = torchaudio.pipelines.HUBERT_LARGE
        self.hubert = bundle.get_model()
        self.hubert.eval()

        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        self.hidden_dim = 1024
        self.model = self.hubert.to(self.dev)

    def encoder(self, wav):
        with torch.no_grad():
            with torch.inference_mode():
                units, _ = self.hubert.extract_features(
                    wav.view(1, -1),
                    lengths=torch.Tensor([wav.size(0)]).to(self.dev),
                    num_layers=18,
                )
                units = units[-1].squeeze()
                if units.dim() < 3:
                    units = units.unsqueeze(0)

                return units.squeeze(0)
