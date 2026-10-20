"""Crée un modèle ONNX de même architecture que l'élève, avec des poids aléatoires.

Le temps de calcul ne dépend pas de la valeur des poids : ce modèle sert uniquement
à mesurer la vitesse sur cette machine, sans transférer les 1,7 Go du vrai modèle.

Usage (depuis le dossier notebooks/) :
    uv run --no-project --with torch --with transformers --with onnx --with sentencepiece \
        --with protobuf python modele_vitesse.py
"""
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

ENSEIGNANT = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
SORTIE = Path("models/eleve_emotions/eleve_vitesse.onnx")
SORTIE.parent.mkdir(parents=True, exist_ok=True)

config = AutoConfig.from_pretrained(ENSEIGNANT)          # quelques Ko : l'architecture, pas les poids


class Eleve(nn.Module):
    def __init__(self):
        super().__init__()
        self.encodeur = AutoModel.from_config(config).float()   # même architecture, poids aléatoires
        self.tete = nn.Linear(config.hidden_size, 28)

    def forward(self, input_ids, attention_mask):
        etats = self.encodeur(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        masque = attention_mask.unsqueeze(-1).to(etats.dtype)
        return self.tete((etats * masque).sum(1) / masque.sum(1).clamp(min=1e-6))


modele = Eleve().eval()
ids = torch.randint(0, config.vocab_size, (2, 64))
args = (ids, torch.ones_like(ids))
options = dict(input_names=["input_ids", "attention_mask"], output_names=["logits"],
               dynamic_axes={"input_ids": {0: "lot", 1: "tokens"}, "attention_mask": {0: "lot", 1: "tokens"},
                             "logits": {0: "lot"}},
               opset_version=17)
with torch.no_grad():
    try:
        torch.onnx.export(modele, args, str(SORTIE), dynamo=False, **options)
    except TypeError:
        torch.onnx.export(modele, args, str(SORTIE), **options)
print("Modèle de mesure créé :", SORTIE, f"({SORTIE.stat().st_size / 1e6:.0f} Mo)")
