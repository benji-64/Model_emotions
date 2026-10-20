"""Mesure la vitesse de l'élève sur le CPU de cette machine, un verbatim à la fois.

Usage (depuis le dossier notebooks/) :
    uv run --no-project --with onnxruntime --with transformers --with sentencepiece \
        --with protobuf --with pandas python bench_cpu.py
"""
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer

DOSSIER = Path("models/eleve_emotions")
MODELES = ["eleve.onnx", "eleve_vitesse.onnx", "eleve_int8_partiel.onnx"]           # ceux qui sont absents sont ignorés
VERBATIMS = Path("../data/verbatim_2026_07_31.csv")
N_TEXTES = 30                                                  # verbatims chronométrés par configuration
MAX_LEN = 512

# --- Tokenizer : copie locale si elle existe, sinon celui de l'enseignant (identique)
if (DOSSIER / "tokenizer").exists():
    tokenizer = AutoTokenizer.from_pretrained(DOSSIER / "tokenizer")
else:
    tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

# --- Verbatims réels, tirés au hasard
df = pd.read_csv(VERBATIMS, sep=";", encoding="utf-8-sig")
textes = (df["Titre"].fillna("").astype(str).str.strip() + "\n\n"
          + df["Description"].fillna("").astype(str).str.strip()).str.strip()
textes = textes[textes.str.len() > 0].sample(N_TEXTES + 3, random_state=0).tolist()
longueurs = [len(ids) for ids in tokenizer(textes, truncation=True, max_length=MAX_LEN)["input_ids"]]
print(f"{N_TEXTES} verbatims, longueur médiane {np.median(longueurs):.0f} tokens, max {max(longueurs)}")
print(f"Processeur : {os.cpu_count()} cœurs logiques\n")


def un_verbatim(session, texte):
    enc = tokenizer([texte], truncation=True, max_length=MAX_LEN, return_tensors="np")
    return session.run(None, {"input_ids": enc["input_ids"].astype(np.int64),
                              "attention_mask": enc["attention_mask"].astype(np.int64)})[0]


lignes = []
for nom in MODELES:
    chemin = DOSSIER / nom
    if not chemin.exists():
        continue
    for n_threads in sorted({1, 2, 4, os.cpu_count()}):
        so = ort.SessionOptions()
        so.intra_op_num_threads = n_threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            session = ort.InferenceSession(str(chemin), so, providers=["CPUExecutionProvider"])
        except Exception as e:                                 # fichier incomplet (transfert interrompu) ou absent
            print(f"{nom} ignoré : impossible de le charger ({type(e).__name__})")
            break
        for t in textes[:3]:                                   # échauffement, non compté
            un_verbatim(session, t)
        durees = []
        for t in textes[3:]:
            t0 = time.perf_counter()
            un_verbatim(session, t)
            durees.append(time.perf_counter() - t0)
        d = np.array(durees)
        lignes.append({"modèle": nom, "cœurs": n_threads, "moyenne (s)": d.mean(),
                       "médiane (s)": np.median(d), "95e centile (s)": np.percentile(d, 95),
                       "1 000 verbatims (min)": d.mean() * 1000 / 60})
        print(f"{nom:<26} {n_threads:>2} cœur(s) : {d.mean():.2f} s par verbatim en moyenne")

print()
print(pd.DataFrame(lignes).round(2).to_string(index=False))
