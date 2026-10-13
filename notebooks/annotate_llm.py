"""Annotation des verbatims par un LLM via vLLM — un processus = un modèle = un GPU.

Lancé par notebooks/llm_annotation.ipynb :
    CUDA_VISIBLE_DEVICES=2 python annotate_llm.py --job ../data/annotations/<run>/jobs/<slug>.json

Le module peut aussi être importé (DIMENSIONS, JSON_SCHEMA, parse_scores) sans charger vLLM.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DIMENSIONS = ["satisfaction", "contestation", "desillusion", "detresse", "perplexite"]

JSON_SCHEMA = {
    "type": "object",
    "properties": {dim: {"type": "integer", "enum": [0, 1, 2, 3]} for dim in DIMENSIONS},
    "required": DIMENSIONS,
    "additionalProperties": False,
}


def build_messages(prompt: str, text: str) -> list[dict]:
    # Consigne d'abord, texte à la fin : le préfixe commun (~900 tokens) est mis en cache par vLLM.
    return [{"role": "user", "content": f"{prompt.strip()}\n\n<texte>\n{text}\n</texte>"}]


def parse_scores(raw) -> dict:
    """Extrait les 5 scores d'une réponse. NaN + parse_ok=False si la réponse est inexploitable."""
    empty = {dim: np.nan for dim in DIMENSIONS}
    if not isinstance(raw, str):
        return {**empty, "parse_ok": False}
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    match = re.search(r"\{.*?\}", raw, flags=re.S)
    if not match:
        return {**empty, "parse_ok": False}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {**empty, "parse_ok": False}
    scores, ok = {}, True
    for dim in DIMENSIONS:
        val = obj.get(dim)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and int(val) == val and 0 <= val <= 3:
            scores[dim] = int(val)
        else:
            scores[dim], ok = np.nan, False
    return {**scores, "parse_ok": ok and set(obj) == set(DIMENSIONS)}


def make_sampling_params(cfg: dict):
    from vllm import SamplingParams

    common = dict(temperature=0.0, max_tokens=cfg["max_new_tokens"], seed=cfg["seed"])
    try:  # vLLM récent
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(**common, structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA))
    except ImportError:  # vLLM plus ancien
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(**common, guided_decoding=GuidedDecodingParams(json=JSON_SCHEMA))


def main(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    model, cfg = job["model"], job["config"]
    out_path = Path(job["output"])
    prompt = Path(job["prompt"]).read_text(encoding="utf-8")

    corpus = pd.read_csv(job["corpus"], dtype={"full_comment": str})
    if out_path.exists():
        done = set(pd.read_csv(out_path, usecols=["id_experience"])["id_experience"])
        corpus = corpus[~corpus["id_experience"].isin(done)]
    print(f"[{model['slug']}] {len(corpus)} textes à annoter", flush=True)
    if corpus.empty:
        return

    from vllm import LLM

    llm_kwargs = dict(
        model=model["id"],
        dtype="bfloat16",
        max_model_len=cfg["max_model_len"],
        tensor_parallel_size=1,
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        enable_prefix_caching=True,
        seed=cfg["seed"],
        trust_remote_code=True,
    )
    if model.get("multimodal"):
        # Texte seul : pas d'images -> l'encodeur de vision n'est pas profilé, VRAM libérée pour le cache KV
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0}
    llm_kwargs.update(job.get("vllm_kwargs") or {})
    llm = LLM(**llm_kwargs)
    params = make_sampling_params(cfg)
    chat_kwargs = job.get("chat_template_kwargs")

    t_start, n_done = time.time(), 0
    for start in range(0, len(corpus), cfg["chunk_size"]):
        chunk = corpus.iloc[start:start + cfg["chunk_size"]]
        msgs = [build_messages(prompt, t) for t in chunk["full_comment"]]
        kwargs = {"sampling_params": params, "use_tqdm": False}
        if chat_kwargs:
            kwargs["chat_template_kwargs"] = chat_kwargs
        outputs = llm.chat(msgs, **kwargs)

        raws = [o.outputs[0].text for o in outputs]
        rows = pd.DataFrame([parse_scores(r) for r in raws])
        rows.insert(0, "id_experience", chunk["id_experience"].to_numpy())
        rows["raw_output"] = raws
        rows["model"] = model["id"]
        rows.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)

        n_done += len(chunk)
        rate = n_done / (time.time() - t_start)
        eta_h = (len(corpus) - n_done) / rate / 3600
        print(f"[{model['slug']}] {n_done}/{len(corpus)} — {rate:.1f} textes/s — "
              f"reste ~{eta_h:.1f} h — échecs de parsing (bloc) : {(~rows['parse_ok']).sum()}", flush=True)

    print(f"[{model['slug']}] TERMINÉ en {(time.time() - t_start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    sys.exit(main(parser.parse_args().job))
