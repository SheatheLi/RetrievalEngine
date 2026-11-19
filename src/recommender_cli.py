"""Reproducible research pipeline. Run with python -m src.recommender_cli."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import json
import platform
import time
from hashlib import sha256
import numpy as np
import torch
from .movielens_data import load, download
from .recommender import Engine, evaluate, fit_retriever, fit_ranker, interaction_statistics, metrics
from .history_retrieval import fit_history_encoder


def train(data_path: Path, destination: Path, seed: int, epochs: int):
    torch.set_num_threads(1)
    data = load(data_path)
    start = time.perf_counter()
    model, losses = fit_retriever(data, seed=seed, epochs=epochs)
    engine = Engine.from_training(data, model)
    engine.ranker = fit_ranker(data, engine, seed)
    history_model, history_losses, history_examples = fit_history_encoder(
        data.train, data.catalog, data.items, seed=seed, epochs=max(1, min(epochs, 8)))
    engine.attach_history_encoder(history_model)
    development = np.concatenate((data.train, data.rank_train))
    simple_model, simple_losses = fit_retriever(replace(data, train=development), seed=seed, epochs=epochs)
    engine.baseline_user_vectors = simple_model.users.weight.detach().numpy().copy()
    engine.baseline_item_vectors = simple_model.items.weight.detach().numpy().copy()
    knn_validations = []
    cooccurrence = engine.interaction.T @ engine.interaction
    norms = np.sqrt(np.maximum(engine.popularity, 1))
    cosine = cooccurrence / (norms[:, None] * norms[None, :])
    np.fill_diagonal(cosine, 0.)
    best_knn = None
    for neighbors in (20, 50, 100):
        for shrink in (0, 10, 20):
            sim = cosine.copy()
            if shrink: sim *= np.minimum(cooccurrence / shrink, 1.)
            cutoff = np.partition(sim, -neighbors, axis=1)[:, -neighbors]
            sim[sim < cutoff[:, None]] = 0.
            engine.baseline_similarity = sim
            ndcg = np.mean([metrics(engine.recommend(u, seen, method="item_knn")[0], relevant)[1]
                            for u, seen, relevant in data.queries("validation")])
            knn_validations.append({"neighbors": neighbors, "shrink": shrink, "ndcg10": float(ndcg)})
            if best_knn is None or ndcg > best_knn[0]: best_knn = (ndcg, sim.copy())
    engine.baseline_similarity = best_knn[1]
    # Parameter selection reads validation only. The test split has been viewed
    # during development, so this is exploratory evidence, not a blind holdout.
    validations = []
    for ef in (16, 32, 64):
        engine.hnsw.hnsw.efSearch = ef
        for budget in (50, 100, 200):
            result = evaluate(data, engine, "validation", budget)
            validations.append({"ef_search": ef, "budget": budget,
                                "ndcg10": result["methods"]["hnsw_ranker"]["ndcg10"],
                                "candidate_recall": result["candidate_recall"]})
    selected = max(validations, key=lambda x: (x["ndcg10"], -x["budget"], -x["ef_search"]))
    engine.ef_search = selected["ef_search"]
    engine.hnsw.hnsw.efSearch = engine.ef_search
    elapsed = time.perf_counter() - start
    manifest = engine.save(destination, {"seed": seed, "epochs": epochs,
        "input_sha256": data.input_sha256, "cutoffs_unix": data.cutoffs,
        "counts": dict(zip(("retriever_train", "ranker_train", "validation", "test"),
                           [len(data.train), len(data.rank_train), len(data.validation), len(data.test)])),
        "catalog": len(data.catalog), "dimension": 48, "selected": selected,
        "training_seconds": elapsed, "training_loss": losses, "mf80_training_loss": simple_losses,
        "history_encoder_training_loss": history_losses, "history_encoder_examples": history_examples,
        "knn_validation_selection": knn_validations,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {p: version(p) for p in ("torch", "numpy", "faiss-cpu", "lightgbm")},
        "platform": platform.platform(),
        "evaluation_status": "EXPLORATORY_TEST_SPLIT_HAS_BEEN_VIEWED_DURING_DEVELOPMENT",
        "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        "data_use": "GroupLens MovieLens 100K, research-only; raw data excluded from delivery"})
    tests = [evaluate(data, engine, "test", budget) for budget in (50, 100, 200)]
    (destination / "results.json").write_text(json.dumps({"manifest": manifest,
        "validation_selection": validations, "tests": tests}, indent=2) + "\n")
    summary = {"seed": seed, "training_seconds": elapsed, "loss": [losses[0], losses[-1]],
               "counts": manifest["counts"], "selected": selected,
               "tests": [{k: v for k, v in t.items() if k != "rows"} for t in tests]}
    print(json.dumps(summary, indent=2))
    return summary


def inspect(folder: Path, data_path: Path, user: int, budget: int):
    data = load(data_path)
    manifest = json.loads((folder / "model.json").read_text())
    if manifest["input_sha256"] != data.input_sha256:
        raise ValueError("serving data/model mismatch")
    engine = Engine.load(folder)
    query = next((q for q in data.queries("test") if q[0] == user), None)
    if query is None:
        raise ValueError("user has no evaluable warm-start future-positive query")
    _, seen, relevant = query
    exact = set(engine.retrieve(user, seen, budget, False).tolist())
    candidates = set(engine.retrieve(user, seen, budget, True).tolist())
    ranked, scores = engine.recommend(user, seen, budget)
    final = set(ranked.tolist())
    return {"user": user, "budget": budget, "recommendations": [
        {"item": int(i), "score": float(s)} for i, s in zip(ranked, scores)],
        "relevant_loss_attribution": [{"item": int(i), "stage":
            "kept" if i in final else "ranking" if i in candidates else
            "ANN" if i in exact else "representation_or_budget"} for i in sorted(relevant)],
        "boundary": "Held-out offline diagnostic; relevance labels are never passed to inference"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    d = commands.add_parser("download"); d.add_argument("destination", type=Path)
    t = commands.add_parser("train")
    t.add_argument("--data", type=Path, required=True); t.add_argument("--output", type=Path, required=True)
    t.add_argument("--seed", type=int, default=7); t.add_argument("--epochs", type=int, default=30)
    i = commands.add_parser("inspect")
    i.add_argument("--data", type=Path, required=True); i.add_argument("--model", type=Path, required=True)
    i.add_argument("--user", type=int, required=True); i.add_argument("--budget", type=int, default=100)
    args = parser.parse_args()
    if args.action == "download": print(download(args.destination))
    elif args.action == "train": train(args.data, args.output, args.seed, args.epochs)
    else: print(json.dumps(inspect(args.model, args.data, args.user, args.budget), indent=2))


if __name__ == "__main__": main()
