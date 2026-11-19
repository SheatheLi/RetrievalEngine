"""Read-only *exact* diagnostic service; never advertises HNSW or ANN metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from .exact_diagnostic import ExactDiagnostic
from .movielens_data import load


def create_app(model_folder: Path, ratings_file: Path) -> FastAPI:
    model = ExactDiagnostic.load(model_folder)
    data = load(ratings_file)
    if model.manifest.get("input_sha256") != data.input_sha256:
        raise ValueError("serving data/model mismatch")
    prefix = data.prefix("test")
    records: dict[int, list[tuple[int, int, int]]] = {}
    # Prefix is sorted by timestamp; ratings from the test window are never used.
    for user, item, rating, timestamp in prefix:
        records.setdefault(int(user), []).append((int(item), int(rating), int(timestamp)))
    trained = set(int(x) for x in data.train[data.train[:, 2] >= 4, 0])
    app = FastAPI(title="EXACT-ONLY recommendation diagnostic service")

    @app.get("/health")
    def health():
        return {"status": "ready", "backend": "numpy_exact_no_faiss", "approximate": False,
                "model_sha256": model.manifest["sha256"], "input_sha256": data.input_sha256}

    @app.get("/recommend/{user}")
    def recommend(user: int, budget: int = Query(100, ge=1, le=200)):
        if not 0 < user < len(model.state["user_vectors"]):
            raise HTTPException(422, "unknown user")
        record = records.get(user, [])
        seen = {item for item, _, _ in record}
        try:
            if user in trained:
                result = model.recommend(user, seen, budget)
            else:
                history = [item for item, rating, _ in record if rating >= 4]
                result = model.recommend(user, seen, budget, history=history)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        result["representation"] = "trained_user_id" if user in trained else "history_encoder"
        result["evidence_boundary"] = "Pre-test prefix only; not a production online serving guarantee"
        return result

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--port", type=int, default=8809)
    a = p.parse_args()
    import uvicorn
    uvicorn.run(create_app(a.model, a.data), host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
