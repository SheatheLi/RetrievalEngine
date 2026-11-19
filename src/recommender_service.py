"""Read-only research inference over a pinned trained model. Bind to loopback."""
from __future__ import annotations

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from .movielens_data import load
from .recommender import Engine


def create_app(model_folder: Path, ratings_file: Path) -> FastAPI:
    manifest = json.loads((model_folder / "model.json").read_text())
    data = load(ratings_file)
    if manifest["input_sha256"] != data.input_sha256:
        raise ValueError("serving data/model mismatch")
    engine = Engine.load(model_folder)
    prefix = data.prefix("test")
    histories = {int(u): set(prefix[prefix[:, 0] == u, 1].tolist()) for u in set(prefix[:, 0].tolist())}
    positive_histories = {int(u): prefix[(prefix[:, 0] == u) & (prefix[:, 2] >= 4), 1].tolist()
                          for u in set(prefix[:, 0].tolist())}
    trained = set(data.train[data.train[:, 2] >= 4, 0].tolist())
    app = FastAPI(title="Two-stage recommendation research service")

    @app.get("/health")
    def health():
        return {"status": "ready", "model_sha256": manifest["sha256"], "input_sha256": data.input_sha256}

    @app.get("/recommend/{user}")
    def recommend(user: int, budget: int = Query(100, ge=10, le=200)):
        if not 0 < user < len(engine.user_vectors):
            raise HTTPException(422, "unknown user ID")
        if user in trained:
            method = "hnsw_ranker"
        elif engine.history_item_vectors is not None and (set(positive_histories.get(user, [])) & set(engine.catalog)):
            method = "history_hnsw"
        else:
            raise HTTPException(422, "no trained ID and no usable known positive history")
        ids, scores = engine.recommend(user, histories.get(user, set()), budget,
                                       method=method, history_positive=positive_histories.get(user, []))
        return {"user": user, "budget": budget, "model_seed": manifest["seed"],
                "representation": "history_encoder" if method == "history_hnsw" else "trained_user_id",
                "items": [{"item": int(i), "score": float(s)} for i, s in zip(ids, scores)]}

    return app


def main():
    import argparse
    import uvicorn
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8808)
    args=parser.parse_args()
    uvicorn.run(create_app(args.model,args.data),host='127.0.0.1',port=args.port)


if __name__=='__main__':main()
