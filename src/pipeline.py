"""Single entrypoint: download -> preprocess -> select k -> fit -> evaluate ->
business translation -> serialize. Idempotent.

    python -m src.pipeline
"""
from __future__ import annotations

import logging

from src import config
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.train import ClusterTrainer


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def run() -> None:
    configure_logging()
    log = logging.getLogger("pipeline")
    log.info("Stage 1/5 - acquisition")
    loader = DataLoader()
    raw_path = loader.download()
    df = loader.load()
    log.info("Stage 2/5 - preprocessing")
    X, _ = Preprocessor().run(df)
    log.info("Stage 3/5 - k selection")
    trainer = ClusterTrainer(X, data_source=raw_path)
    trainer.select_k()
    log.info("Stage 4/5 - fit + profiling")
    trainer.fit()
    evaluation = trainer.evaluate()
    business = trainer.to_business_metrics()
    log.info("Stage 5/5 - serialization")
    trainer.save(evaluation, business)
    log.info("Done. Artifact: %s | Card: %s", config.PIPELINE_PATH, config.MODEL_CARD_PATH)


if __name__ == "__main__":
    run()
