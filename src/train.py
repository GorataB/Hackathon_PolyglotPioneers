"""
train.py

Entry point for the forecasting pipeline.

Pipeline
--------
1. Load cleaned dataset
2. Run persistence baseline
3. Validate dataset
4. Preprocess data
5. Build LSTM sequences
6. Initialise trainer
7. Resume training (if checkpoint exists)
8. Train model
9. Load best model
10. Evaluate model
11. Close resources

Author: Polyglot Pioneers
"""

from __future__ import annotations
from src.utils.correlation import correlation_report
from src.utils.feature_importance import feature_importance_report
from src.config.config import TARGET_COLUMN

import sys
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent

for path in (str(ROOT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.config.config import TARGET_COLUMN

from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.sequence_builder import SequenceBuilder
from src.models.trainer import Trainer

# NEW
from src.models.baseline import persistence_baseline

from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    """
    Execute the complete forecasting pipeline.

    Returns
    -------
    int
        Exit status code.
    """

    trainer = None

    try:

        logger.info("=" * 80)
        logger.info("FORECASTING PIPELINE STARTED")
        logger.info("=" * 80)

        # ==================================================
        # Load Dataset
        # ==================================================

        logger.info("Step 1/6 : Loading dataset")

        loader = DataLoader()

        dataframe, feature_columns = loader.run()

        # ==================================================
        # Persistence Baseline
        # ==================================================

        logger.info("Running persistence baseline...")

        persistence_baseline(
            dataframe[TARGET_COLUMN].values
        )

        # ==================================================
        # Correlation Analysis
        # ==================================================

        logger.info("Running correlation analysis...")

        correlation_report(
            dataframe,
            TARGET_COLUMN,
        )

        # ==================================================
        # Feature Importance
        # ==================================================

        logger.info("Running feature importance analysis...")

        feature_importance_report(
            dataframe,
            feature_columns,
            TARGET_COLUMN,
        )

        # ==================================================
        # Preprocess
        # ==================================================

        logger.info("Step 2/6 : Preprocessing dataset")

        preprocessor = Preprocessor()

        processed_data = preprocessor.run(
            dataframe=dataframe,
            feature_columns=feature_columns,
        )

        # ==================================================
        # Sequence Generation
        # ==================================================

        logger.info("Step 3/6 : Building LSTM sequences")

        builder = SequenceBuilder()

        sequence_data = builder.build(
            processed_data,
        )

        # ==================================================
        # Training
        # ==================================================

        logger.info("Step 4/6 : Initialising trainer")

        trainer = Trainer(sequence_data)

        trainer.resume_training()

        logger.info("Step 5/6 : Training model")

        trainer.train()

        # ==================================================
        # Evaluation
        # ==================================================

        logger.info("Loading best model...")

        trainer.load_best_model()

        logger.info("Step 6/6 : Evaluating model")

        metrics = trainer.evaluate()

        logger.info("=" * 80)
        logger.info("FINAL TEST RESULTS")
        logger.info("=" * 80)

        for metric, value in metrics.items():

            logger.info(
                f"{metric.upper():<8}: {value:.6f}"
            )

        logger.info("=" * 80)
        logger.info("FORECASTING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return 0

    except KeyboardInterrupt:

        logger.warning(
            "Training interrupted by user."
        )

        return 130

    except Exception:

        logger.exception(
            "Pipeline execution failed."
        )

        traceback.print_exc()

        return 1

    finally:

        if trainer is not None:

            trainer.close()


if __name__ == "__main__":

    sys.exit(main())