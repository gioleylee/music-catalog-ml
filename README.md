# Music catalog ML

Two notebooks from an introductory machine learning course exploring audio features and evaluating models on a music catalog.

- **[Catalog exploration](exploration/experiment.ipynb)** audits duplicate and inconsistent tracks, compares standardized and unscaled PCA, examines clustering stability, and uses track-grouped splits for regression. The original run cut energy prediction RMSE from 0.256 to 0.123 versus a mean baseline. Popularity prediction improved by only 1.7%.
- **[Genre classification](genre-classification/experiment.ipynb)** compares k-NN, Gaussian Naive Bayes, trees, ensembles, and SVMs with nested cross-validation. The selected random forest scored 54.8% accuracy and 0.431 macro-F1 on the original holdout. A hip-hop detector reached 63.8% holdout recall against an 80% target, despite meeting the target on out-of-fold training predictions.

The CSVs and [confusion matrix](genre-classification/confusion-matrix.png) record the original runs. The public notebooks use a new fixed seed to remove the course submission identifier, so reruns may produce different splits and scores. OpenAI Codex assisted with implementation, debugging, and analysis.

## Run locally

Install the dependencies with `python -m pip install -r requirements.txt`. Start Jupyter in the repository and run each notebook from its own directory.

- The exploration notebook downloads `tracks.csv` from the course's public data release when the file is missing.
- In `genre-classification`, run `python build_processed_tracks.py` before running the notebook. The script downloads the same source catalog if needed and creates `processed_tracks.csv`. You can also pass the path to a local copy of `tracks.csv`.
