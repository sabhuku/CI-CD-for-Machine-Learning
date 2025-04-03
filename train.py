import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Any
import json
from datetime import datetime
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "data_path": "Data/drug.csv",
    "model_path": "Model/drug_pipeline.skops",
    "results_dir": "Results",
    "test_size": 0.3,
    "random_state": 125,
    "cv_folds": 5,
    "model_params": {
        "n_estimators": 100,
        "random_state": 125
    },
    "hyperparameter_grid": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    },
    "cat_col": [1, 2, 3],
    "num_col": [0, 4]
}

def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    try:
        Path(CONFIG["results_dir"]).mkdir(parents=True, exist_ok=True)
        Path(CONFIG["model_path"]).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Directories setup completed successfully")
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        raise

def load_data() -> pd.DataFrame:
    """Load and validate the dataset."""
    try:
        if not Path(CONFIG["data_path"]).exists():
            raise FileNotFoundError(f"Data file not found at {CONFIG['data_path']}")
        
        with tqdm(total=1, desc="Loading data") as pbar:
            drug_df = pd.read_csv(CONFIG["data_path"])
            drug_df = drug_df.sample(frac=1, random_state=CONFIG["random_state"])
            pbar.update(1)
        
        # Basic data validation
        if drug_df.empty:
            raise ValueError("Dataset is empty")
        if "Drug" not in drug_df.columns:
            raise ValueError("Target column 'Drug' not found in dataset")
            
        logger.info(f"Successfully loaded dataset with shape: {drug_df.shape}")
        return drug_df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_data(drug_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for training."""
    try:
        with tqdm(total=2, desc="Preparing data") as pbar:
            X = drug_df.drop("Drug", axis=1).values
            y = drug_df.Drug.values
            pbar.update(1)
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=CONFIG["test_size"], 
                random_state=CONFIG["random_state"]
            )
            pbar.update(1)
        
        logger.info(f"Data split completed. Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def create_pipeline() -> Any:
    """Create and return the model pipeline."""
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder, StandardScaler

        transform = ColumnTransformer(
            [
                ("encoder", OrdinalEncoder(), CONFIG["cat_col"]),
                ("num_imputer", SimpleImputer(strategy="median"), CONFIG["num_col"]),
                ("num_scaler", StandardScaler(), CONFIG["num_col"]),
            ]
        )
        
        pipe = Pipeline(
            steps=[
                ("preprocessing", transform),
                ("model", RandomForestClassifier(**CONFIG["model_params"])),
            ]
        )
        
        logger.info("Pipeline created successfully")
        return pipe
    except Exception as e:
        logger.error(f"Error creating pipeline: {str(e)}")
        raise

def perform_cross_validation(pipe: Any, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
    """Perform cross-validation on the training data."""
    try:
        from sklearn.model_selection import cross_val_score
        
        with tqdm(total=CONFIG["cv_folds"], desc="Cross-validation") as pbar:
            cv_scores = cross_val_score(
                pipe, X_train, y_train,
                cv=CONFIG["cv_folds"],
                scoring='accuracy'
            )
            pbar.update(CONFIG["cv_folds"])
        
        # Convert numpy types to Python native types
        cv_results = {
            "mean_cv_score": float(np.mean(cv_scores)),
            "std_cv_score": float(np.std(cv_scores)),
            "cv_scores": [float(score) for score in cv_scores]  # Convert each score to float
        }
        
        logger.info(f"Cross-validation completed. Mean score: {cv_results['mean_cv_score']:.3f} (+/- {cv_results['std_cv_score']:.3f})")
        return cv_results
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        raise

def tune_hyperparameters(pipe: Any, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
    """Perform hyperparameter tuning using GridSearchCV."""
    try:
        from sklearn.model_selection import GridSearchCV
        
        # Calculate total number of combinations for progress bar
        total_combinations = len(CONFIG["hyperparameter_grid"]["model__n_estimators"]) * \
                           len(CONFIG["hyperparameter_grid"]["model__max_depth"]) * \
                           len(CONFIG["hyperparameter_grid"]["model__min_samples_split"]) * \
                           len(CONFIG["hyperparameter_grid"]["model__min_samples_leaf"])
        
        with tqdm(total=total_combinations * CONFIG["cv_folds"], desc="Hyperparameter tuning") as pbar:
            grid_search = GridSearchCV(
                pipe,
                param_grid=CONFIG["hyperparameter_grid"],
                cv=CONFIG["cv_folds"],
                n_jobs=-1,
                verbose=0  # Set to 0 to avoid duplicate progress bars
            )
            
            grid_search.fit(X_train, y_train)
            pbar.update(total_combinations * CONFIG["cv_folds"])
        
        # Log best parameters and score
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Convert numpy types to Python native types in cv_results
        cv_results = {}
        for key, value in grid_search.cv_results_.items():
            if isinstance(value, np.ndarray):
                cv_results[key] = value.tolist()
            elif isinstance(value, (np.float64, np.float32, np.int64, np.int32)):
                cv_results[key] = float(value)
            else:
                cv_results[key] = value
        
        # Save tuning results
        tuning_results = {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "cv_results": cv_results
        }
        
        return grid_search.best_estimator_, tuning_results
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {str(e)}")
        raise

def train_model(pipe: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Train the model and save it."""
    try:
        # First perform hyperparameter tuning to get the best model
        best_pipe, tuning_results = tune_hyperparameters(pipe, X_train, y_train)
        
        # Now perform cross-validation on the best model
        cv_results = perform_cross_validation(best_pipe, X_train, y_train)
        
        # Save model
        import skops.io as sio
        sio.dump(best_pipe, CONFIG["model_path"])
        logger.info(f"Model saved to {CONFIG['model_path']}")
        
        # Save training results
        training_results = {
            "cross_validation": cv_results,
            "hyperparameter_tuning": tuning_results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{CONFIG['results_dir']}/training_results.json", "w") as f:
            json.dump(training_results, f, indent=4)
            
        return best_pipe
            
    except Exception as e:
        logger.error(f"Error training/saving model: {str(e)}")
        raise

def evaluate_model(pipe: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate the model and save results."""
    try:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        with tqdm(total=4, desc="Model evaluation") as pbar:
            predictions = pipe.predict(X_test)
            pbar.update(1)
            
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average="macro")
            pbar.update(1)
            
            # Generate and save classification report
            class_report = classification_report(y_test, predictions, output_dict=True)
            with open(f"{CONFIG['results_dir']}/classification_report.txt", "w") as f:
                f.write(classification_report(y_test, predictions))
            pbar.update(1)
            
            # Plot and save confusion matrix
            cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
            disp.plot()
            plt.savefig(f"{CONFIG['results_dir']}/model_results.png", dpi=120)
            plt.close()
            pbar.update(1)
        
        # Save metrics
        metrics = {
            "accuracy": round(accuracy, 2),
            "f1_score": round(f1, 2),
            "classification_report": class_report,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{CONFIG['results_dir']}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Model evaluation completed. Accuracy: {accuracy:.2f}, F1: {f1:.2f}")
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def main():
    """Main execution function."""
    try:
        setup_directories()
        drug_df = load_data()
        X_train, X_test, y_train, y_test = prepare_data(drug_df)
        pipe = create_pipeline()
        best_pipe = train_model(pipe, X_train, y_train)
        metrics = evaluate_model(best_pipe, X_test, y_test)
        
        logger.info("Training pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()