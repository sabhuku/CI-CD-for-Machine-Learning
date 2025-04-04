import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Any
import json
from datetime import datetime
import matplotlib.pyplot as plt
import skops.io as sio
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - moved to session state to allow UI modification
if 'config' not in st.session_state:
    st.session_state.config = {
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
        Path(st.session_state.config["results_dir"]).mkdir(parents=True, exist_ok=True)
        Path(st.session_state.config["model_path"]).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Directories setup completed successfully")
        st.success("Directories setup completed successfully")
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        st.error(f"Error creating directories: {str(e)}")
        raise

def load_data() -> pd.DataFrame:
    """Load and validate the dataset."""
    try:
        if not Path(st.session_state.config["data_path"]).exists():
            raise FileNotFoundError(f"Data file not found at {st.session_state.config['data_path']}")
        
        with st.spinner("Loading data..."):
            drug_df = pd.read_csv(st.session_state.config["data_path"])
            drug_df = drug_df.sample(frac=1, random_state=st.session_state.config["random_state"])
        
        # Basic data validation
        if drug_df.empty:
            raise ValueError("Dataset is empty")
        if "Drug" not in drug_df.columns:
            raise ValueError("Target column 'Drug' not found in dataset")
            
        logger.info(f"Successfully loaded dataset with shape: {drug_df.shape}")
        st.success(f"Successfully loaded dataset with shape: {drug_df.shape}")
        return drug_df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        raise

def prepare_data(drug_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data for training."""
    try:
        with st.spinner("Preparing data..."):
            X = drug_df.drop("Drug", axis=1).values
            y = drug_df.Drug.values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=st.session_state.config["test_size"], 
                random_state=st.session_state.config["random_state"]
            )
        
        logger.info(f"Data split completed. Train size: {len(X_train)}, Test size: {len(X_test)}")
        st.success(f"Data split completed. Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        st.error(f"Error preparing data: {str(e)}")
        raise

def create_pipeline() -> Any:
    """Create and return the model pipeline."""
    try:
        transform = ColumnTransformer(
            [
                ("encoder", OrdinalEncoder(), st.session_state.config["cat_col"]),
                ("num_imputer", SimpleImputer(strategy="median"), st.session_state.config["num_col"]),
                ("num_scaler", StandardScaler(), st.session_state.config["num_col"]),
            ]
        )
        
        pipe = Pipeline(
            steps=[
                ("preprocessing", transform),
                ("model", RandomForestClassifier(**st.session_state.config["model_params"])),
            ]
        )
        
        logger.info("Pipeline created successfully")
        st.success("Pipeline created successfully")
        return pipe
    except Exception as e:
        logger.error(f"Error creating pipeline: {str(e)}")
        st.error(f"Error creating pipeline: {str(e)}")
        raise

def perform_cross_validation(pipe: Any, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
    """Perform cross-validation on the training data."""
    try:
        with st.spinner("Running cross-validation..."):
            cv_scores = cross_val_score(
                pipe, X_train, y_train,
                cv=st.session_state.config["cv_folds"],
                scoring='accuracy'
            )
        
        # Convert numpy types to Python native types
        cv_results = {
            "mean_cv_score": float(np.mean(cv_scores)),
            "std_cv_score": float(np.std(cv_scores)),
            "cv_scores": [float(score) for score in cv_scores]
        }
        
        logger.info(f"Cross-validation completed. Mean score: {cv_results['mean_cv_score']:.3f} (+/- {cv_results['std_cv_score']:.3f})")
        st.success(f"Cross-validation completed. Mean score: {cv_results['mean_cv_score']:.3f} (± {cv_results['std_cv_score']:.3f})")
        return cv_results
    except Exception as e:
        logger.error(f"Error in cross-validation: {str(e)}")
        st.error(f"Error in cross-validation: {str(e)}")
        raise

def tune_hyperparameters(pipe: Any, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
    """Perform hyperparameter tuning using GridSearchCV."""
    try:
        with st.spinner("Running hyperparameter tuning (this may take a while)..."):
            grid_search = GridSearchCV(
                pipe,
                param_grid=st.session_state.config["hyperparameter_grid"],
                cv=st.session_state.config["cv_folds"],
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
        
        # Log best parameters and score
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        st.success(f"Best parameters found: {grid_search.best_params_}")
        st.success(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
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
        st.error(f"Error in hyperparameter tuning: {str(e)}")
        raise

def train_model(pipe: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Train the model and save it."""
    try:
        # First perform hyperparameter tuning to get the best model
        best_pipe, tuning_results = tune_hyperparameters(pipe, X_train, y_train)
        
        # Now perform cross-validation on the best model
        cv_results = perform_cross_validation(best_pipe, X_train, y_train)
        
        # Save model
        sio.dump(best_pipe, st.session_state.config["model_path"])
        logger.info(f"Model saved to {st.session_state.config['model_path']}")
        st.success(f"Model saved to {st.session_state.config['model_path']}")
        
        # Save training results
        training_results = {
            "cross_validation": cv_results,
            "hyperparameter_tuning": tuning_results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{st.session_state.config['results_dir']}/training_results.json", "w") as f:
            json.dump(training_results, f, indent=4)
            
        return best_pipe
            
    except Exception as e:
        logger.error(f"Error training/saving model: {str(e)}")
        st.error(f"Error training/saving model: {str(e)}")
        raise

def evaluate_model(pipe: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate the model and save results."""
    try:
        with st.spinner("Evaluating model..."):
            predictions = pipe.predict(X_test)
            
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average="macro")
            
            # Generate and save classification report
            class_report = classification_report(y_test, predictions, output_dict=True)
            with open(f"{st.session_state.config['results_dir']}/classification_report.txt", "w") as f:
                f.write(classification_report(y_test, predictions))
            
            # Plot and save confusion matrix
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
            disp.plot(ax=ax)
            st.pyplot(fig)
            plt.savefig(f"{st.session_state.config['results_dir']}/model_results.png", dpi=120)
            plt.close()
        
        # Display metrics
        st.subheader("Model Evaluation Results")
        st.metric("Accuracy", f"{accuracy:.2f}")
        st.metric("F1 Score", f"{f1:.2f}")
        
        # Display classification report as a table
        st.subheader("Classification Report")
        class_report_df = pd.DataFrame(class_report).transpose()
        st.dataframe(class_report_df)
        
        # Save metrics
        metrics = {
            "accuracy": round(accuracy, 2),
            "f1_score": round(f1, 2),
            "classification_report": class_report,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{st.session_state.config['results_dir']}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Model evaluation completed. Accuracy: {accuracy:.2f}, F1: {f1:.2f}")
        st.success(f"Model evaluation completed. Accuracy: {accuracy:.2f}, F1: {f1:.2f}")
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        st.error(f"Error evaluating model: {str(e)}")
        raise

def run_pipeline():
    """Run the complete pipeline."""
    try:
        setup_directories()
        drug_df = load_data()
        X_train, X_test, y_train, y_test = prepare_data(drug_df)
        pipe = create_pipeline()
        best_pipe = train_model(pipe, X_train, y_train)
        metrics = evaluate_model(best_pipe, X_test, y_test)
        
        logger.info("Training pipeline completed successfully")
        st.balloons()
        st.success("Training pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        st.error(f"Pipeline failed: {str(e)}")
        raise

def main():
    """Main Streamlit application."""
    st.title("Drug Classification Model Trainer")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        st.session_state.config["test_size"] = st.slider(
            "Test Size Ratio", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.3, 
            step=0.05
        )
        st.session_state.config["cv_folds"] = st.slider(
            "Cross-Validation Folds", 
            min_value=3, 
            max_value=10, 
            value=5
        )
        st.session_state.config["random_state"] = st.number_input(
            "Random State", 
            value=125
        )
        
        # Model parameters
        st.subheader("Model Parameters")
        st.session_state.config["model_params"]["n_estimators"] = st.number_input(
            "Number of Estimators", 
            min_value=10, 
            max_value=500, 
            value=100
        )
        st.session_state.config["model_params"]["random_state"] = st.session_state.config["random_state"]
        
        # File paths
        st.subheader("File Paths")
        st.session_state.config["data_path"] = st.text_input(
            "Data Path", 
            value="Data/drug.csv"
        )
        st.session_state.config["model_path"] = st.text_input(
            "Model Save Path", 
            value="Model/drug_pipeline.skops"
        )
        st.session_state.config["results_dir"] = st.text_input(
            "Results Directory", 
            value="Results"
        )
    
    # Main content area
    if st.button("Run Full Pipeline"):
        run_pipeline()
    
    # Option to run individual steps
    st.header("Run Individual Steps")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Load Data"):
            st.session_state.drug_df = load_data()
            if 'drug_df' in st.session_state:
                st.dataframe(st.session_state.drug_df.head())
    
    with col2:
        if st.button("Prepare Data"):
            if 'drug_df' in st.session_state:
                X_train, X_test, y_train, y_test = prepare_data(st.session_state.drug_df)
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
            else:
                st.warning("Please load data first")
    
    with col3:
        if st.button("Create Pipeline"):
            st.session_state.pipe = create_pipeline()
    
    if st.button("Train Model"):
        if all(key in st.session_state for key in ['pipe', 'X_train', 'y_train']):
            st.session_state.best_pipe = train_model(
                st.session_state.pipe, 
                st.session_state.X_train, 
                st.session_state.y_train
            )
        else:
            st.warning("Please complete previous steps first")
    
    if st.button("Evaluate Model"):
        if all(key in st.session_state for key in ['best_pipe', 'X_test', 'y_test']):
            evaluate_model(
                st.session_state.best_pipe, 
                st.session_state.X_test, 
                st.session_state.y_test
            )
        else:
            st.warning("Please train the model first")

if __name__ == "__main__":
    main()