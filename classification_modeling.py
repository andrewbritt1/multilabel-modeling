#!/usr/bin/env python3
"""
Unified transformer-based model training and inference script.

This module provides functionality for:
Training:
- Multiclass classification
- Multilabel classification  
- Masked Language Model (MLM) pretraining

Inference:
- Multiclass classification inference
- Multilabel classification inference

Usage:
    # Training
    python unified_model.py --mode multiclass -c config.yaml
    python unified_model.py --mode multilabel -c config.yaml
    python unified_model.py --mode mlm -c config.yaml
    
    # Inference
    python unified_model.py --mode multiclass-inference -c inference_config.yaml
    python unified_model.py --mode multilabel-inference -c inference_config.yaml
"""

import argparse
import logging
import math
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import evaluate
import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# ============================================================================
# Training Components
# ============================================================================

class EvalLossCallback(TrainerCallback):
    """Custom callback to log evaluation loss."""
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation loss as a metric."""
        if metrics is not None and "eval_loss" not in metrics and "loss" in metrics:
            metrics["eval_loss"] = metrics["loss"]


class WeightedTrainer(Trainer):
    """Custom Trainer that supports both multiclass and multilabel classification with weighted loss."""

    def __init__(self, *args, class_weights=None, problem_type="multiclass", **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.problem_type = problem_type

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with class weights for both multiclass and multilabel."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.problem_type == "multilabel":
            # Use BCEWithLogitsLoss for multilabel
            loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits, labels.float())
        else:
            # Use CrossEntropyLoss for multiclass
            loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels),
                labels.view(-1)
            )
        
        return (loss, outputs) if return_outputs else loss


class MetricsCalculator:
    """Handles metric computation for both multiclass and multilabel classification."""

    def __init__(self, problem_type="multiclass"):
        self.problem_type = problem_type
        self.accuracy_metric = evaluate.load("accuracy")
        self.precision_metric = evaluate.load("precision")
        self.recall_metric = evaluate.load("recall")
        self.f1_metric = evaluate.load("f1")

    def compute_metrics(self, p):
        """Calculate metrics based on problem type."""
        if self.problem_type == "multilabel":
            # Multilabel metrics
            logits = p.predictions
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            preds = (probs >= 0.5).astype(int)
            
            # Flatten for multilabel metrics
            flat_preds = preds.flatten().tolist()
            flat_labels = p.label_ids.flatten().tolist()
        else:
            # Multiclass metrics
            preds = p.predictions.argmax(-1)
            flat_preds = preds.tolist()
            flat_labels = p.label_ids.tolist()

        # Calculate metrics
        accuracy = self.accuracy_metric.compute(predictions=flat_preds, references=flat_labels)
        precision = self.precision_metric.compute(
            predictions=flat_preds, references=flat_labels, average="weighted"
        )
        recall = self.recall_metric.compute(
            predictions=flat_preds, references=flat_labels, average="weighted"
        )
        f1 = self.f1_metric.compute(
            predictions=flat_preds, references=flat_labels, average="weighted"
        )

        return {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1": f1["f1"],
        }


# ============================================================================
# Inference Components
# ============================================================================

@dataclass
class InferenceConfig:
    """Configuration dataclass for the inference process."""
    input_file: str
    output_file: str
    model_path: str
    text_column: str
    threshold: float = 0.5  # For multilabel
    batch_size: int = 32
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    id2label: Dict[int, str] = None


class TextDataset(TorchDataset):
    """Dataset for text inference."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]
    
    def collate_fn(self, batch):
        """Custom collate function for text batching."""
        encodings = self.tokenizer(
            batch, 
            truncation=True, 
            padding=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        return encodings


# ============================================================================
# Shared Components
# ============================================================================

class MLflowManager:
    """Manages MLflow configuration, including auto-starting the tracking server if needed."""
    
    DEFAULT_PORT = 5000
    
    def __init__(self, config):
        self.config = config
        self.tracking_uri = config["mlflow"].get("tracking_uri", None)
        self.autostart = config["mlflow"].get("autostart", True)
        self.port = config["mlflow"].get("port", self.DEFAULT_PORT)
        self.backend_store_uri = config["mlflow"].get("backend_store_uri", "./mlruns")
        self.artifact_store = config["mlflow"].get("artifact_store", None)
        self.server_process = None
    
    def start_tracking_server_if_needed(self):
        """Start MLflow tracking server if it's not already running and autostart is enabled."""
        if not self.config["mlflow"]["enabled"]:
            print("MLflow tracking is disabled.")
            return
            
        if self.tracking_uri:
            print(f"Using provided MLflow tracking URI: {self.tracking_uri}")
            mlflow.set_tracking_uri(self.tracking_uri)
            return
            
        local_uri = f"http://localhost:{self.port}"
        
        if not self.autostart:
            print(f"MLflow autostart disabled. Using local URI: {local_uri}")
            mlflow.set_tracking_uri(local_uri)
            return
            
        if is_port_in_use(self.port):
            print(f"MLflow server already running on port {self.port}. Using URI: {local_uri}")
            mlflow.set_tracking_uri(local_uri)
            return
            
        print(f"Starting MLflow tracking server on port {self.port}...")
        cmd = ["mlflow", "server", "--host", "0.0.0.0", "--port", str(self.port), 
               "--backend-store-uri", self.backend_store_uri]
        
        if self.artifact_store:
            cmd.extend(["--default-artifact-root", self.artifact_store])
        
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            for _ in range(10):
                time.sleep(0.5)
                if is_port_in_use(self.port):
                    break
            
            if is_port_in_use(self.port):
                print(f"MLflow tracking server started successfully on {local_uri}")
                mlflow.set_tracking_uri(local_uri)
            else:
                print("Warning: MLflow server did not start properly. Check logs for details.")
                
        except Exception as e:
            print(f"Failed to start MLflow tracking server: {e}")
            mlflow.set_tracking_uri(local_uri)
    
    def cleanup(self):
        """Terminate the MLflow server if it was started by this process."""
        if self.server_process:
            print("Shutting down MLflow tracking server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print("MLflow tracking server stopped.")
            except subprocess.TimeoutExpired:
                print("MLflow tracking server did not terminate gracefully, forcing shutdown.")
                self.server_process.kill()


def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


# ============================================================================
# Main Unified Class
# ============================================================================

class UnifiedTransformerPipeline:
    """Unified class for training and inference with transformer models."""

    def __init__(self, config: Dict, mode: str = "multiclass"):
        """Initialize the unified transformer pipeline.
        
        Args:
            config: Dictionary containing configuration parameters
            mode: One of "multiclass", "multilabel", "mlm", "multiclass-inference", "multilabel-inference"
        """
        self.config = config
        self.mode = mode
        self.is_inference = mode.endswith("-inference")
        self.base_mode = mode.replace("-inference", "") if self.is_inference else mode
        self.logger = self._setup_logging()
        
        # Set seeds for reproducibility (training only)
        if not self.is_inference:
            seed = config.get("seed", 42)
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Disable parallel processing to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Initialize MLflow manager (training only)
        if not self.is_inference:
            if "mlflow" not in self.config:
                self.config["mlflow"] = {}
            if "enabled" not in self.config["mlflow"]:
                self.config["mlflow"]["enabled"] = True
            if "autostart" not in self.config["mlflow"]:
                self.config["mlflow"]["autostart"] = True
                
            self.mlflow_manager = MLflowManager(self.config)
        
        # Initialize based on mode
        if self.is_inference:
            self._init_inference()
        else:
            self._init_training()

    def _init_training(self):
        """Initialize components for training."""
        # Model attributes
        self.model_name = self.config["model"]["name"]
        self.text_column = self.config["data"]["text_column"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        
        # Data attributes
        self.train_dataset = None
        self.dev_dataset = None
        self.max_length = None
        
        # Mode-specific initialization
        if self.mode in ["multiclass", "multilabel"]:
            self.metrics_calculator = MetricsCalculator(
                problem_type="multi_label_classification" if self.mode == "multilabel" else "multiclass"
            )
            self.class_weights = None
            self.id2label = None
            self.label2id = None
            
            if self.mode == "multiclass":
                self.label_column = self.config["data"]["label_column"]
                self.label_encoder = None
            else:  # multilabel
                self.label_columns = None
        else:  # mlm
            self.data_collator = None
            self.mlm_probability = self.config.get("mlm", {}).get("mlm_probability", 0.15)
            self.do_whole_word_mask = self.config.get("mlm", {}).get("do_whole_word_mask", True)

    def _init_inference(self):
        """Initialize components for inference."""
        # Create InferenceConfig from the loaded config
        inference_config = InferenceConfig(
            input_file=self.config['inference']['input_file'],
            output_file=self.config['inference']['output_file'],
            model_path=self.config['inference']['model_path'],
            text_column=self.config['inference']['text_column'],
            threshold=self.config['inference'].get('threshold', 0.5),
            batch_size=self.config['inference'].get('batch_size', 32),
            max_length=self.config['inference'].get('max_length', 512),
            device=self.config['inference'].get('device', "cuda" if torch.cuda.is_available() else "cpu"),
        )
        
        # Load id2label mapping if provided
        if 'id2label' in self.config['inference']:
            inference_config.id2label = {int(k): v for k, v in self.config['inference']['id2label'].items()}
        
        self.inference_config = inference_config
        
        # Load model and tokenizer
        self.logger.info(f"Loading model and tokenizer from {inference_config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(inference_config.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(inference_config.model_path)
        self.model.eval()
        self.model.to(inference_config.device)
        
        # Extract id2label mapping from model if not provided
        if self.inference_config.id2label is None:
            self.inference_config.id2label = self.model.config.id2label
            self.logger.info(f"Using model's id2label mapping: {self.inference_config.id2label}")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        log_level = getattr(logging, self.config.get("logging", {}).get("level", "INFO").upper())
        logger.setLevel(log_level)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        log_file = self.config.get("logging", {}).get("file")
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger

    # ========================================================================
    # Training Methods
    # ========================================================================

    def setup_mlflow(self):
        """Configure MLflow tracking."""
        self.mlflow_manager.start_tracking_server_if_needed()
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        mlflow.transformers.autolog()

    def prepare_data(self) -> int:
        """Load and prepare the dataset based on the mode.
        
        Returns:
            Number of labels in the dataset (for classification) or 0 for MLM
        """
        if self.mode == "multiclass":
            return self._prepare_multiclass_data()
        elif self.mode == "multilabel":
            return self._prepare_multilabel_data()
        else:  # mlm
            return self._prepare_mlm_data()

    def _prepare_multiclass_data(self) -> int:
        """Prepare data for multiclass classification."""
        self.logger.info(f"Loading multiclass data from {self.config['data']['path']}")
        
        # Load dataset
        df = pd.read_csv(
            self.config["data"]["path"], 
            usecols=[self.text_column, self.label_column]
        ).dropna().drop_duplicates()
        
        # Clean and format data
        df = (df
            .rename(columns={self.text_column: 'text', self.label_column: 'label'})
            .dropna(subset=['label', 'text'])
            .drop_duplicates()
            .groupby('label')
            .filter(lambda x: len(x) > 1)
            .loc[:, ['text', 'label']]
        )
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        df['label'] = self.label_encoder.fit_transform(df['label'])
        num_labels = len(self.label_encoder.classes_)
        
        # Create mappings
        self.id2label = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        self.label2id = {label: i for i, label in enumerate(self.label_encoder.classes_)}
        
        # Split data
        test_size = self.config["data"].get("test_size", 0.2)
        val_split = self.config["training"].get("val_split", test_size)
        train_df, dev_df = train_test_split(
            df, 
            test_size=val_split, 
            stratify=df["label"], 
            random_state=42
        )
        
        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_df['label']),
            y=train_df['label']
        )
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        
        # Convert to datasets
        self.train_dataset = Dataset.from_pandas(train_df)
        self.dev_dataset = Dataset.from_pandas(dev_df)
        
        # Remove pandas index if present
        for dataset in [self.train_dataset, self.dev_dataset]:
            if "__index_level_0__" in dataset.column_names:
                dataset = dataset.remove_columns(["__index_level_0__"])
                
        return num_labels

    def _prepare_multilabel_data(self) -> int:
        """Prepare data for multilabel classification."""
        self.logger.info(f"Loading multilabel data from {self.config['data']['path']}")
        
        # Load the CSV
        df = pd.read_csv(self.config["data"]["path"]).dropna().drop_duplicates()
        
        # Rename text column and identify label columns
        df = df.rename(columns={self.text_column: "text"})
        self.label_columns = [col for col in df.columns if col != "text"]
        num_labels = len(self.label_columns)
        
        # Remove rows with no positive labels
        df = df[df[self.label_columns].sum(axis=1) > 0]
        
        # Train-test split
        val_split = self.config["training"].get("val_split", 0.2)
        train_df, dev_df = train_test_split(
            df, test_size=val_split, random_state=42
        )
        
        # Compute positive weights for each label
        pos_weights = []
        for col in self.label_columns:
            positives = train_df[col].sum()
            negatives = len(train_df) - positives
            pos_weight = negatives / positives if positives > 0 else 1.0
            pos_weights.append(pos_weight)
        self.class_weights = torch.tensor(pos_weights, dtype=torch.float)
        
        self.logger.info(f"Positive weights for each label: {self.class_weights}")
        
        # Create label mappings
        self.id2label = {i: label for i, label in enumerate(self.label_columns)}
        self.label2id = {label: i for i, label in enumerate(self.label_columns)}
        
        # Convert to HF datasets
        self.train_dataset = Dataset.from_pandas(train_df).remove_columns(
            train_df.columns.difference(["text"] + self.label_columns)
        )
        self.dev_dataset = Dataset.from_pandas(dev_df).remove_columns(
            dev_df.columns.difference(["text"] + self.label_columns)
        )
        
        return num_labels

    def _prepare_mlm_data(self) -> int:
        """Prepare data for MLM pretraining."""
        self.logger.info(f"Loading MLM data from {self.config['data']['path']}")
        
        # Load CSV into a Dataset object
        df = pd.read_csv(
            self.config["data"]["path"], 
            usecols=[self.text_column]
        ).dropna().drop_duplicates()
        
        dataset = Dataset.from_pandas(df)
        
        # Train-dev split
        train_test_split_ratio = self.config.get("mlm", {}).get("train_test_split", 0.05)
        split_dataset = dataset.train_test_split(
            test_size=train_test_split_ratio, 
            seed=42
        )
        self.train_dataset = split_dataset["train"]
        self.dev_dataset = split_dataset["test"]
        
        return 0  # No labels for MLM

    def calculate_max_length(self) -> None:
        """Calculate a reasonable maximum sequence length for the dataset."""
        # For MLM, check if max_length is "auto"
        if self.mode == "mlm" and self.config["data"].get("max_length") == "auto":
            percentile = 95
        else:
            percentile = self.config["data"].get("length_percentile", 95)
        
        # Get text column based on mode
        if self.mode == "mlm":
            text_data = self.train_dataset[self.text_column]
        else:
            text_data = self.train_dataset["text"]
        
        lengths = [
            len(self.tokenizer.encode(text, truncation=False)) 
            for text in text_data
        ]
        
        percentile_length = int(np.percentile(lengths, percentile))
        
        # Get max_length from config or use calculated value
        if self.mode == "mlm" and self.config["data"].get("max_length") == "auto":
            self.max_length = min(percentile_length, 512)  # Cap at 512 for MLM
        else:
            self.max_length = self.config["data"].get("max_length", percentile_length)
        
        self.logger.info(f"Using max sequence length: {self.max_length} tokens (at {percentile}th percentile)")

    def tokenize_datasets(self):
        """Tokenize the datasets based on the mode."""
        self.logger.info("Tokenizing datasets")
        
        if self.mode == "multiclass":
            tokenize_function = self._tokenize_multiclass
        elif self.mode == "multilabel":
            tokenize_function = self._tokenize_multilabel
        else:  # mlm
            tokenize_function = self._tokenize_mlm
        
        # Apply tokenization
        remove_columns = self.train_dataset.column_names
        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=remove_columns,
            load_from_cache_file=False
        )
        
        self.dev_dataset = self.dev_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=remove_columns,
            load_from_cache_file=False
        )
        
        # Set format to PyTorch tensors
        if self.mode in ["multiclass", "multilabel"]:
            columns = ['input_ids', 'attention_mask', 'labels']
        else:  # mlm
            columns = ['input_ids', 'attention_mask']
            if 'token_type_ids' in self.train_dataset.column_names:
                columns.append('token_type_ids')
                
        self.train_dataset.set_format(type='torch', columns=columns)
        self.dev_dataset.set_format(type='torch', columns=columns)

    def _tokenize_multiclass(self, examples):
        """Tokenize for multiclass classification."""
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=self.max_length
        )
        tokenized_inputs['labels'] = examples["label"]
        return tokenized_inputs

    def _tokenize_multilabel(self, examples):
        """Tokenize for multilabel classification."""
        tokenized_inputs = self.tokenizer(
            examples["text"], 
            truncation=True, 
            padding=True, 
            max_length=self.max_length
        )
        
        # For each example, combine all label column values into a list
        labels = []
        for i in range(len(examples["text"])):
            labels.append([examples[col][i] for col in self.label_columns])
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def _tokenize_mlm(self, examples):
        """Tokenize for MLM pretraining."""
        return self.tokenizer(
            examples[self.text_column],
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True
        )

    def prepare_model(self, num_labels: int = None):
        """Initialize the model with proper configuration."""
        if self.mode == "mlm":
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.logger.info(f"Loaded MLM model: {self.model_name}")
        else:
            problem_type = "multi_label_classification" if self.mode == "multilabel" else "single_label_classification"
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                problem_type=problem_type,
            )
            
            # Set id2label and label2id mappings
            self.model.config.id2label = self.id2label
            self.model.config.label2id = self.label2id
            
            self.logger.info(f"Loaded {self.mode} classification model: {self.model_name}")

    def setup_data_collator(self):
        """Set up the appropriate data collator based on mode."""
        if self.mode == "mlm":
            if self.do_whole_word_mask:
                self.data_collator = DataCollatorForWholeWordMask(
                    tokenizer=self.tokenizer,
                    mlm=True,
                    mlm_probability=self.mlm_probability,
                    pad_to_multiple_of=8
                )
            else:
                self.data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=True,
                    mlm_probability=self.mlm_probability,
                    pad_to_multiple_of=8
                )
        else:
            self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def get_output_dir(self) -> str:
        """Get the output directory path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = self.model_name.replace("/", "_")
        base_dir = self.config.get("output", {}).get("dir", "output")
        
        return str(Path(base_dir) / f"{model_name_safe}-{self.mode}-{timestamp}")

    def train(self):
        """Train the model using the prepared datasets."""
        # Setup MLflow if enabled
        if self.config["mlflow"]["enabled"]:
            self.setup_mlflow()
        
        # Prepare data
        num_labels = self.prepare_data()
        
        # Calculate max length
        self.calculate_max_length()
        
        # Tokenize datasets
        self.tokenize_datasets()
        
        # Prepare model
        if self.mode in ["multiclass", "multilabel"]:
            self.prepare_model(num_labels)
        else:  # mlm
            self.prepare_model()
        
        # Setup data collator
        self.setup_data_collator()
        
        # Calculate steps
        per_device_batch_size = self.config["training"]["batch_size"]
        save_steps = round(len(self.train_dataset) / per_device_batch_size)
        num_train_epochs = self.config["training"]["epochs"]
        total_steps = int(num_train_epochs * save_steps)
        
        # Output directory
        output_dir = self.get_output_dir()
        
        try:
            with mlflow.start_run() as run:
                # Log parameters
                if self.config["mlflow"]["enabled"]:
                    mlflow.log_param("mode", self.mode)
                    mlflow.log_param("model_name", self.model_name)
                    mlflow.log_param("max_length", self.max_length)
                    mlflow.log_param("batch_size", per_device_batch_size)
                    mlflow.log_param("epochs", num_train_epochs)
                    
                    if self.mode in ["multiclass", "multilabel"]:
                        mlflow.log_param("id2label", self.id2label)
                    elif self.mode == "mlm":
                        mlflow.log_param("mlm_probability", self.mlm_probability)
                        mlflow.log_param("do_whole_word_mask", self.do_whole_word_mask)
                
                # Configure training arguments
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    overwrite_output_dir=True,
                    max_steps=total_steps,
                    eval_strategy="steps" if self.dev_dataset is not None else "no",
                    save_strategy="steps" if self.dev_dataset is not None else "no",
                    logging_strategy="steps",
                    per_device_train_batch_size=per_device_batch_size,
                    per_device_eval_batch_size=per_device_batch_size,
                    eval_steps=save_steps,
                    save_steps=save_steps,
                    logging_steps=save_steps,
                    save_total_limit=1,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    fp16=self.config["training"]["fp16"],
                    learning_rate=self.config["training"]["learning_rate"],
                    gradient_checkpointing=self.config["training"]["gradient_checkpointing"],
                    dataloader_num_workers=self.config["training"].get("num_workers", 0),
                    remove_unused_columns=False if self.mode in ["multiclass", "multilabel"] else True,
                    prediction_loss_only=True if self.mode == "mlm" else False,
                )
                
                # Initialize trainer
                if self.mode in ["multiclass", "multilabel"]:
                    callbacks = [EvalLossCallback()] if self.mode == "multiclass" else []
                    
                    trainer = WeightedTrainer(
                        model=self.model,
                        args=training_args,
                        train_dataset=self.train_dataset,
                        eval_dataset=self.dev_dataset,
                        tokenizer=self.tokenizer,
                        data_collator=self.data_collator,
                        class_weights=self.class_weights,
                        problem_type=self.mode,
                        compute_metrics=self.metrics_calculator.compute_metrics,
                        callbacks=callbacks,
                    )
                else:  # mlm
                    trainer = Trainer(
                        model=self.model,
                        args=training_args,
                        data_collator=self.data_collator,
                        train_dataset=self.train_dataset,
                        eval_dataset=self.dev_dataset,
                        tokenizer=self.tokenizer,
                    )
                
                # Train model
                self.logger.info("Starting training...")
                trainer.train()
                
                # Save the final model and tokenizer
                self.logger.info(f"Saving model and tokenizer to: {output_dir}")
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                
                # Log model artifacts to MLflow
                if self.config["mlflow"]["enabled"]:
                    mlflow.log_artifacts(output_dir, artifact_path="model")
                
                # Final evaluation
                if self.dev_dataset is not None:
                    eval_metrics = trainer.evaluate()
                    self.logger.info(f"Final evaluation metrics: {eval_metrics}")
                
                self.logger.info("Training complete!")
                
        finally:
            # Cleanup MLflow server if it was started by this process
            if self.config["mlflow"]["enabled"] and self.config["mlflow"].get("autostart", True):
                self.mlflow_manager.cleanup()

    # ========================================================================
    # Inference Methods
    # ========================================================================

    def load_inference_data(self) -> pd.DataFrame:
        """Load and preprocess the input data for inference."""
        self.logger.info(f"Loading data from {self.inference_config.input_file}")
        df = pd.read_csv(self.inference_config.input_file)
        
        # Check if the text column exists
        if self.inference_config.text_column not in df.columns:
            self.logger.error(f"Text column '{self.inference_config.text_column}' not found in the data")
            raise ValueError(f"Text column '{self.inference_config.text_column}' not found in the data")
        
        # Drop rows with NaN in the text column
        initial_length = len(df)
        df = df.dropna(subset=[self.inference_config.text_column])
        dropped_rows = initial_length - len(df)
        
        if dropped_rows > 0:
            self.logger.warning(f"Dropped {dropped_rows} rows with NaN values in text column")
        
        self.logger.info(f"Total examples loaded: {len(df)}")
        return df

    def run_multiclass_inference(self, df: pd.DataFrame) -> Tuple[List[int], List[Dict]]:
        """Run multiclass inference on the loaded data."""
        texts = df[self.inference_config.text_column].tolist()
        self.logger.info(f"Running multiclass inference on {len(texts)} texts")
        
        # Create dataset and dataloader
        dataset = TextDataset(texts, self.tokenizer, self.inference_config.max_length)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.inference_config.batch_size, 
            collate_fn=dataset.collate_fn
        )
        
        predictions = []
        predicted_confidences = []
        
        # Process data in batches with progress bar
        for batch_encodings in tqdm(dataloader, desc="Inference progress"):
            # Move tensors to the specified device
            batch_encodings = {k: v.to(self.inference_config.device) for k, v in batch_encodings.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch_encodings)
                logits = outputs.logits
                # Apply softmax to obtain probabilities for multiclass
                probs = torch.softmax(logits, dim=1)
                # Get the class with highest probability
                pred_classes = torch.argmax(probs, dim=1).cpu().numpy()
                
            # Process each sample in the batch
            for j in range(len(pred_classes)):
                pred_class = int(pred_classes[j])
                row_probs = probs[j].cpu().numpy().tolist()
                
                predictions.append(pred_class)
                # Store confidence score for the predicted class
                pred_conf = {
                    self.inference_config.id2label[pred_class]: row_probs[pred_class]
                }
                predicted_confidences.append(pred_conf)
            
            # Free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.logger.info(f"Inference complete. Total predictions: {len(predictions)}")
        return predictions, predicted_confidences

    def run_multilabel_inference(self, df: pd.DataFrame) -> Tuple[List[List[int]], List[Dict]]:
        """Run multilabel inference on the loaded data."""
        texts = df[self.inference_config.text_column].tolist()
        self.logger.info(f"Running multilabel inference on {len(texts)} texts")
        
        # Create dataset and dataloader
        dataset = TextDataset(texts, self.tokenizer, self.inference_config.max_length)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.inference_config.batch_size, 
            collate_fn=dataset.collate_fn
        )
        
        predictions = []
        predicted_confidences = []
        
        # Process data in batches with progress bar
        for batch_encodings in tqdm(dataloader, desc="Inference progress"):
            # Move tensors to the specified device
            batch_encodings = {k: v.to(self.inference_config.device) for k, v in batch_encodings.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch_encodings)
                logits = outputs.logits
                # Apply sigmoid to obtain probabilities
                probs = torch.sigmoid(logits)
                # Determine binary predictions using the threshold
                binary_preds = (probs >= self.inference_config.threshold).int().cpu().numpy()
            
            # Process each sample in the batch
            for j in range(len(binary_preds)):
                row_preds = binary_preds[j].tolist()
                row_probs = probs[j].cpu().numpy().tolist()
                
                predictions.append(row_preds)
                # Store confidence scores for positive predictions
                pred_conf = {
                    self.inference_config.id2label[i]: row_probs[i] 
                    for i in range(len(row_preds)) 
                    if row_preds[i] == 1
                }
                predicted_confidences.append(pred_conf)
            
            # Free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.logger.info(f"Inference complete. Total predictions: {len(predictions)}")
        return predictions, predicted_confidences

    def map_multiclass_predictions(self, predictions: List[int]) -> List[str]:
        """Map numeric predictions to human-readable labels for multiclass."""
        predicted_labels = [self.inference_config.id2label[pred] for pred in predictions]
        return predicted_labels

    def map_multilabel_predictions(self, predictions: List[List[int]]) -> List[List[str]]:
        """Map binary predictions to human-readable labels for multilabel."""
        predicted_labels = []
        for preds in predictions:
            labels = [self.inference_config.id2label[i] for i, flag in enumerate(preds) if flag == 1]
            predicted_labels.append(labels)
        return predicted_labels

    def save_inference_results(self, df: pd.DataFrame, predicted_labels, predicted_confidences) -> None:
        """Save the inference results to a CSV file."""
        # Verify lengths match before assignment
        if len(predicted_labels) != len(df) or len(predicted_confidences) != len(df):
            self.logger.error(
                f"Mismatch in number of predictions ({len(predicted_labels)}) and DataFrame rows ({len(df)})"
            )
            raise ValueError("Prediction length mismatch with DataFrame")
        
        # Add predictions to DataFrame
        result_df = df.copy()
        
        if self.mode == "multiclass-inference":
            result_df["predicted_label"] = predicted_labels
            result_df["predicted_confidence"] = predicted_confidences
        else:  # multilabel-inference
            result_df["predicted_labels"] = predicted_labels
            result_df["predicted_confidences"] = predicted_confidences
        
        # Save to CSV
        self.logger.info(f"Saving results to {self.inference_config.output_file}")
        result_df.to_csv(self.inference_config.output_file, index=False)
        self.logger.info("Results saved successfully")

    def run_inference(self):
        """Execute the full inference pipeline."""
        df = self.load_inference_data()
        
        if self.mode == "multiclass-inference":
            predictions, confidences = self.run_multiclass_inference(df)
            predicted_labels = self.map_multiclass_predictions(predictions)
        else:  # multilabel-inference
            predictions, confidences = self.run_multilabel_inference(df)
            predicted_labels = self.map_multilabel_predictions(predictions)
        
        self.save_inference_results(df, predicted_labels, confidences)

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    def run(self):
        """Execute the pipeline based on the mode."""
        if self.is_inference:
            self.run_inference()
        else:
            self.train()


# ============================================================================
# Utility Functions
# ============================================================================

def register_signal_handlers(pipeline: Optional[UnifiedTransformerPipeline] = None):
    """Register signal handlers for graceful shutdown."""
    
    def signal_handler(sig, frame):
        print("\nReceived termination signal. Cleaning up...")
        if pipeline and hasattr(pipeline, "mlflow_manager"):
            pipeline.mlflow_manager.cleanup()
        print("Cleanup complete. Exiting.")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # termination signal


def create_config_template(output_path: str, mode: str = "all") -> None:
    """Create a template configuration YAML file."""
    if mode in ["multiclass-inference", "multilabel-inference"]:
        template = f"""# Configuration for {mode.replace('-', ' ').title()}

# Inference configuration
inference:
  input_file: "data/test.csv"  # CSV file with text to classify
  output_file: "predictions.csv"  # Where to save predictions
  model_path: "path/to/trained/model"  # Path to trained model directory
  text_column: "text"  # Column name containing text
  {"threshold: 0.5  # Threshold for positive predictions" if mode == "multilabel-inference" else ""}
  batch_size: 32  # Batch size for inference
  max_length: 512  # Maximum sequence length
  device: "cuda"  # Device to use (cuda or cpu)
  
  # Optional: Override model's id2label mapping
  # id2label:
  #   '0': 'CLASS_A'
  #   '1': 'CLASS_B'
  #   '2': 'CLASS_C'

# Logging configuration
logging:
  level: "INFO"
  file: null  # Optional log file
"""
    elif mode == "mlm":
        template = """# Configuration for Masked Language Model pretraining

# Model configuration
model:
  name: "bert-base-uncased"  # Pre-trained model to use

# Data configuration
data:
  path: "data/texts.csv"  # CSV file with text data
  text_column: "text"  # Column name containing text
  max_length: 512  # Max sequence length (or "auto")

# Training configuration
training:
  batch_size: 16
  epochs: 3
  learning_rate: 5e-5
  fp16: true
  gradient_checkpointing: true
  num_workers: 4

# MLM specific configuration
mlm:
  mlm_probability: 0.15  # Probability of masking tokens
  do_whole_word_mask: true  # Use whole word masking
  train_test_split: 0.05  # Validation split

# MLflow configuration
mlflow:
  enabled: true
  experiment_name: "mlm-pretraining"
  tracking_uri: null
  autostart: true
  port: 5000
  backend_store_uri: "./mlruns"

# Output configuration
output:
  dir: "models"

# Random seed
seed: 42
"""
    else:
        # Use the combined template
        template = """# Unified Configuration for All Modes
# Usage: 
#   Training:
#     python unified_model.py --mode multiclass -c config.yaml
#     python unified_model.py --mode multilabel -c config.yaml
#     python unified_model.py --mode mlm -c config.yaml
#   Inference:
#     python unified_model.py --mode multiclass-inference -c config.yaml
#     python unified_model.py --mode multilabel-inference -c config.yaml

# Model configuration (for training)
model:
  name: "bert-base-uncased"  # HuggingFace model to use

# Data configuration (for training)
data:
  path: "data/dataset.csv"  # Path to your CSV file
  text_column: "text"  # Name of the text column
  
  # For multiclass only:
  label_column: "category"  # Name of the label column
  
  # Data splitting
  test_size: 0.2
  random_state: 42
  max_length: 512  # Max sequence length (or "auto" for MLM)
  length_percentile: 95

# Training configuration
training:
  batch_size: 32  # Consider 16 for multilabel/MLM
  epochs: 10
  learning_rate: 5e-5
  val_split: 0.2
  fp16: true
  gradient_checkpointing: true
  num_workers: 0

# MLM specific configuration (only used for MLM mode)
mlm:
  mlm_probability: 0.15
  do_whole_word_mask: true
  train_test_split: 0.05

# Inference configuration (for inference modes)
inference:
  input_file: "data/test.csv"
  output_file: "predictions.csv"
  model_path: "path/to/trained/model"
  text_column: "text"
  threshold: 0.5  # For multilabel only
  batch_size: 32
  max_length: 512
  device: "cuda"  # or "cpu"
  
  # Optional: id2label mapping
  # id2label:
  #   '0': 'CLASS_A'
  #   '1': 'CLASS_B'

# MLflow configuration (for training)
mlflow:
  enabled: true
  experiment_name: "unified-transformer-experiments"
  tracking_uri: null
  autostart: true
  port: 5000
  backend_store_uri: "./mlruns"
  artifact_store: null

# Logging configuration
logging:
  level: "INFO"
  file: null

# Output configuration (for training)
output:
  dir: "models"

# Random seed (for training)
seed: 42
"""
    
    with open(output_path, "w") as f:
        f.write(template)
    print(f"Created template configuration at: {output_path}")


def main():
    """Main entry point for command line execution."""
    parser = argparse.ArgumentParser(
        description="Unified transformer model for training and inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--mode", "-m",
        type=str,
        required=True,
        choices=["multiclass", "multilabel", "mlm", "multiclass-inference", "multilabel-inference"],
        help="Operation mode: training (multiclass/multilabel/mlm) or inference (multiclass-inference/multilabel-inference)"
    )
    
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--no-mlflow", 
        action="store_true", 
        help="Disable MLflow tracking (training only)"
    )
    
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="Disable MLflow server autostart (training only)"
    )
    
    parser.add_argument(
        "--mlflow-port",
        type=int,
        default=5000,
        help="Port for MLflow tracking server if autostarted (training only)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--create-config-template",
        action="store_true",
        help="Create a template configuration file and exit"
    )
    
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
        help="Disable progress bar during inference"
    )
    
    # Override arguments (training only)
    override_group = parser.add_argument_group("Configuration Overrides (training only)")
    override_group.add_argument("--model", type=str, help="Override model name")
    override_group.add_argument("--data", type=str, help="Override data path")
    override_group.add_argument("--epochs", type=int, help="Override number of epochs")
    override_group.add_argument("--batch-size", type=int, help="Override batch size")
    override_group.add_argument("--learning-rate", "--lr", type=float, help="Override learning rate")
    
    args = parser.parse_args()
    
    # Create template if requested
    if args.create_config_template:
        create_config_template(args.config, args.mode)
        return
    
    # Load YAML config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    
    # Apply command line overrides
    is_inference = args.mode.endswith("-inference")
    
    if not is_inference:
        # Training mode overrides
        if "mlflow" not in config:
            config["mlflow"] = {}
        
        if args.no_mlflow:
            config["mlflow"]["enabled"] = False
        if args.no_autostart:
            config["mlflow"]["autostart"] = False
        if args.mlflow_port:
            config["mlflow"]["port"] = args.mlflow_port
            
        # Apply configuration overrides
        if args.model:
            config["model"]["name"] = args.model
        if args.data:
            config["data"]["path"] = args.data
        if args.epochs:
            config["training"]["epochs"] = args.epochs
        if args.batch_size:
            config["training"]["batch_size"] = args.batch_size
        if args.learning_rate:
            config["training"]["learning_rate"] = args.learning_rate
    
    if args.debug:
        config.setdefault("logging", {})["level"] = "DEBUG"
    
    # Disable progress bar if requested
    if args.disable_progress_bar and is_inference:
        # Monkey patch tqdm
        def noop_tqdm(*args, **kwargs):
            if args:
                return args[0]
            return []
        import tqdm as tqdm_module
        tqdm_module.tqdm = noop_tqdm
        tqdm = noop_tqdm
    
    # Print GPU availability
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Initialize and run
    try:
        pipeline = UnifiedTransformerPipeline(config, mode=args.mode)
        
        # Register signal handlers for graceful shutdown (training only)
        if not is_inference:
            register_signal_handlers(pipeline)
        
        pipeline.run()
    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()