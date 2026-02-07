# ============================================================================
# 📁 utils.py - SHARED UTILITIES
# ============================================================================

"""
UTILITY FUNCTIONS USED ACROSS ALL PHASES
EDIT HERE FOR CUSTOM DATA PROCESSING
"""

import json
import logging
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

class RagLogger:
    """Centralized logging for RAG tests with all required methods"""
    
    def __init__(self, log_file="rag_tests.log"):
        self.logger = logging.getLogger("RAG_Tests")
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_experiment_start(self, experiment_name: str, config: Dict):
        """Log experiment start - FIXED FOR PHASE 1"""
        self.logger.info(f"🚀 Starting experiment: {experiment_name}")
        self.logger.info(f"   Config: {json.dumps(config, indent=2)[:500]}...")
    
    def log_experiment_end(self, experiment_name: str, results: Dict):
        """Log experiment end with results - NEW METHOD"""
        self.logger.info(f"✅ Completed experiment: {experiment_name}")
        self.logger.info(f"   Results: {json.dumps(results, indent=2)[:500]}...")
    
    def log_step(self, step_name: str, details: str = None):
        """Log a step within an experiment - NEW METHOD"""
        if details:
            self.logger.info(f"   ↳ {step_name}: {details}")
        else:
            self.logger.info(f"   ↳ {step_name}")
    
    def log_metric(self, metric_name: str, value: Any):
        """Log a metric - NEW METHOD"""
        self.logger.info(f"   📊 {metric_name}: {value}")
    
    # Keep your original methods for backward compatibility
    def log_metrics(self, phase: str, metrics: Dict):
        """Log experiment metrics - OLD METHOD (keep for compatibility)"""
        self.logger.info(f"📊 {phase} Metrics: {json.dumps(metrics, indent=2)}")
    
    def log_error(self, phase: str, error: str):
        """Log errors"""
        self.logger.error(f"❌ {phase} Error: {error}")
    
    # Delegate standard logging methods
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def debug(self, message: str):
        self.logger.debug(message)

# EDIT HERE: Add custom data processing utilities
class DataProcessor:
    """Custom data processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # EDIT HERE: Add custom text cleaning logic
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.replace('\r', ' ').replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """Extract common grant sections"""
        # EDIT HERE: Customize section extraction for your grant format
        sections = {
            "abstract": "",
            "specific_aims": "",
            "background": "",
            "methods": "",
            "results": "",
            "discussion": "",
            "other": ""
        }
        
        # Simple section detection (customize for your documents)
        lines = text.split('\n')
        current_section = "other"
        current_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            if "abstract" in line_lower and len(line_lower.split()) < 5:
                if current_content:
                    sections[current_section] = " ".join(current_content)
                current_section = "abstract"
                current_content = []
            elif "specific aims" in line_lower or "aims" in line_lower:
                if current_content:
                    sections[current_section] = " ".join(current_content)
                current_section = "specific_aims"
                current_content = []
            # Add more section detection as needed...
            else:
                if line.strip():
                    current_content.append(line.strip())
        
        if current_content:
            sections[current_section] = " ".join(current_content)
        
        return sections

# Initialize logger globally with all required methods
logger = RagLogger()

# Add helper functions for backward compatibility
def log_experiment_start(experiment_name: str, config: Dict):
    """Wrapper for backward compatibility"""
    logger.log_experiment_start(experiment_name, config)

def log_experiment_end(experiment_name: str, results: Dict):
    """Wrapper for backward compatibility"""
    logger.log_experiment_end(experiment_name, results)

def log_step(step_name: str, details: str = None):
    """Wrapper for backward compatibility"""
    logger.log_step(step_name, details)

def log_metric(metric_name: str, value: Any):
    """Wrapper for backward compatibility"""
    logger.log_metric(metric_name, value)