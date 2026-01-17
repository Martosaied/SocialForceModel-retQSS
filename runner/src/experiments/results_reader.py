"""Results reading and parsing utilities for experiment analysis."""

import os
import re
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path


class ResultsReader:
    """Read and parse experiment results from directories."""
    
    def __init__(self, results_base_dir: str):
        self.results_base_dir = Path(results_base_dir)
    
    def get_result_directories(self) -> List[str]:
        """Get all result directory names."""
        if not self.results_base_dir.exists():
            return []
        return [d.name for d in self.results_base_dir.iterdir() if d.is_dir()]
    
    def parse_directory_name(self, dir_name: str, 
                            params_pattern: Optional[Dict[str, type]] = None) -> Dict[str, Any]:
        """Parse parameters from directory name (e.g., 'cell_size_5.0_impl_retqss')."""
        if params_pattern is None:
            params_pattern = {
                'width': float,
                'cell_size': float,
                'implementation': str,
                'neighborhood': int,
                'n': int,
                'R': float,
                'A': float,
                'B': float,
            }
        
        result = {}
        for param_name, param_type in params_pattern.items():
            pattern = rf'{param_name}_([0-9.]+|[a-zA-Z_]+)'
            match = re.search(pattern, dir_name)
            if match:
                value_str = match.group(1)
                try:
                    result[param_name] = param_type(value_str)
                except (ValueError, TypeError):
                    result[param_name] = value_str
        
        return result
    
    def read_metrics_csv(self, experiment_dir: str) -> Optional[pd.DataFrame]:
        """Read metrics.csv from experiment directory."""
        metrics_path = self.results_base_dir / experiment_dir / 'latest' / 'metrics.csv'
        if metrics_path.exists():
            return pd.read_csv(metrics_path)
        return None
    
    def aggregate_metrics(self, group_by_param: str, 
                         metric_columns: List[str] = None) -> Dict:
        """Aggregate metrics across all directories, grouped by parameter."""
        if metric_columns is None:
            metric_columns = ['time', 'clustering_based_groups', 'total_collisions']
        
        results = {}
        
        for dir_name in self.get_result_directories():
            df = self.read_metrics_csv(dir_name)
            if df is None:
                continue
            
            params = self.parse_directory_name(dir_name)
            group_key = params.get(group_by_param)
            
            if group_key is None:
                continue
            
            if group_key not in results:
                results[group_key] = {col: [] for col in metric_columns}
            
            for col in metric_columns:
                if col in df.columns:
                    results[group_key][col].extend(df[col].dropna().tolist())
        
        # Calculate statistics
        stats_results = {}
        for group_key, metrics in results.items():
            stats_results[group_key] = {}
            for metric_name, values in metrics.items():
                if values:
                    stats_results[group_key][metric_name] = {
                        'mean': pd.Series(values).mean(),
                        'std': pd.Series(values).std(),
                        'data': values
                    }
        
        return stats_results
    
    def get_metrics_by_params(self, params_filter: Dict[str, Any]) -> List[pd.DataFrame]:
        """Get metrics DataFrames matching specific parameter values."""
        matching_dfs = []
        
        for dir_name in self.get_result_directories():
            params = self.parse_directory_name(dir_name)
            
            if all(params.get(k) == v for k, v in params_filter.items()):
                df = self.read_metrics_csv(dir_name)
                if df is not None:
                    matching_dfs.append(df)
        
        return matching_dfs

