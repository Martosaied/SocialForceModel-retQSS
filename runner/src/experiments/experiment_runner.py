"""Experiment execution helper utilities."""

from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import copy_results_to_latest


class ExperimentRunner:
    """Helper for common experiment execution patterns."""
    
    @staticmethod
    def run_standard_experiment(config: dict, output_dir: str, model_name: str, 
                                 compile_c: bool = True, 
                                 compile_model_flag: bool = True,
                                 copy_results: bool = True) -> None:
        """Run experiment with compilation and result copying."""
        if compile_c:
            compile_c_code()
        
        if compile_model_flag:
            compile_model(model_name)
        
        run_experiment(config, output_dir, model_name, 
                      plot=False, copy_results=copy_results)
        
        copy_results_to_latest(output_dir)

