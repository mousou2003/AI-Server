#!/usr/bin/env python3
"""
Configuration Loader for Qwen Churn Assistant

This module provides centralized configuration loading to avoid duplication
between Docker Compose files and Python code.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List


class QwenConfig:
    """Configuration loader for Qwen Churn Assistant"""
    
    def __init__(self, config_file: str = "qwen_config.yml"):
        """
        Initialize configuration loader
        
        Args:
            config_file (str): Path to YAML configuration file
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    @property
    def project_name(self) -> str:
        """Get project name"""
        return self.config['project']['name']
    
    @property
    def project_description(self) -> str:
        """Get project description"""
        return self.config['project']['description']
    
    def get_container_config(self, service: str) -> Dict[str, Any]:
        """
        Get container configuration for a service
        
        Args:
            service (str): Service name ('ollama' or 'webui')
            
        Returns:
            Dict containing container configuration
        """
        return self.config['containers'][service]
    
    def get_model_config(self, cpu_mode: bool = False, large_model: bool = False) -> Dict[str, Any]:
        """
        Get model configuration based on mode
        
        Args:
            cpu_mode (bool): If True, use CPU-optimized model
            large_model (bool): If True, force large model even in CPU mode
            
        Returns:
            Dict containing model configuration
        """
        if cpu_mode and not large_model:
            return self.config['models']['cpu_small']
        else:
            return self.config['models']['gpu_large']
    
    def get_compose_files(self, cpu_mode: bool = False) -> List[str]:
        """
        Get list of compose files to use
        
        Args:
            cpu_mode (bool): If True, exclude GPU override
            
        Returns:
            List of compose file paths
        """
        files = self.config['compose_files']['base']
        
        if not cpu_mode:
            return files + [self.config['compose_files']['overrides']['gpu']] + [self.config['compose_files']['overrides']['qwen']]
        else:
            return files + [self.config['compose_files']['overrides']['qwen']]
    
    def get_volume_names(self) -> List[str]:
        """Get list of volume names"""
        return self.config['volumes']
    
    def get_urls(self) -> Dict[str, str]:
        """Get service URLs"""
        ollama_port = self.config['containers']['ollama']['port']
        webui_port = self.config['containers']['webui']['external_port']
        
        return {
            'ollama': f"http://localhost:{ollama_port}",
            'webui': f"http://localhost:{webui_port}"
        }
