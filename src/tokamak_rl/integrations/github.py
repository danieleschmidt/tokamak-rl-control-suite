"""
GitHub integration for model sharing and collaboration.

This module provides functionality to integrate with GitHub for
sharing trained models, experimental results, and collaboration.
"""

import os
import json
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path
import warnings
from datetime import datetime


class GitHubIntegration:
    """GitHub integration for tokamak RL control suite."""
    
    def __init__(self, token: Optional[str] = None, 
                 repo_owner: str = "tokamak-rl",
                 repo_name: str = "models"):
        """
        Initialize GitHub integration.
        
        Args:
            token: GitHub personal access token
            repo_owner: Repository owner username
            repo_name: Repository name for models
        """
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.base_url = "https://api.github.com"
        
        if not self.token:
            warnings.warn("GitHub token not provided. Some features will be unavailable.")
            
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests."""
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'tokamak-rl-control-suite'
        }
        
        if self.token:
            headers['Authorization'] = f'token {self.token}'
            
        return headers
        
    def upload_model(self, model_path: str, model_name: str,
                    description: str = "", tags: Optional[List[str]] = None,
                    metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Upload trained model to GitHub repository.
        
        Args:
            model_path: Path to model file
            model_name: Name for the model
            description: Model description
            tags: Model tags for categorization
            metrics: Performance metrics
            
        Returns:
            Upload result information
        """
        if not self.token:
            raise ValueError("GitHub token required for uploading models")
            
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Read model file
        with open(model_path, 'rb') as f:
            model_data = f.read()
            
        # Encode for GitHub API
        encoded_content = base64.b64encode(model_data).decode('utf-8')
        
        # Create model metadata
        metadata = {
            'name': model_name,
            'description': description,
            'tags': tags or [],
            'metrics': metrics or {},
            'upload_time': datetime.now().isoformat(),
            'file_size': len(model_data),
            'model_type': 'pytorch' if model_path.suffix == '.pt' else 'unknown'
        }
        
        # Upload files (model + metadata)
        upload_results = {}
        
        try:
            # Upload model file
            model_upload = self._upload_file(
                f"models/{model_name}/{model_path.name}",
                encoded_content,
                f"Upload model: {model_name}"
            )
            upload_results['model'] = model_upload
            
            # Upload metadata
            metadata_content = base64.b64encode(
                json.dumps(metadata, indent=2).encode('utf-8')
            ).decode('utf-8')
            
            metadata_upload = self._upload_file(
                f"models/{model_name}/metadata.json",
                metadata_content,
                f"Upload metadata for model: {model_name}"
            )
            upload_results['metadata'] = metadata_upload
            
            return {
                'success': True,
                'model_name': model_name,
                'uploads': upload_results,
                'metadata': metadata
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
            
    def _upload_file(self, file_path: str, content: str, message: str) -> Dict[str, Any]:
        """Upload file to GitHub repository."""
        import requests
        
        url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/contents/{file_path}"
        
        data = {
            'message': message,
            'content': content
        }
        
        response = requests.put(url, headers=self._get_headers(), json=data)
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            raise RuntimeError(f"Failed to upload file: {response.status_code} - {response.text}")
            
    def download_model(self, model_name: str, output_dir: str = "./models") -> Dict[str, Any]:
        """
        Download model from GitHub repository.
        
        Args:
            model_name: Name of model to download
            output_dir: Directory to save model
            
        Returns:
            Download result information
        """
        import requests
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download metadata first
            metadata_url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/contents/models/{model_name}/metadata.json"
            
            response = requests.get(metadata_url, headers=self._get_headers())
            
            if response.status_code != 200:
                raise RuntimeError(f"Model metadata not found: {response.status_code}")
                
            metadata_info = response.json()
            metadata_content = base64.b64decode(metadata_info['content']).decode('utf-8')
            metadata = json.loads(metadata_content)
            
            # Save metadata
            metadata_path = output_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Download model files
            models_url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/contents/models/{model_name}"
            
            response = requests.get(models_url, headers=self._get_headers())
            
            if response.status_code != 200:
                raise RuntimeError(f"Model files not found: {response.status_code}")
                
            files_info = response.json()
            downloaded_files = []
            
            for file_info in files_info:
                if file_info['name'] != 'metadata.json':
                    # Download model file
                    file_content = base64.b64decode(file_info['content'])
                    file_path = output_dir / f"{model_name}_{file_info['name']}"
                    
                    with open(file_path, 'wb') as f:
                        f.write(file_content)
                        
                    downloaded_files.append(str(file_path))
                    
            return {
                'success': True,
                'model_name': model_name,
                'metadata': metadata,
                'files': downloaded_files,
                'metadata_path': str(metadata_path)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
            
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models in repository.
        
        Returns:
            List of model information
        """
        import requests
        
        try:
            url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/contents/models"
            
            response = requests.get(url, headers=self._get_headers())
            
            if response.status_code != 200:
                return []
                
            contents = response.json()
            models = []
            
            for item in contents:
                if item['type'] == 'dir':
                    # Try to get metadata for each model
                    try:
                        metadata_url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/contents/models/{item['name']}/metadata.json"
                        
                        metadata_response = requests.get(metadata_url, headers=self._get_headers())
                        
                        if metadata_response.status_code == 200:
                            metadata_info = metadata_response.json()
                            metadata_content = base64.b64decode(metadata_info['content']).decode('utf-8')
                            metadata = json.loads(metadata_content)
                            
                            models.append({
                                'name': item['name'],
                                'metadata': metadata,
                                'url': item['html_url'] if 'html_url' in item else None
                            })
                        else:
                            # Model without metadata
                            models.append({
                                'name': item['name'],
                                'metadata': {},
                                'url': item['html_url'] if 'html_url' in item else None
                            })
                            
                    except Exception:
                        # Skip models with invalid metadata
                        continue
                        
            return models
            
        except Exception as e:
            warnings.warn(f"Failed to list models: {e}")
            return []
            
    def create_release(self, tag: str, name: str, description: str,
                      model_paths: List[str]) -> Dict[str, Any]:
        """
        Create GitHub release with model files.
        
        Args:
            tag: Release tag
            name: Release name
            description: Release description
            model_paths: Paths to model files to include
            
        Returns:
            Release creation result
        """
        if not self.token:
            raise ValueError("GitHub token required for creating releases")
            
        import requests
        
        try:
            # Create release
            release_url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/releases"
            
            release_data = {
                'tag_name': tag,
                'name': name,
                'body': description,
                'draft': False,
                'prerelease': False
            }
            
            response = requests.post(release_url, headers=self._get_headers(), json=release_data)
            
            if response.status_code != 201:
                raise RuntimeError(f"Failed to create release: {response.status_code} - {response.text}")
                
            release_info = response.json()
            upload_url = release_info['upload_url'].replace('{?name,label}', '')
            
            # Upload model files as release assets
            uploaded_assets = []
            
            for model_path in model_paths:
                model_path = Path(model_path)
                
                if not model_path.exists():
                    warnings.warn(f"Model file not found: {model_path}")
                    continue
                    
                with open(model_path, 'rb') as f:
                    file_data = f.read()
                    
                asset_url = f"{upload_url}?name={model_path.name}"
                
                asset_headers = self._get_headers()
                asset_headers['Content-Type'] = 'application/octet-stream'
                
                asset_response = requests.post(asset_url, headers=asset_headers, data=file_data)
                
                if asset_response.status_code == 201:
                    uploaded_assets.append(asset_response.json())
                else:
                    warnings.warn(f"Failed to upload asset {model_path.name}: {asset_response.status_code}")
                    
            return {
                'success': True,
                'release': release_info,
                'assets': uploaded_assets
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def share_experiment_results(self, experiment_name: str,
                               results: Dict[str, Any],
                               plots: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Share experiment results as GitHub issue or discussion.
        
        Args:
            experiment_name: Name of experiment
            results: Experiment results dictionary
            plots: Paths to plot images
            
        Returns:
            Sharing result
        """
        if not self.token:
            raise ValueError("GitHub token required for sharing results")
            
        import requests
        
        try:
            # Format results as markdown
            markdown_content = f"# Experiment Results: {experiment_name}\n\n"
            markdown_content += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Add metrics table
            if 'metrics' in results:
                markdown_content += "## Performance Metrics\n\n"
                markdown_content += "| Metric | Value |\n|--------|-------|\n"
                
                for metric, value in results['metrics'].items():
                    markdown_content += f"| {metric} | {value} |\n"
                    
                markdown_content += "\n"
                
            # Add configuration
            if 'config' in results:
                markdown_content += "## Configuration\n\n```json\n"
                markdown_content += json.dumps(results['config'], indent=2)
                markdown_content += "\n```\n\n"
                
            # Add plots if provided
            if plots:
                markdown_content += "## Plots\n\n"
                for plot_path in plots:
                    plot_name = Path(plot_path).name
                    markdown_content += f"![{plot_name}]({plot_path})\n\n"
                    
            # Create issue
            issue_url = f"{self.base_url}/repos/{self.repo_owner}/{self.repo_name}/issues"
            
            issue_data = {
                'title': f"Experiment Results: {experiment_name}",
                'body': markdown_content,
                'labels': ['experiment-results', 'tokamak-rl']
            }
            
            response = requests.post(issue_url, headers=self._get_headers(), json=issue_data)
            
            if response.status_code == 201:
                return {
                    'success': True,
                    'issue': response.json()
                }
            else:
                raise RuntimeError(f"Failed to create issue: {response.status_code} - {response.text}")
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def create_github_integration(token: Optional[str] = None) -> GitHubIntegration:
    """Factory function to create GitHub integration."""
    return GitHubIntegration(token=token)