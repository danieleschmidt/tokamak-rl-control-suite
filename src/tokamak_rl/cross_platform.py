"""
Cross-platform compatibility and deployment support for tokamak RL control.

This module provides platform abstraction, environment detection, 
deployment utilities, and system-specific optimizations.
"""

import os
import sys
import platform
import subprocess
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create mock psutil for basic functionality
    class MockPsutil:
        @staticmethod
        def cpu_count(logical=True):
            return 4
        
        @staticmethod
        def virtual_memory():
            class MemInfo:
                total = 8 * 1024**3  # 8GB
            return MemInfo()
        
        @staticmethod
        def net_if_addrs():
            return {"eth0": []}
    
    psutil = MockPsutil()
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
import json

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    import kubernetes
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False


class PlatformType(Enum):
    """Supported platform types."""
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"
    FREEBSD = "freebsd"
    UNKNOWN = "unknown"


class ArchitectureType(Enum):
    """Supported CPU architectures."""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARM32 = "arm32"
    UNKNOWN = "unknown"


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    CONTAINER = "container"
    KUBERNETES = "kubernetes"
    EDGE = "edge"


@dataclass
class SystemInfo:
    """System information and capabilities."""
    platform: PlatformType
    architecture: ArchitectureType
    os_version: str
    python_version: str
    cpu_count: int
    memory_gb: float
    gpu_available: bool
    gpu_info: Optional[Dict[str, Any]] = None
    docker_available: bool = False
    kubernetes_available: bool = False
    network_interfaces: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.network_interfaces is None:
            self.network_interfaces = []


@dataclass
class PerformanceConfig:
    """Platform-specific performance configuration."""
    max_workers: int
    memory_limit_gb: Optional[float]
    cpu_affinity: Optional[List[int]]
    gpu_memory_fraction: float
    use_cuda: bool
    use_mps: bool  # Apple Metal Performance Shaders
    numa_aware: bool
    io_threads: int
    
    @classmethod
    def create_optimal(cls, system_info: SystemInfo) -> 'PerformanceConfig':
        """Create optimal configuration for the system."""
        # Calculate optimal worker count
        cpu_count = system_info.cpu_count
        memory_gb = system_info.memory_gb
        
        # Conservative worker calculation to avoid oversubscription
        max_workers = min(cpu_count, int(memory_gb // 2), 32)
        max_workers = max(1, max_workers)
        
        # Memory limit - reserve 2GB for system
        memory_limit = max(1.0, memory_gb - 2.0) if memory_gb > 4 else None
        
        # Platform-specific optimizations
        use_cuda = system_info.gpu_available and system_info.platform == PlatformType.LINUX
        use_mps = system_info.gpu_available and system_info.platform == PlatformType.MACOS
        
        # NUMA awareness on multi-socket systems
        numa_aware = cpu_count > 16 and system_info.platform == PlatformType.LINUX
        
        return cls(
            max_workers=max_workers,
            memory_limit_gb=memory_limit,
            cpu_affinity=None,  # Auto-detect
            gpu_memory_fraction=0.8,
            use_cuda=use_cuda,
            use_mps=use_mps,
            numa_aware=numa_aware,
            io_threads=min(4, max_workers)
        )


class SystemDetector:
    """System information detection and capabilities assessment."""
    
    @staticmethod
    def detect_platform() -> PlatformType:
        """Detect the current platform."""
        system = platform.system().lower()
        
        if system == "linux":
            return PlatformType.LINUX
        elif system == "windows":
            return PlatformType.WINDOWS
        elif system == "darwin":
            return PlatformType.MACOS
        elif system == "freebsd":
            return PlatformType.FREEBSD
        else:
            return PlatformType.UNKNOWN
    
    @staticmethod
    def detect_architecture() -> ArchitectureType:
        """Detect CPU architecture."""
        machine = platform.machine().lower()
        
        if machine in ["x86_64", "amd64"]:
            return ArchitectureType.X86_64
        elif machine in ["arm64", "aarch64"]:
            return ArchitectureType.ARM64
        elif machine.startswith("arm"):
            return ArchitectureType.ARM32
        else:
            return ArchitectureType.UNKNOWN
    
    @staticmethod
    def detect_gpu() -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Detect GPU availability and capabilities."""
        gpu_info = {"type": "none", "devices": []}
        
        # Try NVIDIA GPU detection
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_info["type"] = "nvidia"
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        name, memory = line.split(', ')
                        gpu_info["devices"].append({
                            "name": name.strip(),
                            "memory_mb": int(memory.strip())
                        })
                return True, gpu_info
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Try AMD GPU detection (Linux)
        try:
            if Path("/sys/class/drm").exists():
                drm_devices = list(Path("/sys/class/drm").glob("card*"))
                if drm_devices:
                    gpu_info["type"] = "amd"
                    gpu_info["devices"] = [{"name": "AMD GPU", "memory_mb": "unknown"}]
                    return True, gpu_info
        except Exception:
            pass
        
        # Try Intel GPU detection
        try:
            if Path("/dev/dri").exists():
                dri_devices = list(Path("/dev/dri").glob("renderD*"))
                if dri_devices:
                    gpu_info["type"] = "intel"
                    gpu_info["devices"] = [{"name": "Intel GPU", "memory_mb": "unknown"}]
                    return True, gpu_info
        except Exception:
            pass
        
        # Apple Silicon detection
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            gpu_info["type"] = "apple_silicon"
            gpu_info["devices"] = [{"name": "Apple GPU", "memory_mb": "unified"}]
            return True, gpu_info
        
        return False, None
    
    @staticmethod
    def detect_network_interfaces() -> List[Dict[str, str]]:
        """Detect network interfaces."""
        interfaces = []
        
        try:
            if_addrs = psutil.net_if_addrs()
            for interface_name, addresses in if_addrs.items():
                for addr in addresses:
                    if addr.family.name == 'AF_INET':  # IPv4
                        interfaces.append({
                            "name": interface_name,
                            "ip": addr.address,
                            "netmask": addr.netmask,
                            "type": "ipv4"
                        })
                    elif addr.family.name == 'AF_INET6':  # IPv6
                        interfaces.append({
                            "name": interface_name,
                            "ip": addr.address,
                            "netmask": addr.netmask,
                            "type": "ipv6"
                        })
        except Exception as e:
            warnings.warn(f"Failed to detect network interfaces: {e}")
        
        return interfaces
    
    @classmethod
    def get_system_info(cls) -> SystemInfo:
        """Get comprehensive system information."""
        platform_type = cls.detect_platform()
        architecture = cls.detect_architecture()
        
        # Basic system info
        os_version = platform.platform()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Hardware info
        cpu_count = psutil.cpu_count(logical=True)
        memory_bytes = psutil.virtual_memory().total
        memory_gb = memory_bytes / (1024**3)
        
        # GPU detection
        gpu_available, gpu_info = cls.detect_gpu()
        
        # Container/orchestration detection
        docker_available = DOCKER_AVAILABLE and cls._is_docker_available()
        kubernetes_available = KUBERNETES_AVAILABLE and cls._is_kubernetes_available()
        
        # Network interfaces
        network_interfaces = cls.detect_network_interfaces()
        
        return SystemInfo(
            platform=platform_type,
            architecture=architecture,
            os_version=os_version,
            python_version=python_version,
            cpu_count=cpu_count,
            memory_gb=round(memory_gb, 2),
            gpu_available=gpu_available,
            gpu_info=gpu_info,
            docker_available=docker_available,
            kubernetes_available=kubernetes_available,
            network_interfaces=network_interfaces
        )
    
    @staticmethod
    def _is_docker_available() -> bool:
        """Check if Docker is available."""
        try:
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False
    
    @staticmethod
    def _is_kubernetes_available() -> bool:
        """Check if Kubernetes is available."""
        try:
            # Check for kubectl
            subprocess.run(["kubectl", "version", "--client"], 
                         capture_output=True, timeout=5)
            return True
        except Exception:
            return False


class PathManager:
    """Cross-platform path management."""
    
    def __init__(self, system_info: SystemInfo):
        self.system_info = system_info
        self.platform = system_info.platform
    
    def get_app_data_dir(self, app_name: str = "tokamak_rl") -> Path:
        """Get platform-appropriate application data directory."""
        if self.platform == PlatformType.WINDOWS:
            base = Path(os.environ.get("APPDATA", "~"))
        elif self.platform == PlatformType.MACOS:
            base = Path("~/Library/Application Support")
        else:  # Linux and others
            base = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share"))
        
        return (base / app_name).expanduser()
    
    def get_config_dir(self, app_name: str = "tokamak_rl") -> Path:
        """Get platform-appropriate configuration directory."""
        if self.platform == PlatformType.WINDOWS:
            base = Path(os.environ.get("APPDATA", "~"))
        elif self.platform == PlatformType.MACOS:
            base = Path("~/Library/Preferences")
        else:  # Linux and others
            base = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config"))
        
        return (base / app_name).expanduser()
    
    def get_cache_dir(self, app_name: str = "tokamak_rl") -> Path:
        """Get platform-appropriate cache directory."""
        if self.platform == PlatformType.WINDOWS:
            base = Path(os.environ.get("TEMP", "~"))
        elif self.platform == PlatformType.MACOS:
            base = Path("~/Library/Caches")
        else:  # Linux and others
            base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache"))
        
        return (base / app_name).expanduser()
    
    def get_log_dir(self, app_name: str = "tokamak_rl") -> Path:
        """Get platform-appropriate log directory."""
        if self.platform == PlatformType.WINDOWS:
            base = Path(os.environ.get("PROGRAMDATA", "C:/ProgramData"))
            return base / app_name / "logs"
        elif self.platform == PlatformType.MACOS:
            return Path(f"~/Library/Logs/{app_name}").expanduser()
        else:  # Linux and others
            # Try system log directory first, fall back to user
            system_log = Path(f"/var/log/{app_name}")
            if system_log.parent.exists() and os.access(system_log.parent, os.W_OK):
                return system_log
            else:
                return Path(f"~/.local/share/{app_name}/logs").expanduser()
    
    def ensure_directories(self, app_name: str = "tokamak_rl") -> Dict[str, Path]:
        """Ensure all application directories exist."""
        directories = {
            "data": self.get_app_data_dir(app_name),
            "config": self.get_config_dir(app_name),
            "cache": self.get_cache_dir(app_name),
            "logs": self.get_log_dir(app_name)
        }
        
        for dir_type, path in directories.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                warnings.warn(f"Cannot create {dir_type} directory: {path}")
        
        return directories


class EnvironmentManager:
    """Environment configuration and deployment management."""
    
    def __init__(self, system_info: SystemInfo):
        self.system_info = system_info
        self.path_manager = PathManager(system_info)
    
    def detect_environment(self) -> DeploymentEnvironment:
        """Detect current deployment environment."""
        # Check for container environment
        if Path("/.dockerenv").exists() or os.environ.get("container") == "docker":
            return DeploymentEnvironment.CONTAINER
        
        # Check for Kubernetes
        if (os.environ.get("KUBERNETES_SERVICE_HOST") or 
            Path("/var/run/secrets/kubernetes.io").exists()):
            return DeploymentEnvironment.KUBERNETES
        
        # Check environment variables
        env_type = os.environ.get("DEPLOYMENT_ENV", "").lower()
        if env_type == "production":
            return DeploymentEnvironment.PRODUCTION
        elif env_type == "staging":
            return DeploymentEnvironment.STAGING
        elif env_type == "testing":
            return DeploymentEnvironment.TESTING
        elif env_type == "development":
            return DeploymentEnvironment.DEVELOPMENT
        
        # Check for edge/embedded indicators
        if (self.system_info.memory_gb < 4 or 
            self.system_info.architecture in [ArchitectureType.ARM32, ArchitectureType.ARM64]):
            return DeploymentEnvironment.EDGE
        
        # Default to development
        return DeploymentEnvironment.DEVELOPMENT
    
    def get_environment_config(self, env: Optional[DeploymentEnvironment] = None) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        if env is None:
            env = self.detect_environment()
        
        base_config = {
            "environment": env.value,
            "debug": env in [DeploymentEnvironment.DEVELOPMENT, DeploymentEnvironment.TESTING],
            "log_level": "DEBUG" if env == DeploymentEnvironment.DEVELOPMENT else "INFO",
            "enable_metrics": True,
            "enable_tracing": env != DeploymentEnvironment.PRODUCTION
        }
        
        # Environment-specific overrides
        if env == DeploymentEnvironment.PRODUCTION:
            base_config.update({
                "log_level": "WARNING",
                "enable_profiling": False,
                "cache_size_multiplier": 2.0,
                "security_mode": "strict"
            })
        elif env == DeploymentEnvironment.CONTAINER:
            base_config.update({
                "use_host_networking": False,
                "resource_limits_enabled": True,
                "health_check_enabled": True
            })
        elif env == DeploymentEnvironment.KUBERNETES:
            base_config.update({
                "service_discovery": "kubernetes",
                "config_source": "configmap",
                "secret_source": "kubernetes",
                "metrics_endpoint": "/metrics"
            })
        elif env == DeploymentEnvironment.EDGE:
            base_config.update({
                "resource_constrained": True,
                "cache_size_multiplier": 0.5,
                "background_processing": False,
                "reduced_functionality": True
            })
        
        return base_config
    
    def setup_environment(self) -> Dict[str, Any]:
        """Setup environment with platform-specific optimizations."""
        env = self.detect_environment()
        config = self.get_environment_config(env)
        performance_config = PerformanceConfig.create_optimal(self.system_info)
        
        # Ensure directories exist
        directories = self.path_manager.ensure_directories()
        
        # Set up environment variables
        env_vars = self._setup_environment_variables(env, directories)
        
        # Apply performance optimizations
        self._apply_performance_optimizations(performance_config)
        
        setup_info = {
            "environment": env.value,
            "system_info": self.system_info,
            "performance_config": performance_config,
            "directories": {str(k): str(v) for k, v in directories.items()},
            "environment_variables": env_vars,
            "config": config
        }
        
        return setup_info
    
    def _setup_environment_variables(self, env: DeploymentEnvironment, 
                                   directories: Dict[str, Path]) -> Dict[str, str]:
        """Setup environment variables."""
        env_vars = {}
        
        # Set application directories
        env_vars["TOKAMAK_RL_DATA_DIR"] = str(directories["data"])
        env_vars["TOKAMAK_RL_CONFIG_DIR"] = str(directories["config"])
        env_vars["TOKAMAK_RL_CACHE_DIR"] = str(directories["cache"])
        env_vars["TOKAMAK_RL_LOG_DIR"] = str(directories["logs"])
        
        # Set platform info
        env_vars["TOKAMAK_RL_PLATFORM"] = self.system_info.platform.value
        env_vars["TOKAMAK_RL_ARCH"] = self.system_info.architecture.value
        env_vars["TOKAMAK_RL_ENV"] = env.value
        
        # GPU configuration
        if self.system_info.gpu_available:
            env_vars["TOKAMAK_RL_GPU_AVAILABLE"] = "true"
            if self.system_info.gpu_info:
                env_vars["TOKAMAK_RL_GPU_TYPE"] = self.system_info.gpu_info["type"]
        
        # Apply to current environment
        for key, value in env_vars.items():
            os.environ[key] = value
        
        return env_vars
    
    def _apply_performance_optimizations(self, config: PerformanceConfig) -> None:
        """Apply platform-specific performance optimizations."""
        
        # CPU affinity (Linux only)
        if (self.system_info.platform == PlatformType.LINUX and 
            config.cpu_affinity and hasattr(os, 'sched_setaffinity')):
            try:
                os.sched_setaffinity(0, config.cpu_affinity)
            except OSError:
                warnings.warn("Failed to set CPU affinity")
        
        # Memory limits (if supported)
        if config.memory_limit_gb:
            try:
                import resource
                memory_limit_bytes = int(config.memory_limit_gb * 1024**3)
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
            except (ImportError, OSError):
                warnings.warn("Failed to set memory limits")
        
        # GPU optimizations
        if config.use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
            os.environ["CUDA_MEMORY_FRACTION"] = str(config.gpu_memory_fraction)
        
        if config.use_mps:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class ContainerUtils:
    """Container and orchestration utilities."""
    
    @staticmethod
    def create_dockerfile(target_platform: PlatformType = PlatformType.LINUX,
                         architecture: ArchitectureType = ArchitectureType.X86_64) -> str:
        """Generate Dockerfile for the application."""
        
        # Base image selection
        if architecture == ArchitectureType.ARM64:
            base_image = "python:3.9-slim-arm64v8"
        else:
            base_image = "python:3.9-slim"
        
        dockerfile = f"""# Multi-stage build for tokamak RL control system
FROM {base_image} as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TOKAMAK_RL_ENV=container

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash tokamak
WORKDIR /home/tokamak

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/

# Set ownership
RUN chown -R tokamak:tokamak /home/tokamak

# Switch to application user
USER tokamak

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import src.tokamak_rl; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "src.tokamak_rl"]
"""
        return dockerfile
    
    @staticmethod
    def create_kubernetes_manifest(app_name: str = "tokamak-rl",
                                 namespace: str = "default",
                                 replicas: int = 1) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest."""
        
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment", 
            "metadata": {
                "name": app_name,
                "namespace": namespace,
                "labels": {
                    "app": app_name,
                    "version": "v1"
                }
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {
                        "app": app_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": app_name,
                            "version": "v1"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": app_name,
                            "image": f"{app_name}:latest",
                            "ports": [{
                                "containerPort": 8080,
                                "name": "http"
                            }],
                            "env": [
                                {
                                    "name": "TOKAMAK_RL_ENV",
                                    "value": "kubernetes"
                                },
                                {
                                    "name": "POD_NAME",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "metadata.name"
                                        }
                                    }
                                },
                                {
                                    "name": "POD_NAMESPACE",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "metadata.namespace"
                                        }
                                    }
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "100m",
                                    "memory": "256Mi"
                                },
                                "limits": {
                                    "cpu": "1000m",
                                    "memory": "2Gi"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }],
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 1000
                        }
                    }
                }
            }
        }
        
        return manifest


# Global instances
_system_info = None
_environment_manager = None


def get_system_info() -> SystemInfo:
    """Get cached system information."""
    global _system_info
    if _system_info is None:
        _system_info = SystemDetector.get_system_info()
    return _system_info


def get_environment_manager() -> EnvironmentManager:
    """Get environment manager instance."""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = EnvironmentManager(get_system_info())
    return _environment_manager


def setup_cross_platform_environment() -> Dict[str, Any]:
    """Setup cross-platform environment with optimal configuration."""
    return get_environment_manager().setup_environment()