"""
Database and data management components for tokamak RL control.

This module provides data persistence, caching, and experimental data
integration for the tokamak plasma control system.
"""

import sqlite3
import json
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import time
from datetime import datetime
import hashlib
from dataclasses import dataclass, asdict
import warnings

from .physics import PlasmaState, TokamakConfig


@dataclass
class ExperimentRecord:
    """Record structure for experimental data."""
    
    experiment_id: str
    tokamak: str
    timestamp: float
    discharge_number: Optional[int]
    plasma_state: Dict[str, Any]
    control_actions: List[float]
    safety_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class ExperimentDatabase:
    """SQLite database for storing experimental data and training results."""
    
    def __init__(self, db_path: str = "./data/tokamak_experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self._initialize_database()
        
    def _initialize_database(self) -> None:
        """Initialize database tables."""
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        
        # Create tables
        self._create_tables()
        
    def _create_tables(self) -> None:
        """Create database tables."""
        cursor = self.connection.cursor()
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT UNIQUE NOT NULL,
                tokamak TEXT NOT NULL,
                timestamp REAL NOT NULL,
                discharge_number INTEGER,
                created_at REAL NOT NULL,
                metadata TEXT
            )
        """)
        
        # Plasma states table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plasma_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                q_min REAL,
                plasma_beta REAL,
                shape_error REAL,
                elongation REAL,
                triangularity REAL,
                disruption_probability REAL,
                plasma_current REAL,
                state_data BLOB,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        """)
        
        # Control actions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS control_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                action_data BLOB,
                reward REAL,
                safety_modified BOOLEAN,
                violations TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        """)
        
        # Training sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                agent_type TEXT NOT NULL,
                tokamak TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                total_episodes INTEGER,
                total_steps INTEGER,
                best_reward REAL,
                final_reward REAL,
                hyperparameters TEXT,
                model_path TEXT
            )
        """)
        
        # Create indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiment_id ON plasma_states (experiment_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON plasma_states (timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_control_exp ON control_actions (experiment_id)")
        
        self.connection.commit()
        
    def store_experiment(self, record: ExperimentRecord) -> None:
        """Store experimental record in database."""
        cursor = self.connection.cursor()
        
        try:
            # Insert experiment record
            cursor.execute("""
                INSERT OR REPLACE INTO experiments 
                (experiment_id, tokamak, timestamp, discharge_number, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                record.experiment_id,
                record.tokamak,
                record.timestamp,
                record.discharge_number,
                time.time(),
                json.dumps(record.metadata)
            ))
            
            # Store plasma state
            state_blob = pickle.dumps(record.plasma_state)
            cursor.execute("""
                INSERT INTO plasma_states
                (experiment_id, step_number, timestamp, q_min, plasma_beta, 
                 shape_error, elongation, triangularity, disruption_probability,
                 plasma_current, state_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.experiment_id,
                0,  # Single step for now
                record.timestamp,
                record.plasma_state.get('q_min'),
                record.plasma_state.get('plasma_beta'),
                record.plasma_state.get('shape_error'),
                record.plasma_state.get('elongation'),
                record.plasma_state.get('triangularity'),
                record.plasma_state.get('disruption_probability'),
                record.plasma_state.get('plasma_current'),
                state_blob
            ))
            
            # Store control actions
            action_blob = pickle.dumps(record.control_actions)
            cursor.execute("""
                INSERT INTO control_actions
                (experiment_id, step_number, timestamp, action_data, 
                 safety_modified, violations)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                record.experiment_id,
                0,
                record.timestamp,
                action_blob,
                record.safety_metrics.get('action_modified', False),
                json.dumps(record.safety_metrics.get('violations', []))
            ))
            
            self.connection.commit()
            
        except Exception as e:
            self.connection.rollback()
            raise RuntimeError(f"Failed to store experiment: {e}")
            
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Retrieve experimental record by ID."""
        cursor = self.connection.cursor()
        
        # Get experiment metadata
        cursor.execute("""
            SELECT * FROM experiments WHERE experiment_id = ?
        """, (experiment_id,))
        
        exp_row = cursor.fetchone()
        if not exp_row:
            return None
            
        # Get plasma state
        cursor.execute("""
            SELECT * FROM plasma_states WHERE experiment_id = ? ORDER BY step_number LIMIT 1
        """, (experiment_id,))
        
        state_row = cursor.fetchone()
        if not state_row:
            return None
            
        # Get control actions
        cursor.execute("""
            SELECT * FROM control_actions WHERE experiment_id = ? ORDER BY step_number LIMIT 1
        """, (experiment_id,))
        
        action_row = cursor.fetchone()
        
        try:
            # Reconstruct record
            plasma_state = pickle.loads(state_row['state_data'])
            control_actions = pickle.loads(action_row['action_data']) if action_row else []
            violations = json.loads(action_row['violations']) if action_row else []
            
            safety_metrics = {
                'action_modified': action_row['safety_modified'] if action_row else False,
                'violations': violations
            }
            
            metadata = json.loads(exp_row['metadata']) if exp_row['metadata'] else {}
            
            return ExperimentRecord(
                experiment_id=exp_row['experiment_id'],
                tokamak=exp_row['tokamak'],
                timestamp=exp_row['timestamp'],
                discharge_number=exp_row['discharge_number'],
                plasma_state=plasma_state,
                control_actions=control_actions,
                safety_metrics=safety_metrics,
                metadata=metadata
            )
            
        except Exception as e:
            warnings.warn(f"Failed to deserialize experiment {experiment_id}: {e}")
            return None
            
    def query_experiments(self, tokamak: Optional[str] = None,
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None,
                         limit: int = 100) -> List[ExperimentRecord]:
        """Query experiments with filters."""
        cursor = self.connection.cursor()
        
        # Build query
        conditions = []
        params = []
        
        if tokamak:
            conditions.append("tokamak = ?")
            params.append(tokamak)
            
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
            
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
            
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            SELECT experiment_id FROM experiments 
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        params.append(limit)
        cursor.execute(query, params)
        
        experiment_ids = [row['experiment_id'] for row in cursor.fetchall()]
        
        # Retrieve full records
        experiments = []
        for exp_id in experiment_ids:
            record = self.get_experiment(exp_id)
            if record:
                experiments.append(record)
                
        return experiments
        
    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()


class EquilibriumCache:
    """High-performance cache for plasma equilibrium solutions."""
    
    def __init__(self, cache_dir: str = "./data/cache", max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        
        # In-memory cache for frequently accessed equilibria
        self.memory_cache: Dict[str, Tuple[PlasmaState, float]] = {}
        self.access_times: Dict[str, float] = {}
        
    def _generate_key(self, config: TokamakConfig, control_params: np.ndarray) -> str:
        """Generate cache key from configuration and control parameters."""
        # Create hash from configuration and control parameters
        config_str = f"{config.major_radius}_{config.minor_radius}_{config.toroidal_field}"
        params_str = "_".join([f"{p:.6f}" for p in control_params])
        combined = f"{config_str}_{params_str}"
        
        return hashlib.md5(combined.encode()).hexdigest()
        
    def get(self, config: TokamakConfig, control_params: np.ndarray) -> Optional[PlasmaState]:
        """Retrieve cached equilibrium solution."""
        key = self._generate_key(config, control_params)
        
        # Check memory cache first
        if key in self.memory_cache:
            state, cached_time = self.memory_cache[key]
            self.access_times[key] = time.time()
            return state
            
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    state = pickle.load(f)
                    
                # Add to memory cache
                self._add_to_memory_cache(key, state)
                return state
                
            except Exception as e:
                warnings.warn(f"Failed to load cached equilibrium {key}: {e}")
                
        return None
        
    def store(self, config: TokamakConfig, control_params: np.ndarray, 
              state: PlasmaState) -> None:
        """Store equilibrium solution in cache."""
        key = self._generate_key(config, control_params)
        
        # Store in memory cache
        self._add_to_memory_cache(key, state)
        
        # Store on disk
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            warnings.warn(f"Failed to cache equilibrium {key}: {e}")
            
    def _add_to_memory_cache(self, key: str, state: PlasmaState) -> None:
        """Add equilibrium to memory cache with LRU eviction."""
        current_time = time.time()
        
        # Check if cache is full
        if len(self.memory_cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.memory_cache[lru_key]
            del self.access_times[lru_key]
            
        self.memory_cache[key] = (state, current_time)
        self.access_times[key] = current_time
        
    def clear(self) -> None:
        """Clear all cached data."""
        self.memory_cache.clear()
        self.access_times.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                warnings.warn(f"Failed to remove cache file {cache_file}: {e}")
                
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        disk_files = list(self.cache_dir.glob("*.pkl"))
        
        return {
            'memory_entries': len(self.memory_cache),
            'disk_entries': len(disk_files),
            'total_size_mb': sum(f.stat().st_size for f in disk_files) / (1024 * 1024),
            'max_size': self.max_size
        }


class DataRepository:
    """Central repository for managing all tokamak RL data."""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.experiment_db = ExperimentDatabase(str(self.base_dir / "experiments.db"))
        self.equilibrium_cache = EquilibriumCache(str(self.base_dir / "cache"))
        
        # Create data directories
        self.models_dir = self.base_dir / "models"
        self.experimental_dir = self.base_dir / "experimental"
        self.training_dir = self.base_dir / "training"
        
        for directory in [self.models_dir, self.experimental_dir, self.training_dir]:
            directory.mkdir(exist_ok=True)
            
    def store_training_episode(self, session_id: str, episode: int,
                              plasma_states: List[PlasmaState],
                              actions: List[np.ndarray],
                              rewards: List[float],
                              safety_info: List[Dict[str, Any]]) -> None:
        """Store complete training episode data."""
        experiment_id = f"{session_id}_ep_{episode:06d}"
        
        # Create synthetic experiment record
        metadata = {
            'session_id': session_id,
            'episode': episode,
            'type': 'training',
            'steps': len(plasma_states)
        }
        
        if plasma_states:
            record = ExperimentRecord(
                experiment_id=experiment_id,
                tokamak="training",  # Will be overridden by actual config
                timestamp=time.time(),
                discharge_number=None,
                plasma_state=asdict(plasma_states[-1]),  # Final state
                control_actions=actions[-1].tolist() if actions else [],
                safety_metrics=safety_info[-1] if safety_info else {},
                metadata=metadata
            )
            
            self.experiment_db.store_experiment(record)
            
    def get_training_data(self, session_id: str, 
                         episodes: Optional[List[int]] = None) -> List[ExperimentRecord]:
        """Retrieve training data for analysis."""
        all_experiments = self.experiment_db.query_experiments()
        
        # Filter by session
        session_experiments = [
            exp for exp in all_experiments 
            if exp.metadata.get('session_id') == session_id
        ]
        
        # Filter by episodes if specified
        if episodes:
            session_experiments = [
                exp for exp in session_experiments
                if exp.metadata.get('episode') in episodes
            ]
            
        return session_experiments
        
    def cleanup_old_data(self, retention_days: int = 30) -> None:
        """Remove old training data beyond retention period."""
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        
        # Clean experiment database
        cursor = self.experiment_db.connection.cursor()
        cursor.execute("""
            DELETE FROM experiments WHERE created_at < ?
        """, (cutoff_time,))
        self.experiment_db.connection.commit()
        
        # Clean cache
        cache_files = list(self.equilibrium_cache.cache_dir.glob("*.pkl"))
        for cache_file in cache_files:
            if cache_file.stat().st_mtime < cutoff_time:
                try:
                    cache_file.unlink()
                except Exception as e:
                    warnings.warn(f"Failed to remove old cache file {cache_file}: {e}")
                    
    def export_data(self, output_path: str, 
                   format: str = "json") -> None:
        """Export data for external analysis."""
        output_path = Path(output_path)
        
        # Get all experiments
        experiments = self.experiment_db.query_experiments(limit=10000)
        
        if format.lower() == "json":
            data = [asdict(exp) for exp in experiments]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics."""
        cursor = self.experiment_db.connection.cursor()
        
        # Experiment counts
        cursor.execute("SELECT COUNT(*) as count FROM experiments")
        total_experiments = cursor.fetchone()['count']
        
        cursor.execute("""
            SELECT tokamak, COUNT(*) as count 
            FROM experiments 
            GROUP BY tokamak
        """)
        experiments_by_tokamak = dict(cursor.fetchall())
        
        # Cache statistics
        cache_stats = self.equilibrium_cache.stats()
        
        return {
            'total_experiments': total_experiments,
            'experiments_by_tokamak': experiments_by_tokamak,
            'cache_statistics': cache_stats,
            'data_directories': {
                'models': len(list(self.models_dir.glob("*"))),
                'experimental': len(list(self.experimental_dir.glob("*"))),
                'training': len(list(self.training_dir.glob("*")))
            }
        }
        
    def close(self) -> None:
        """Close all database connections."""
        self.experiment_db.close()


def create_data_repository(base_dir: str = "./data") -> DataRepository:
    """Factory function to create data repository."""
    return DataRepository(base_dir)