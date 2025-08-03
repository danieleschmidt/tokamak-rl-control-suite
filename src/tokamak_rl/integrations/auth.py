"""
Authentication and authorization services for tokamak RL control suite.

This module provides secure authentication and authorization for
accessing tokamak control systems and shared resources.
"""

import os
import jwt
import bcrypt
import secrets
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class User:
    """User representation for authentication."""
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


@dataclass
class AuthToken:
    """Authentication token representation."""
    token: str
    user: User
    expires_at: datetime
    scopes: List[str]


class AuthenticationProvider(ABC):
    """Base class for authentication providers."""
    
    @abstractmethod
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        pass
        
    @abstractmethod
    def create_user(self, username: str, email: str, password: str, 
                   roles: List[str] = None) -> User:
        """Create new user account."""
        pass
        
    @abstractmethod
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        pass


class LocalAuthProvider(AuthenticationProvider):
    """Local file-based authentication provider."""
    
    def __init__(self, users_file: str = "./data/users.json"):
        """
        Initialize local authentication provider.
        
        Args:
            users_file: Path to users database file
        """
        self.users_file = users_file
        self.users_db: Dict[str, Dict[str, Any]] = {}
        self._load_users()
        
    def _load_users(self) -> None:
        """Load users from file."""
        import json
        from pathlib import Path
        
        users_path = Path(self.users_file)
        
        if users_path.exists():
            try:
                with open(users_path, 'r') as f:
                    self.users_db = json.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load users file: {e}")
                self.users_db = {}
        else:
            # Create empty users database
            users_path.parent.mkdir(parents=True, exist_ok=True)
            self.users_db = {}
            self._save_users()
            
    def _save_users(self) -> None:
        """Save users to file."""
        import json
        from pathlib import Path
        
        users_path = Path(self.users_file)
        users_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(users_path, 'w') as f:
                json.dump(self.users_db, f, indent=2, default=str)
        except Exception as e:
            warnings.warn(f"Failed to save users file: {e}")
            
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
        
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        if username not in self.users_db:
            return None
            
        user_data = self.users_db[username]
        
        if not user_data.get('is_active', True):
            return None
            
        if not self._verify_password(password, user_data['password_hash']):
            return None
            
        # Update last login
        user_data['last_login'] = datetime.now().isoformat()
        self._save_users()
        
        return User(
            username=username,
            email=user_data['email'],
            roles=user_data.get('roles', []),
            permissions=user_data.get('permissions', []),
            created_at=datetime.fromisoformat(user_data['created_at']),
            last_login=datetime.now(),
            is_active=user_data.get('is_active', True)
        )
        
    def create_user(self, username: str, email: str, password: str,
                   roles: List[str] = None) -> User:
        """Create new user account."""
        if username in self.users_db:
            raise ValueError(f"User {username} already exists")
            
        if roles is None:
            roles = ['user']
            
        # Generate default permissions based on roles
        permissions = []
        if 'admin' in roles:
            permissions.extend(['read', 'write', 'admin', 'control'])
        elif 'operator' in roles:
            permissions.extend(['read', 'write', 'control'])
        elif 'researcher' in roles:
            permissions.extend(['read', 'write'])
        else:
            permissions.append('read')
            
        user_data = {
            'email': email,
            'password_hash': self._hash_password(password),
            'roles': roles,
            'permissions': permissions,
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'is_active': True
        }
        
        self.users_db[username] = user_data
        self._save_users()
        
        return User(
            username=username,
            email=email,
            roles=roles,
            permissions=permissions,
            created_at=datetime.now(),
            is_active=True
        )
        
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        if username not in self.users_db:
            return None
            
        user_data = self.users_db[username]
        
        return User(
            username=username,
            email=user_data['email'],
            roles=user_data.get('roles', []),
            permissions=user_data.get('permissions', []),
            created_at=datetime.fromisoformat(user_data['created_at']),
            last_login=datetime.fromisoformat(user_data['last_login']) if user_data.get('last_login') else None,
            is_active=user_data.get('is_active', True)
        )


class JWTAuthService:
    """JWT-based authentication service."""
    
    def __init__(self, secret_key: Optional[str] = None,
                 algorithm: str = 'HS256',
                 token_expiry: timedelta = timedelta(hours=24)):
        """
        Initialize JWT authentication service.
        
        Args:
            secret_key: Secret key for JWT signing
            algorithm: JWT algorithm
            token_expiry: Token expiration time
        """
        self.secret_key = secret_key or os.getenv('JWT_SECRET_KEY') or secrets.token_urlsafe(32)
        self.algorithm = algorithm
        self.token_expiry = token_expiry
        
        if not secret_key and not os.getenv('JWT_SECRET_KEY'):
            warnings.warn("Using generated JWT secret key. Set JWT_SECRET_KEY environment variable for production.")
            
    def create_token(self, user: User, scopes: List[str] = None) -> AuthToken:
        """
        Create JWT token for user.
        
        Args:
            user: User to create token for
            scopes: Token scopes/permissions
            
        Returns:
            Authentication token
        """
        if scopes is None:
            scopes = user.permissions
            
        expires_at = datetime.utcnow() + self.token_expiry
        
        payload = {
            'username': user.username,
            'email': user.email,
            'roles': user.roles,
            'permissions': user.permissions,
            'scopes': scopes,
            'exp': expires_at,
            'iat': datetime.utcnow(),
            'iss': 'tokamak-rl-control-suite'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return AuthToken(
            token=token,
            user=user,
            expires_at=expires_at,
            scopes=scopes
        )
        
    def verify_token(self, token: str) -> Optional[AuthToken]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded authentication token or None if invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            user = User(
                username=payload['username'],
                email=payload['email'],
                roles=payload['roles'],
                permissions=payload['permissions'],
                created_at=datetime.utcnow(),  # Not stored in token
                is_active=True
            )
            
            return AuthToken(
                token=token,
                user=user,
                expires_at=datetime.utcfromtimestamp(payload['exp']),
                scopes=payload.get('scopes', [])
            )
            
        except jwt.ExpiredSignatureError:
            warnings.warn("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            warnings.warn(f"Invalid JWT token: {e}")
            return None


class AuthenticationService:
    """Main authentication service combining providers and JWT."""
    
    def __init__(self, provider: Optional[AuthenticationProvider] = None,
                 jwt_service: Optional[JWTAuthService] = None):
        """
        Initialize authentication service.
        
        Args:
            provider: Authentication provider
            jwt_service: JWT service for token management
        """
        self.provider = provider or LocalAuthProvider()
        self.jwt_service = jwt_service or JWTAuthService()
        self.active_tokens: Dict[str, AuthToken] = {}
        
    def login(self, username: str, password: str, 
             scopes: List[str] = None) -> Optional[AuthToken]:
        """
        Authenticate user and create session token.
        
        Args:
            username: Username
            password: Password
            scopes: Requested token scopes
            
        Returns:
            Authentication token if successful
        """
        user = self.provider.authenticate(username, password)
        if not user:
            return None
            
        token = self.jwt_service.create_token(user, scopes)
        self.active_tokens[token.token] = token
        
        return token
        
    def verify_token(self, token: str) -> Optional[AuthToken]:
        """
        Verify authentication token.
        
        Args:
            token: Token to verify
            
        Returns:
            Verified token or None if invalid
        """
        # Check active tokens first
        if token in self.active_tokens:
            auth_token = self.active_tokens[token]
            if auth_token.expires_at > datetime.utcnow():
                return auth_token
            else:
                # Remove expired token
                del self.active_tokens[token]
                
        # Verify with JWT service
        return self.jwt_service.verify_token(token)
        
    def logout(self, token: str) -> bool:
        """
        Logout user by invalidating token.
        
        Args:
            token: Token to invalidate
            
        Returns:
            True if successful
        """
        if token in self.active_tokens:
            del self.active_tokens[token]
            return True
        return False
        
    def create_user(self, username: str, email: str, password: str,
                   roles: List[str] = None) -> User:
        """
        Create new user account.
        
        Args:
            username: Username
            email: Email address
            password: Password
            roles: User roles
            
        Returns:
            Created user
        """
        return self.provider.create_user(username, email, password, roles)
        
    def check_permission(self, token: str, required_permission: str) -> bool:
        """
        Check if token has required permission.
        
        Args:
            token: Authentication token
            required_permission: Required permission
            
        Returns:
            True if permission granted
        """
        auth_token = self.verify_token(token)
        if not auth_token:
            return False
            
        return required_permission in auth_token.scopes or 'admin' in auth_token.user.roles
        
    def require_permission(self, required_permission: str):
        """
        Decorator to require specific permission for function access.
        
        Args:
            required_permission: Required permission
            
        Returns:
            Decorator function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Extract token from kwargs or args
                token = kwargs.get('token') or (args[0] if args else None)
                
                if not token or not self.check_permission(token, required_permission):
                    raise PermissionError(f"Permission '{required_permission}' required")
                    
                return func(*args, **kwargs)
            return wrapper
        return decorator
        
    def get_user_info(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Get user information from token.
        
        Args:
            token: Authentication token
            
        Returns:
            User information dictionary
        """
        auth_token = self.verify_token(token)
        if not auth_token:
            return None
            
        return {
            'username': auth_token.user.username,
            'email': auth_token.user.email,
            'roles': auth_token.user.roles,
            'permissions': auth_token.user.permissions,
            'scopes': auth_token.scopes,
            'expires_at': auth_token.expires_at.isoformat()
        }


def create_authentication_service() -> AuthenticationService:
    """Create authentication service with default configuration."""
    provider = LocalAuthProvider()
    jwt_service = JWTAuthService()
    return AuthenticationService(provider, jwt_service)