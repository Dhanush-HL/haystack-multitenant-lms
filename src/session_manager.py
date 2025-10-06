"""
Session Manager for HayStack Multi-Tenant LMS
Handles user sessions, authentication state, and context management
"""

import uuid
import time
import json
import logging
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class UserSession:
    """User session data"""
    session_id: str
    user_id: str
    tenant_key: str
    created_at: datetime
    last_activity: datetime
    user_roles: Set[str]
    preferences: Dict[str, Any]
    is_active: bool = True

class SessionManager:
    """Manages user sessions across multiple tenants"""
    
    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, UserSession] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of session_ids
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self._lock = Lock()
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
        
        logger.info(f"✅ Session manager initialized (timeout: {session_timeout_minutes}min)")
    
    def _start_cleanup_task(self):
        """Start background task to clean expired sessions"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    self.cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Session cleanup error: {e}")
        
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            # No event loop running
            pass
    
    def create_session(self, 
                      user_id: str, 
                      tenant_key: str,
                      user_roles: Set[str] = None,
                      preferences: Dict[str, Any] = None) -> str:
        """Create new user session"""
        
        session_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            tenant_key=tenant_key,
            created_at=current_time,
            last_activity=current_time,
            user_roles=user_roles or set(),
            preferences=preferences or {}
        )
        
        with self._lock:
            # Store session
            self.sessions[session_id] = session
            
            # Track user sessions
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
        
        logger.info(f"✅ Created session {session_id[:8]}... for user {user_id} in tenant {tenant_key}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        with self._lock:
            session = self.sessions.get(session_id)
            
            if session and session.is_active:
                # Check if session expired
                if datetime.now() - session.last_activity > self.session_timeout:
                    self._invalidate_session(session_id)
                    return None
                
                # Update last activity
                session.last_activity = datetime.now()
                return session
            
            return None
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity timestamp"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session and session.is_active:
                session.last_activity = datetime.now()
                return True
            return False
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a specific session"""
        with self._lock:
            return self._invalidate_session(session_id)
    
    def _invalidate_session(self, session_id: str) -> bool:
        """Internal method to invalidate session (assumes lock held)"""
        session = self.sessions.get(session_id)
        if session:
            session.is_active = False
            
            # Remove from user sessions tracking
            if session.user_id in self.user_sessions:
                self.user_sessions[session.user_id].discard(session_id)
                if not self.user_sessions[session.user_id]:
                    del self.user_sessions[session.user_id]
            
            logger.info(f"🔒 Invalidated session {session_id[:8]}... for user {session.user_id}")
            return True
        
        return False
    
    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user"""
        count = 0
        with self._lock:
            if user_id in self.user_sessions:
                session_ids = self.user_sessions[user_id].copy()
                for session_id in session_ids:
                    if self._invalidate_session(session_id):
                        count += 1
        
        logger.info(f"🔒 Invalidated {count} sessions for user {user_id}")
        return count
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        with self._lock:
            for session_id, session in self.sessions.items():
                if (session.is_active and 
                    current_time - session.last_activity > self.session_timeout):
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                self._invalidate_session(session_id)
                del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """Get all active sessions for a user"""
        sessions = []
        with self._lock:
            if user_id in self.user_sessions:
                for session_id in self.user_sessions[user_id]:
                    session = self.sessions.get(session_id)
                    if session and session.is_active:
                        sessions.append(session)
        
        return sessions
    
    def get_tenant_sessions(self, tenant_key: str) -> List[UserSession]:
        """Get all active sessions for a tenant"""
        sessions = []
        with self._lock:
            for session in self.sessions.values():
                if session.tenant_key == tenant_key and session.is_active:
                    sessions.append(session)
        
        return sessions
    
    def update_user_roles(self, session_id: str, roles: Set[str]) -> bool:
        """Update user roles in session"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session and session.is_active:
                session.user_roles = roles
                logger.info(f"🔄 Updated roles for session {session_id[:8]}...")
                return True
            return False
    
    def update_user_preferences(self, session_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences in session"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session and session.is_active:
                session.preferences.update(preferences)
                logger.info(f"⚙️ Updated preferences for session {session_id[:8]}...")
                return True
            return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        with self._lock:
            total_sessions = len(self.sessions)
            active_sessions = sum(1 for s in self.sessions.values() if s.is_active)
            
            # Count by tenant
            tenant_counts = {}
            for session in self.sessions.values():
                if session.is_active:
                    tenant_counts[session.tenant_key] = tenant_counts.get(session.tenant_key, 0) + 1
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "unique_users": len(self.user_sessions),
                "tenant_distribution": tenant_counts,
                "session_timeout_minutes": self.session_timeout.total_seconds() / 60
            }
    
    def export_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export session data as dictionary"""
        session = self.get_session(session_id)
        if session:
            data = asdict(session)
            # Convert datetime to ISO format
            data['created_at'] = session.created_at.isoformat()
            data['last_activity'] = session.last_activity.isoformat()
            # Convert set to list for JSON serialization
            data['user_roles'] = list(session.user_roles)
            return data
        return None
    
    def shutdown(self):
        """Cleanup on shutdown"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        with self._lock:
            active_count = sum(1 for s in self.sessions.values() if s.is_active)
            self.sessions.clear()
            self.user_sessions.clear()
        
        logger.info(f"🛑 Session manager shutdown - cleared {active_count} active sessions")

# Factory functions
def create_session_manager(timeout_minutes: int = 30) -> SessionManager:
    """Create session manager with specified timeout"""
    return SessionManager(session_timeout_minutes=timeout_minutes)

# Context manager for session
class SessionContext:
    """Context manager for automatic session handling"""
    
    def __init__(self, session_manager: SessionManager, session_id: str):
        self.session_manager = session_manager
        self.session_id = session_id
        self.session = None
    
    def __enter__(self) -> Optional[UserSession]:
        self.session = self.session_manager.get_session(self.session_id)
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session_manager.update_session_activity(self.session_id)