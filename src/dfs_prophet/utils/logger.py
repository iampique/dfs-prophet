"""
Logging utilities for DFS Prophet.

This module provides structured logging setup with JSON formatting for production,
console formatting for development, performance timing decorators, and error tracking.
"""

import json
import logging
import logging.config
import time
import traceback
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union
from datetime import datetime

from ..config import get_settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Console formatter for development with colored output."""
    
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
        "RESET": "\033[0m"      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output with colors."""
        # Get color for log level
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        
        # Format message
        message = record.getMessage()
        
        # Add module and function info
        module_info = f"{record.module}.{record.funcName}:{record.lineno}"
        
        # Format the log entry
        log_entry = f"{color}[{record.levelname:8}]{reset} {timestamp} | {module_info} | {message}"
        
        # Add exception info if present
        if record.exc_info:
            log_entry += f"\n{color}Exception:{reset}\n{traceback.format_exception(*record.exc_info)}"
        
        return log_entry


class PerformanceLogger:
    """Performance logging utilities with timing and metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._timers: Dict[str, float] = {}
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._timers[name] = time.time()
        self.logger.debug(f"Started timer: {name}")
    
    def end_timer(self, name: str, log_level: str = "INFO") -> Optional[float]:
        """End a named timer and log the duration."""
        if name not in self._timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return None
        
        duration = time.time() - self._timers[name]
        duration_ms = duration * 1000
        
        # Log based on duration
        if duration < 0.1:
            level = "DEBUG"
        elif duration < 1.0:
            level = "INFO"
        elif duration < 5.0:
            level = "WARNING"
        else:
            level = "ERROR"
        
        # Override with specified level if provided
        if log_level:
            level = log_level.upper()
        
        log_message = f"Timer '{name}' completed in {duration_ms:.2f}ms ({duration:.3f}s)"
        
        if level == "DEBUG":
            self.logger.debug(log_message)
        elif level == "INFO":
            self.logger.info(log_message)
        elif level == "WARNING":
            self.logger.warning(log_message)
        elif level == "ERROR":
            self.logger.error(log_message)
        
        del self._timers[name]
        return duration
    
    @contextmanager
    def timer(self, name: str, log_level: str = "INFO"):
        """Context manager for timing operations."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name, log_level)


def performance_timer(name: Optional[str] = None, log_level: str = "INFO"):
    """Decorator for timing function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or f"{func.__module__}.{func.__name__}"
            logger = get_logger(func.__module__)
            perf_logger = PerformanceLogger(logger)
            
            with perf_logger.timer(timer_name, log_level):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class ErrorTracker:
    """Error tracking and reporting utilities."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._error_counts: Dict[str, int] = {}
    
    def track_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Track an error with optional context."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Increment error count
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        
        # Prepare error data
        error_data = {
            "error_type": error_type,
            "error_message": error_message,
            "error_count": self._error_counts[error_type],
            "traceback": traceback.format_exc()
        }
        
        if context:
            error_data["context"] = context
        
        # Log error with structured data
        self.logger.error(
            f"Error tracked: {error_type} - {error_message}",
            extra={"error_data": error_data}
        )
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of tracked errors."""
        return self._error_counts.copy()
    
    def reset_error_counts(self) -> None:
        """Reset error count tracking."""
        self._error_counts.clear()
        self.logger.info("Error counts reset")


def setup_logging() -> None:
    """Setup logging configuration based on environment."""
    settings = get_settings()
    
    # Determine formatter based on environment
    if settings.is_production():
        formatter = JSONFormatter()
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
    else:
        formatter = ConsoleFormatter()
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.app.log_level))
    root_logger.addHandler(handler)
    
    # Configure specific loggers
    loggers = {
        "dfs_prophet": {
            "level": settings.app.log_level,
            "handlers": ["console"],
            "propagate": False
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        },
        "fastapi": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        }
    }
    
    # Apply logger configurations
    for logger_name, config in loggers.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, config["level"]))
        logger.addHandler(handler)
        logger.propagate = config["propagate"]


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


def get_performance_logger(name: str) -> PerformanceLogger:
    """Get a performance logger instance."""
    return PerformanceLogger(get_logger(name))


def get_error_tracker(name: str) -> ErrorTracker:
    """Get an error tracker instance."""
    return ErrorTracker(get_logger(name))


# Initialize logging on module import
setup_logging()

# Create default loggers
logger = get_logger("dfs_prophet")
performance_logger = get_performance_logger("dfs_prophet")
error_tracker = get_error_tracker("dfs_prophet")


# Convenience functions for common logging patterns
def log_function_entry(func_name: str, **kwargs) -> None:
    """Log function entry with parameters."""
    logger.debug(f"Entering {func_name}", extra={"function_params": kwargs})


def log_function_exit(func_name: str, result: Any = None) -> None:
    """Log function exit with result."""
    logger.debug(f"Exiting {func_name}", extra={"function_result": result})


def log_api_request(method: str, path: str, status_code: int, duration: float) -> None:
    """Log API request details."""
    logger.info(
        f"API Request: {method} {path} - {status_code}",
        extra={
            "api_request": {
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration * 1000
            }
        }
    )


def log_database_operation(operation: str, table: str, duration: float, success: bool) -> None:
    """Log database operation details."""
    level = "INFO" if success else "ERROR"
    logger.log(
        getattr(logging, level),
        f"Database {operation} on {table}",
        extra={
            "database_operation": {
                "operation": operation,
                "table": table,
                "duration_ms": duration * 1000,
                "success": success
            }
        }
    )


def log_vector_operation(operation: str, collection: str, vector_count: int, duration: float) -> None:
    """Log vector database operation details."""
    logger.info(
        f"Vector {operation} on {collection}",
        extra={
            "vector_operation": {
                "operation": operation,
                "collection": collection,
                "vector_count": vector_count,
                "duration_ms": duration * 1000
            }
        }
    )


