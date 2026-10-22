"""reserv — OpenVINO encoder inference service."""
from .server import VERSION, create_app

__all__ = ["VERSION", "create_app"]
