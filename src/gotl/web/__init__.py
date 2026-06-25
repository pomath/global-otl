"""Web dashboard for the global OTL pipeline."""

from __future__ import annotations


def run_server(
    host: str = "0.0.0.0",
    port: int = 8050,
    reload: bool = False,
) -> None:
    """Start the uvicorn ASGI server."""
    import uvicorn

    uvicorn.run(
        "gotl.web.app:app",
        host=host,
        port=port,
        reload=reload,
    )
