"""NASA Earthdata authentication for CDDIS access."""

from __future__ import annotations

import logging
import netrc
import os
from pathlib import Path

log = logging.getLogger(__name__)

_EARTHDATA_HOST = "urs.earthdata.nasa.gov"


def get_earthdata_session():
    """Create a requests.Session with NASA Earthdata authentication.

    Auth methods (tried in order):
        1. EARTHDATA_TOKEN env var (bearer token)
        2. ~/.netrc entry for urs.earthdata.nasa.gov
        3. EARTHDATA_USER + EARTHDATA_PASSWORD env vars

    Returns:
        requests.Session configured for CDDIS access.

    Raises:
        RuntimeError: If no credentials are found.
    """
    import requests

    session = requests.Session()

    # Method 1: Bearer token
    token = os.environ.get("EARTHDATA_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
        log.info("Using Earthdata bearer token from EARTHDATA_TOKEN")
        return session

    # Method 2: .netrc
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            nrc = netrc.netrc(str(netrc_path))
            auth = nrc.authenticators(_EARTHDATA_HOST)
            if auth:
                session.auth = (auth[0], auth[2])
                log.info("Using Earthdata credentials from ~/.netrc")
                return session
        except netrc.NetrcParseError:
            log.debug("Failed to parse ~/.netrc")

    # Method 3: Environment variables
    user = os.environ.get("EARTHDATA_USER")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if user and password:
        session.auth = (user, password)
        log.info("Using Earthdata credentials from EARTHDATA_USER/PASSWORD")
        return session

    raise RuntimeError(
        "No NASA Earthdata credentials found. Set one of:\n"
        "  1. EARTHDATA_TOKEN env var\n"
        "  2. ~/.netrc with machine urs.earthdata.nasa.gov\n"
        "  3. EARTHDATA_USER + EARTHDATA_PASSWORD env vars\n"
        "Register at https://urs.earthdata.nasa.gov/users/new"
    )
