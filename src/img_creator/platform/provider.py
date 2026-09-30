from io import BytesIO
import ipaddress
import socket
import time
from urllib.parse import urlparse
import httpx
from PIL import Image
from .catalog import STYLES


class ProviderRejected(RuntimeError):
    """Known terminal rejection. Credit reservation may be refunded."""


class ProviderUncertain(RuntimeError):
    """Submission may have incurred cost: hold for reconciliation, never auto-resubmit."""


def safe_url(url, polling=False):
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in {None, 443}:
        raise ValueError("Invalid provider URL")
    if polling and not (host == "api.bfl.ai" or host.endswith(".bfl.ai")):
        raise ValueError("Unexpected polling host")
    for address in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
        if not ipaddress.ip_address(address[4][0]).is_global:
            raise ValueError("Private provider URL rejected")
    return url


class BFLProvider:
    def __init__(self, key):
        self.key = key

    def generate(self, request, heartbeat):
        native, _ = request.dimensions()
        payload = {
            "prompt": (request.prompt + " " + STYLES[request.style]).strip(),
            "width": native[0],
            "height": native[1],
            "seed": request.seed,
            "output_format": "png",
        }
        if request.model == "flux-2-flex":
            payload["steps"] = {"low": 20, "medium": 35, "high": 50}[request.effort]
            payload["guidance"] = 4.5
        # Provider moderation defaults are retained. No user override or fallback bypass.
        headers = {"x-key": self.key}
        with httpx.Client(timeout=45, follow_redirects=False) as client:
            try:
                response = client.post("https://api.bfl.ai/v1/" + request.model, headers=headers, json=payload)
                if 400 <= response.status_code < 500:
                    raise ProviderRejected("Provider rejected the request. Credits have been returned.")
                response.raise_for_status()
                task = response.json()
                poll_url = safe_url(task["polling_url"], polling=True)
                heartbeat({"id": task["id"], "polling_url": poll_url})
            except ProviderRejected:
                raise
            except Exception as exc:
                raise ProviderUncertain("Provider submission could not be confirmed. Owner review required.") from exc
            for _ in range(240):
                heartbeat(None)
                time.sleep(2)
                try:
                    result = client.get(poll_url, headers=headers)
                    result.raise_for_status()
                    result = result.json()
                except httpx.HTTPError:
                    continue
                status = result.get("status")
                if status in {"Error", "Failed", "Request Moderated", "Content Moderated"}:
                    raise ProviderRejected("Provider could not complete this image. Credits have been returned.")
                if status == "Ready":
                    url = safe_url(result["result"]["sample"])
                    buffer = BytesIO()
                    # Never forward the API key to the image host, or follow arbitrary redirects.
                    with client.stream("GET", url) as image_response:
                        image_response.raise_for_status()
                        for chunk in image_response.iter_bytes():
                            buffer.write(chunk)
                            if buffer.tell() > 50 * 1024 * 1024:
                                raise ValueError("Provider image too large")
                    buffer.seek(0)
                    with Image.open(buffer) as image:
                        if image.width * image.height > 16_000_000:
                            raise ValueError("Provider image exceeds pixel limit")
                        output = image.convert("RGB")
                    return output, {
                        "provider": "bfl",
                        "provider_job_id": task["id"],
                        "tokens": None,
                        "provider_cost": None,
                        "native_size": list(output.size),
                        "steps": payload.get("steps"),
                        "effective_prompt": payload["prompt"],
                    }
            raise ProviderUncertain("Provider has not completed the request. Owner review required.")
