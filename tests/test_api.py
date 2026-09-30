from fastapi.testclient import TestClient
from img_creator.api import create_app
from img_creator.config import Settings
from img_creator.generator import ImageGenerator
from test_generation import FakeBackend


def test_api_roundtrip_auth_and_upscale(tmp_path):
    service = ImageGenerator(Settings(output_dir=tmp_path, api_key="test-key"), backend=FakeBackend())
    client = TestClient(create_app(service))
    assert client.get("/health").status_code == 200
    assert client.post("/generate", json={"prompt": "Tree"}).status_code == 401
    headers = {"x-api-key": "test-key"}
    assert client.post("/generate", headers=headers, json={"prompt": " "}).status_code == 422
    response = client.post("/generate", headers=headers, json={"prompt": "Tree", "seed": 12})
    assert response.status_code == 200, response.text
    result = response.json()
    assert client.get(result["image_url"], headers=headers).status_code == 200
    assert client.get(result["metadata_url"], headers=headers).json()["seed"] == 12
    upscale = client.post(
        "/upscale",
        headers=headers,
        json={"generation_id": result["generation"]["generation_id"], "output_resolution": "2K", "upscaler": "lanczos"},
    )
    assert upscale.status_code == 200, upscale.text
    assert upscale.json()["generation"]["parent_generation_id"] == result["generation"]["generation_id"]
    assert client.get("/images/" + "z" * 32, headers=headers).status_code == 422
    assert client.get("/images/" + "0" * 32, headers=headers).status_code == 404


def test_error_does_not_expose_server_paths(tmp_path):
    backend = FakeBackend()

    def fail(*args):
        raise RuntimeError("/private/credentials token secret")

    backend.load = fail
    client = TestClient(create_app(ImageGenerator(Settings(output_dir=tmp_path), backend=backend)))
    r = client.post("/generate", json={"prompt": "Tree"})
    assert r.status_code == 503
    assert "private" not in r.text and "secret" not in r.text
