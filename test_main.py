from fastapi.testclient import TestClient
from requests_toolbelt.multipart.encoder import MultipartEncoder
from main import app

client = TestClient(app)

def test_upload():
    filename = '4.jpg'

    with open(filename, 'rb') as file:
        data = MultipartEncoder(
            fields={'file': (filename, file, 'image/jpeg')}
        )

        response = client.post('/predict/', data="test", headers={'Content-Type': 'multipart/form-data'})

        assert response.status_code == 200
        assert response.json() == {'prediction': 4}