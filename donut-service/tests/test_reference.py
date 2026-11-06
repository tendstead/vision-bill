import io
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient
from PIL import Image

import server
from decoding import clean_sequence
from reference import build_multipart, read_response


class DecodingTests(unittest.TestCase):
    def test_preserves_receipt_structure_and_german_characters(self):
        sequence = "<s_cord-v2><s_menu><s_nm>Brötchen</s_nm></s_menu></s><pad>"
        self.assertEqual(
            clean_sequence(sequence, "<s_cord-v2>", "</s>", "<pad>"),
            "<s_menu><s_nm>Brötchen</s_nm></s_menu>",
        )

    def test_does_not_remove_first_field_when_task_prefix_is_absent(self):
        self.assertEqual(
            clean_sequence("<s_total>12.90</s_total></s>", "<s_cord-v2>", "</s>", "<pad>"),
            "<s_total>12.90</s_total>",
        )


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = Mock()
        self.engine.extract.return_value = {
            "prediction": {"menu": [{"nm": "Mate", "price": "12.90"}]},
            "raw_sequence": "<s_menu>Mate</s_menu>",
            "generated_tokens": 10,
            "truncated": False,
        }
        server.app.state.engine = self.engine
        self.client = TestClient(server.app)
        buf = io.BytesIO()
        Image.new("RGB", (12, 20), "white").save(buf, format="PNG")
        self.image = buf.getvalue()

    def tearDown(self):
        server.app.state.engine = None

    def test_returns_recognized_content_without_pixels(self):
        response = self.client.post(
            "/ocr", files={"image": ("receipt.png", self.image, "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["prediction"]["menu"][0]["nm"], "Mate")
        self.assertEqual(body["image_size"], [12, 20])
        self.assertNotIn("input_img", body)
        self.assertLess(len(response.content), 2000)

    def test_rejects_corrupt_image_before_inference(self):
        response = self.client.post(
            "/ocr", files={"image": ("bad.png", b"not an image", "image/png")}
        )
        self.assertEqual(response.status_code, 422)
        self.engine.extract.assert_not_called()

    def test_reports_unready_service(self):
        server.app.state.engine = None
        self.assertEqual(self.client.get("/health").status_code, 503)
        response = self.client.post(
            "/ocr", files={"image": ("receipt.png", self.image, "image/png")}
        )
        self.assertEqual(response.status_code, 503)

    def test_rejects_concurrent_inference_without_queueing(self):
        with server.inference_lock:
            response = self.client.post(
                "/ocr", files={"image": ("receipt.png", self.image, "image/png")}
            )
        self.assertEqual(response.status_code, 429)
        self.engine.extract.assert_not_called()

    def test_recovers_lock_after_inference_error(self):
        self.engine.extract.side_effect = RuntimeError("failed")
        with self.assertLogs("server", level="ERROR"):
            response = self.client.post(
                "/ocr", files={"image": ("receipt.png", self.image, "image/png")}
            )
        self.assertEqual(response.status_code, 500)
        self.engine.extract.side_effect = None
        response = self.client.post(
            "/ocr", files={"image": ("receipt.png", self.image, "image/png")}
        )
        self.assertEqual(response.status_code, 200)


class ClientTests(unittest.TestCase):
    def test_multipart_keeps_binary_image_intact(self):
        data = b"\x00\xffreceipt\r\n"
        body, content_type = build_multipart("receipt.png", data, "fixed-boundary")
        self.assertIn(b'name="image"', body)
        self.assertIn(b"Content-Type: image/png", body)
        self.assertIn(data, body)
        self.assertEqual(content_type, "multipart/form-data; boundary=fixed-boundary")

    def test_rejects_large_response_instead_of_loading_pixel_json(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            read_response(io.BytesIO(b"x" * 11), limit=10)

    def test_decodes_json_response(self):
        self.assertEqual(read_response(io.BytesIO(b'{"text":"Mate"}')), {"text": "Mate"})


if __name__ == "__main__":
    unittest.main()
