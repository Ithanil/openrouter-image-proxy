import base64
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import openrouter_media_proxy


class ProxyAudioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(openrouter_media_proxy.app)

    def tearDown(self) -> None:
        self.client.close()

    def test_transcriptions_translate_multipart_into_input_audio(self) -> None:
        captured: dict = {}

        async def fake_call_upstream(client, body, headers, rid, idx):
            captured["body"] = body
            captured["headers"] = headers
            return (
                {
                    "choices": [{"message": {"content": "Guten Tag"}}],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 3,
                        "total_tokens": 15,
                        "prompt_tokens_details": {
                            "audio_tokens": 10,
                            "text_tokens": 2,
                        },
                    },
                },
                None,
            )

        with patch.object(openrouter_media_proxy, "_call_upstream", new=fake_call_upstream):
            response = self.client.post(
                "/v1/audio/transcriptions",
                headers={"Authorization": "Bearer test-key"},
                data={
                    "model": "google/gemini-2.5-flash",
                    "language": "de",
                    "prompt": "Prefer medical terminology.",
                },
                files={"file": ("clip.wav", b"RIFFDATA", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "text": "Guten Tag",
                "usage": {
                    "input_token_details": {
                        "audio_tokens": 10,
                        "text_tokens": 2,
                    },
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "total_tokens": 15,
                    "type": "tokens",
                },
            },
        )

        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(captured["body"]["model"], "google/gemini-2.5-flash")
        message_parts = captured["body"]["messages"][0]["content"]
        self.assertEqual(message_parts[1]["type"], "input_audio")
        self.assertEqual(message_parts[1]["input_audio"]["format"], "wav")
        self.assertEqual(
            message_parts[1]["input_audio"]["data"],
            base64.b64encode(b"RIFFDATA").decode(),
        )
        self.assertIn("medical terminology", message_parts[0]["text"])
        self.assertIn("de", message_parts[0]["text"])

    def test_translations_verbose_json_is_normalized(self) -> None:
        captured: dict = {}

        async def fake_call_upstream(client, body, headers, rid, idx):
            captured["body"] = body
            return (
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "duration": 1.25,
                                        "segments": [
                                            {
                                                "avg_logprob": -0.1,
                                                "compression_ratio": 1.0,
                                                "end": 1.25,
                                                "id": 0,
                                                "no_speech_prob": 0.0,
                                                "seek": 0,
                                                "start": 0.0,
                                                "temperature": 0.0,
                                                "text": "Hello world",
                                                "tokens": [],
                                            }
                                        ],
                                        "text": "Hello world",
                                    }
                                )
                            }
                        }
                    ]
                },
                None,
            )

        with patch.object(openrouter_media_proxy, "_call_upstream", new=fake_call_upstream):
            response = self.client.post(
                "/v1/audio/translations",
                data={
                    "model": "google/gemini-2.5-flash",
                    "response_format": "verbose_json",
                },
                files={"file": ("clip.mp3", b"ID3DATA", "audio/mpeg")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "duration": 1.25,
                "language": "english",
                "segments": [
                    {
                        "avg_logprob": -0.1,
                        "compression_ratio": 1.0,
                        "end": 1.25,
                        "id": 0,
                        "no_speech_prob": 0.0,
                        "seek": 0,
                        "start": 0.0,
                        "temperature": 0.0,
                        "text": "Hello world",
                        "tokens": [],
                    }
                ],
                "text": "Hello world",
            },
        )

        message_parts = captured["body"]["messages"][0]["content"]
        self.assertEqual(message_parts[1]["input_audio"]["format"], "mp3")
        self.assertIn("Translate the provided audio into English", message_parts[0]["text"])
        self.assertIn("Return only valid JSON", message_parts[0]["text"])

    def test_speech_collects_streamed_audio_into_binary_response(self) -> None:
        captured: dict = {}

        async def fake_collect_speech_audio(body, headers, rid):
            captured["body"] = body
            captured["headers"] = headers
            return b"\x00\x01\x02", "Hello there", None

        with patch.object(
            openrouter_media_proxy,
            "_collect_speech_audio",
            new=fake_collect_speech_audio,
        ):
            response = self.client.post(
                "/v1/audio/speech",
                headers={"Authorization": "Bearer test-key"},
                json={
                    "input": "Hello there",
                    "instructions": "Warm and calm.",
                    "model": "openai/gpt-4o-audio-preview",
                    "response_format": "wav",
                    "speed": 1.25,
                    "voice": {"id": "voice_1234"},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"\x00\x01\x02")
        self.assertTrue(response.headers["content-type"].startswith("audio/wav"))
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(captured["body"]["audio"], {"voice": "voice_1234", "format": "wav"})
        self.assertEqual(captured["body"]["modalities"], ["text", "audio"])
        self.assertTrue(captured["body"]["stream"])
        self.assertIn("Warm and calm", captured["body"]["messages"][0]["content"])
        self.assertIn("1.25x", captured["body"]["messages"][0]["content"])
        self.assertEqual(captured["body"]["messages"][1]["content"], "Hello there")


class ProxyImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(openrouter_media_proxy.app)

    def tearDown(self) -> None:
        self.client.close()

    def test_generations_translate_request_and_collect_parallel_results(self) -> None:
        calls: list[dict] = []

        async def fake_call_upstream(client, body, headers, rid, idx):
            calls.append(
                {
                    "body": body,
                    "headers": headers,
                    "idx": idx,
                }
            )
            return (
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Revised sunset prompt",
                                "images": [
                                    {
                                        "image_url": {
                                            "url": "data:image/png;base64,QUJDRA=="
                                        }
                                    }
                                ],
                            }
                        }
                    ]
                },
                None,
            )

        with patch.object(openrouter_media_proxy, "_call_upstream", new=fake_call_upstream):
            response = self.client.post(
                "/v1/images/generations",
                headers={"Authorization": "Bearer test-key"},
                json={
                    "background": "transparent",
                    "model": "google/gemini-2.5-flash-image-preview",
                    "n": 2,
                    "prompt": "A mountain cabin at sunset",
                    "quality": "high",
                    "size": "1792x1024",
                    "style": "vivid",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("created", payload)
        self.assertEqual(
            payload["data"],
            [
                {"b64_json": "QUJDRA==", "revised_prompt": "Revised sunset prompt"},
                {"b64_json": "QUJDRA==", "revised_prompt": "Revised sunset prompt"},
            ],
        )

        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(call["headers"]["Authorization"], "Bearer test-key")
            self.assertEqual(
                call["body"]["model"],
                "google/gemini-2.5-flash-image-preview",
            )
            self.assertEqual(call["body"]["modalities"], openrouter_media_proxy._image_modalities())
            self.assertEqual(
                call["body"]["image_config"],
                {"aspect_ratio": "16:9", "image_size": "4K"},
            )
            self.assertIn("A mountain cabin at sunset", call["body"]["messages"][0]["content"])
            self.assertIn("vivid, dramatic style", call["body"]["messages"][0]["content"])
            self.assertIn(
                "transparent background",
                call["body"]["messages"][0]["content"],
            )
        self.assertEqual(sorted(call["idx"] for call in calls), [0, 1])

    def test_edits_json_translates_images_and_mask(self) -> None:
        captured: dict = {}

        async def fake_call_upstream(client, body, headers, rid, idx):
            captured["body"] = body
            captured["headers"] = headers
            return (
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Edited image",
                                "images": [
                                    {
                                        "image_url": {
                                            "url": "data:image/png;base64,RURJVA=="
                                        }
                                    }
                                ],
                            }
                        }
                    ]
                },
                None,
            )

        with patch.object(openrouter_media_proxy, "_call_upstream", new=fake_call_upstream):
            response = self.client.post(
                "/v1/images/edits",
                headers={"Authorization": "Bearer test-key"},
                json={
                    "background": "transparent",
                    "images": [
                        {"image_url": "data:image/png;base64,SU1BR0Ux"},
                    ],
                    "mask": {"image_url": "data:image/png;base64,TUFTSw=="},
                    "model": "google/gemini-2.5-flash-image-preview",
                    "prompt": "Add a rainbow",
                    "quality": "hd",
                    "size": "1024x1024",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("created", payload)
        self.assertEqual(
            payload["data"],
            [{"b64_json": "RURJVA==", "revised_prompt": "Edited image"}],
        )

        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(
            captured["body"]["image_config"],
            {"aspect_ratio": "1:1", "image_size": "2K"},
        )
        content_parts = captured["body"]["messages"][0]["content"]
        self.assertEqual(content_parts[0]["type"], "text")
        self.assertIn("Add a rainbow", content_parts[0]["text"])
        self.assertIn("transparent background", content_parts[0]["text"])
        self.assertEqual(
            content_parts[1:],
            [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,SU1BR0Ux"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,TUFTSw=="}},
            ],
        )

    def test_edits_multipart_translates_uploaded_files_into_data_urls(self) -> None:
        captured: dict = {}

        async def fake_call_upstream(client, body, headers, rid, idx):
            captured["body"] = body
            return (
                {
                    "choices": [
                        {
                            "message": {
                                "images": [
                                    {
                                        "image_url": {
                                            "url": "data:image/png;base64,TU9ESUZJRUQ="
                                        }
                                    }
                                ],
                            }
                        }
                    ]
                },
                None,
            )

        with patch.object(openrouter_media_proxy, "_call_upstream", new=fake_call_upstream):
            response = self.client.post(
                "/v1/images/edits",
                data={
                    "model": "google/gemini-2.5-flash-image-preview",
                    "prompt": "Replace the sky with stars",
                },
                files=[
                    ("image", ("source.png", b"\x89PNG", "image/png")),
                    ("mask", ("mask.png", b"MASK", "image/png")),
                ],
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["data"],
            [{"b64_json": "TU9ESUZJRUQ="}],
        )

        content_parts = captured["body"]["messages"][0]["content"]
        self.assertEqual(content_parts[0]["text"], "Replace the sky with stars")
        self.assertEqual(
            content_parts[1:],
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw==",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,TUFTSw==",
                    },
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
