"""Focused tests for Chat response rendering."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import chat  # noqa: E402


class ChatRenderingTests(unittest.TestCase):
    def test_plain_response(self):
        self.assertEqual(chat._visible_reply("Hello **there**"), "Hello **there**")

    def test_reasoning_is_replaced_while_streaming(self):
        raw = "<think>\nWorking through the request"
        self.assertEqual(chat._visible_reply(raw), "Thinking...")

    def test_final_answer_is_displayed_without_reasoning(self):
        raw = "<think>private reasoning</think>\n\nVisible answer"
        self.assertEqual(chat._visible_reply(raw), "Visible answer")
        self.assertEqual(chat._visible_reply(raw, finished=True), "Visible answer")

    def test_unfinished_reasoning_has_visible_fallback(self):
        raw = "<think>reasoning that reached the token limit"
        result = chat._visible_reply(raw, finished=True)
        self.assertIn("output limit", result)
        self.assertNotIn("<think>", result)

    def test_completed_reasoning_without_answer_has_visible_fallback(self):
        raw = "<think>private reasoning</think>"
        result = chat._visible_reply(raw, finished=True)
        self.assertIn("did not produce a final answer", result)


if __name__ == "__main__":
    unittest.main()
