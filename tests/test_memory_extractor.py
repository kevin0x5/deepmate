import unittest

from deepmate.memory.extractor import (
    EXTRACTION_SYSTEM_PROMPT,
    should_skip_memory_extraction,
)


class MemoryExtractorPromptTests(unittest.TestCase):
    def test_product_context_is_not_profile_memory(self) -> None:
        prompt = EXTRACTION_SYSTEM_PROMPT

        self.assertIn("memory_facts: global durable preferences", prompt)
        self.assertIn("product-specific", prompt)
        self.assertIn("Put product/task/project context in session_only", prompt)
        self.assertNotIn("project facts, decisions, or notes", prompt)

    def test_order_number_does_not_skip_memory_extraction(self) -> None:
        decision = should_skip_memory_extraction(
            "我刚才提到的订单号是 1234567890123456，但长期偏好是回答要直接。"
        )

        self.assertFalse(decision.should_skip)

    def test_sensitive_number_context_skips_memory_extraction(self) -> None:
        decision = should_skip_memory_extraction("我的银行卡号是 4111 1111 1111 1111。")

        self.assertTrue(decision.should_skip)
        self.assertEqual(decision.reason, "sensitive")

    def test_inline_fenced_command_does_not_skip_memory_extraction(self) -> None:
        decision = should_skip_memory_extraction(
            "以后遇到目录排查时，请先运行 ```ls -la``` 再解释。"
        )

        self.assertFalse(decision.should_skip)

    def test_standalone_fenced_block_still_skips_memory_extraction(self) -> None:
        decision = should_skip_memory_extraction("```python\nprint('hello')\n```")

        self.assertTrue(decision.should_skip)
        self.assertEqual(decision.reason, "code_or_log")


if __name__ == "__main__":
    unittest.main()
