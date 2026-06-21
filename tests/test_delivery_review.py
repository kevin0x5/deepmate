from __future__ import annotations

import unittest

from deepmate.runtime import (
    DeliveryReviewInput,
    DeliveryReviewStatus,
    build_delivery_review_input,
    review_final_response,
    should_run_llm_delivery_review,
)
from deepmate.providers import ModelToolExchange, ModelToolRequest, ModelToolResult


class DeliveryReviewTest(unittest.TestCase):
    def test_accepts_simple_complete_response(self) -> None:
        review = review_final_response(
            DeliveryReviewInput(
                user_request="Explain the current status.",
                final_response_draft="The current status is ready.",
            )
        )

        self.assertEqual(review.status, DeliveryReviewStatus.ACCEPTED)
        self.assertTrue(review.is_accepted())

    def test_blocks_empty_final_response(self) -> None:
        review = review_final_response(
            DeliveryReviewInput(
                user_request="Explain the current status.",
                final_response_draft=" ",
            )
        )

        self.assertEqual(review.status, DeliveryReviewStatus.BLOCKED)
        self.assertIn("final_response_empty", review.issues)

    def test_requires_artifacts_and_validation_for_workspace_write(self) -> None:
        review = review_final_response(
            DeliveryReviewInput(
                user_request="Update the code.",
                final_response_draft="I updated the code.",
                workspace_write_used=True,
            )
        )

        self.assertEqual(review.status, DeliveryReviewStatus.NEEDS_REVISION)
        self.assertIn("workspace_write_without_artifact_refs", review.issues)
        self.assertIn("workspace_write_without_validation_summary", review.issues)

    def test_requires_known_limits_to_be_disclosed(self) -> None:
        review = review_final_response(
            DeliveryReviewInput(
                user_request="Update the code.",
                final_response_draft="I updated the code.",
                known_limits=("tests not run",),
            )
        )

        self.assertEqual(review.status, DeliveryReviewStatus.NEEDS_REVISION)
        self.assertIn("known_limits_not_disclosed", review.issues)

    def test_llm_review_is_only_suggested_for_riskier_delivery(self) -> None:
        self.assertFalse(
            should_run_llm_delivery_review(
                DeliveryReviewInput(
                    user_request="Explain.",
                    final_response_draft="Done.",
                )
            )
        )
        self.assertTrue(
            should_run_llm_delivery_review(
                DeliveryReviewInput(
                    user_request="Update.",
                    final_response_draft="Done.",
                    workspace_write_used=True,
                )
            )
        )
        self.assertTrue(
            should_run_llm_delivery_review(
                DeliveryReviewInput(
                    user_request="Merge child outputs.",
                    final_response_draft="Done.",
                    accepted_subagent_reviews=("one", "two"),
                )
            )
        )
        self.assertTrue(
            should_run_llm_delivery_review(
                DeliveryReviewInput(
                    user_request="Merge child outputs.",
                    final_response_draft="Done.",
                    non_accepted_subagent_reviews=("missing evidence",),
                )
            )
        )

    def test_builds_packet_from_subagent_tool_exchange(self) -> None:
        exchange = ModelToolExchange(
            tool_requests=(
                ModelToolRequest(name="run_subagent", id="call_1"),
            ),
            tool_results=(
                ModelToolResult(
                    name="run_subagent",
                    request_id="call_1",
                    content=(
                        "{\"run_id\":\"run_1\",\"status\":\"completed\","
                        "\"summary\":\"done\","
                        "\"artifact_refs\":[\"result.txt\"],"
                        "\"review\":{\"status\":\"accepted\",\"summary\":\"ok\","
                        "\"missing\":[],\"retryable\":false}}"
                    ),
                    refs=("run_1",),
                ),
            ),
        )

        review_input = build_delivery_review_input(
            user_request="Inspect.",
            final_response_draft="Done.",
            tool_exchanges=(exchange,),
            errors=(),
            reached_max_steps=False,
        )

        self.assertIn("ok", review_input.accepted_subagent_reviews)
        self.assertEqual(review_input.artifact_refs, ("result.txt",))
        self.assertTrue(review_input.workspace_write_used)

    def test_build_packet_tolerates_none_refs(self) -> None:
        exchange = ModelToolExchange(
            tool_requests=(
                ModelToolRequest(name="write_text_file", id="call_1"),
            ),
            tool_results=(
                ModelToolResult(
                    name="write_text_file",
                    request_id="call_1",
                    content="ok",
                    refs=None,
                ),
            ),
        )

        review_input = build_delivery_review_input(
            user_request="Write.",
            final_response_draft="Done.",
            tool_exchanges=(exchange,),
            errors=(),
            reached_max_steps=False,
        )

        self.assertEqual(review_input.evidence_refs, ())
        self.assertEqual(review_input.artifact_refs, ())
        self.assertTrue(review_input.workspace_write_used)


if __name__ == "__main__":
    unittest.main()
