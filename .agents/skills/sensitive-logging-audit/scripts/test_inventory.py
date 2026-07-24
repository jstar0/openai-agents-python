from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from _inventory import (
    collect_source_files,
    compare_findings,
    inventory_source,
    summarize,
    validate_review_ledger,
)


class InventoryTests(unittest.TestCase):
    def test_inventories_static_dynamic_and_exception_logger_calls(self) -> None:
        findings = inventory_source(
            """
from ..logger import logger

logger.debug("ready")
logger.warning(f"Request {request_id} failed")
logger.error("Request failed: %s", error)
logger.exception("Unhandled failure")
""",
            "src/agents/run_internal/fixture.py",
        )

        self.assertEqual(
            [(item.method, item.shape, item.confidence) for item in findings],
            [
                ("debug", "static-message", "confirmed"),
                ("warning", "dynamic-message", "confirmed"),
                ("error", "payload", "confirmed"),
                ("exception", "exception-payload", "confirmed"),
            ],
        )

    def test_inventories_keyword_messages_and_helper_payloads(self) -> None:
        findings = inventory_source(
            """
from agents.logger import logger, log_tool_action_error

logger.error(msg=secret)
logger.info(msg="ready")
log_tool_action_error(target_logger=logger, message=secret, exc=error)
"""
        )

        self.assertEqual(
            [(item.kind, item.shape) for item in findings],
            [
                ("logger", "dynamic-message"),
                ("logger", "static-message"),
                ("sensitive-helper", "payload"),
            ],
        )
        self.assertEqual(summarize(findings)["dynamic"], 2)

    def test_recognizes_logger_factories_aliases_methods_and_partial(self) -> None:
        findings = inventory_source(
            """
import functools
import logging

base = logging.getLogger(__name__)
alias = base
emit = alias.error
emit("failed", secret)
handler = functools.partial(alias.warning, "failed: %s")
handler(secret)
schedule(alias.info)
"""
        )

        self.assertEqual(
            [(item.kind, item.method) for item in findings],
            [
                ("logger", "error"),
                ("logger", "warning"),
                ("logger-callback", "info"),
            ],
        )

    def test_imported_partial_is_resolved(self) -> None:
        findings = inventory_source(
            """
from functools import partial
from agents.logger import logger

report = partial(logger.error, "failed: %s")
report(secret)
"""
        )

        self.assertEqual([(item.kind, item.method) for item in findings], [("logger", "error")])

    def test_inventories_raw_output_and_unknown_receivers(self) -> None:
        findings = inventory_source(
            """
import sys
import traceback
import warnings

print(secret)
warnings.warn(secret)
sys.stderr.write(secret)
traceback.print_exception(error)
task.exception()
"""
        )

        self.assertEqual(
            [(item.kind, item.method, item.confidence) for item in findings],
            [
                ("raw-output", "print", "confirmed"),
                ("raw-output", "warnings.warn", "confirmed"),
                ("raw-output", "stderr.write", "confirmed"),
                ("raw-output", "traceback.print_exception", "confirmed"),
                ("logging-candidate", "exception", "unknown"),
            ],
        )

    def test_inventories_directly_imported_output_streams(self) -> None:
        findings = inventory_source(
            """
from sys import stderr, stdout as out

stderr.write(secret)
out.write(secret)
"""
        )

        self.assertEqual(
            [(item.kind, item.method, item.shape) for item in findings],
            [
                ("raw-output", "stderr.write", "dynamic-message"),
                ("raw-output", "stdout.write", "dynamic-message"),
            ],
        )

    def test_requires_exact_policy_provenance_and_correct_polarity(self) -> None:
        findings = inventory_source(
            """
from agents import _debug
from agents.logger import logger

if not _debug.DONT_LOG_TOOL_DATA:
    logger.error("diagnostic: %s", secret)
if _debug.DONT_LOG_TOOL_DATA:
    logger.error("leak: %s", secret)
if not request.DONT_LOG_TOOL_DATA:
    logger.error("unrelated: %s", secret)
"""
        )

        self.assertEqual([item.policy for item in findings], ["tool-guard", "none", "none"])

    def test_scopes_policy_facts_to_lexical_bindings(self) -> None:
        findings = inventory_source(
            """
from agents import _debug
from agents.logger import logger

def configure():
    flag = _debug.DONT_LOG_TOOL_DATA
    return flag

def report(flag, secret):
    if not flag:
        logger.error("unguarded: %s", secret)
"""
        )

        self.assertEqual(findings[0].policy, "none")

    def test_preserves_direct_policy_aliases_but_not_composite_boolean_aliases(self) -> None:
        findings = inventory_source(
            """
from agents import _debug
from agents.logger import logger

def direct(secret):
    flag = _debug.DONT_LOG_TOOL_DATA
    if not flag:
        logger.error("guarded: %s", secret)

def composite(enabled, secret):
    guard = _debug.DONT_LOG_TOOL_DATA and enabled
    if not guard:
        logger.error("unguarded: %s", secret)
"""
        )

        self.assertEqual([item.policy for item in findings], ["tool-guard", "none"])

    def test_policy_assignments_do_not_flow_backward_or_through_conditional_rebinding(
        self,
    ) -> None:
        findings = inventory_source(
            """
from agents import _debug
from agents.logger import logger

def assigned_later(flag, secret):
    if not flag:
        logger.error("unguarded before assignment: %s", secret)
    flag = _debug.DONT_LOG_TOOL_DATA

def conditionally_reassigned(flag, condition, secret):
    if condition:
        flag = _debug.DONT_LOG_TOOL_DATA
    if not flag:
        logger.error("unguarded after conditional assignment: %s", secret)
"""
        )

        self.assertEqual([item.policy for item in findings], ["none", "none"])

    def test_combines_exact_model_and_tool_guards(self) -> None:
        findings = inventory_source(
            """
from agents import _debug
from agents.logger import logger

if not _debug.DONT_LOG_MODEL_DATA and not _debug.DONT_LOG_TOOL_DATA:
    logger.error("diagnostic: %s", secret)
"""
        )

        self.assertEqual(findings[0].policy, "model+tool-guard")

    def test_recognizes_early_return_policy_guards(self) -> None:
        findings = inventory_source(
            """
from agents import _debug
from agents.logger import logger

def report(secret):
    if _debug.DONT_LOG_TOOL_DATA:
        logger.debug("Tool failed")
        return
    logger.error("Tool failed: %s", secret)
"""
        )

        self.assertEqual([item.policy for item in findings], ["none", "tool-guard"])

    def test_recognizes_early_continue_policy_guards(self) -> None:
        findings = inventory_source(
            """
from agents import _debug

for item in items:
    if _debug.DONT_LOG_MODEL_DATA or _debug.DONT_LOG_TOOL_DATA:
        print("Data is redacted")
        continue
    print(item)
""",
            "src/agents/fixture.py",
        )

        self.assertEqual(
            [(item.shape, item.policy) for item in findings],
            [("static-message", "none"), ("dynamic-message", "model+tool-guard")],
        )

    def test_trusts_only_the_exact_known_helper_import(self) -> None:
        trusted = inventory_source(
            """
from agents.run_internal.tool_execution import log_tool_action_error as report_failure

report_failure("Tool failed", error)
"""
        )
        untrusted = inventory_source(
            """
from analytics.logger import log_tool_action_error

log_tool_action_error("Tool failed", error)
"""
        )

        self.assertEqual(
            (trusted[0].kind, trusted[0].policy, trusted[0].confidence),
            ("sensitive-helper", "tool-helper", "confirmed"),
        )
        self.assertEqual(
            (untrusted[0].kind, untrusted[0].policy, untrusted[0].confidence),
            ("helper-candidate", "none", "unknown"),
        )

    def test_recognizes_the_known_helper_in_its_defining_module(self) -> None:
        findings = inventory_source(
            """
def log_tool_action_error(message, error):
    pass

log_tool_action_error("Tool failed", error)
""",
            "src/agents/run_internal/tool_execution.py",
        )

        self.assertEqual(
            (findings[0].kind, findings[0].policy, findings[0].confidence),
            ("sensitive-helper", "tool-helper", "confirmed"),
        )

    def test_recognizes_shared_model_and_mixed_helpers(self) -> None:
        findings = inventory_source(
            """
from agents.logger import (
    log_model_action_debug,
    log_model_action_error,
    log_model_and_tool_action_error,
    log_model_and_tool_action_warning,
)

log_model_action_debug(logger, "Model cleanup failed", error)
log_model_action_error(logger, "Model failed", error)
log_model_and_tool_action_error(logger, "Trace failed", error)
log_model_and_tool_action_warning(logger, "Session failed", error)
"""
        )

        self.assertEqual(
            [(item.method, item.policy) for item in findings],
            [
                ("log_model_action_debug", "model-helper"),
                ("log_model_action_error", "model-helper"),
                ("log_model_and_tool_action_error", "model+tool-helper"),
                ("log_model_and_tool_action_warning", "model+tool-helper"),
            ],
        )

    def test_recognizes_shared_tool_helpers_at_multiple_levels(self) -> None:
        findings = inventory_source(
            """
from agents.logger import log_tool_action_debug, log_tool_action_warning

log_tool_action_debug(logger, "Cleanup cancelled", error)
log_tool_action_warning(logger, "Cleanup failed", error)
"""
        )

        self.assertEqual(
            [(item.method, item.policy) for item in findings],
            [
                ("log_tool_action_debug", "tool-helper"),
                ("log_tool_action_warning", "tool-helper"),
            ],
        )

    def test_flags_caught_values_and_implicit_exception_payloads(self) -> None:
        findings = inventory_source(
            """
from agents.logger import logger

try:
    run()
except Exception as exc:
    logger.error("Run failed: %s", exc)
    logger.error("Run failed", exc_info=True)
    logger.exception("Run failed")
"""
        )

        self.assertEqual(
            [(item.shape, item.catch_value) for item in findings],
            [
                ("payload", "exc"),
                ("exception-payload", "active exception"),
                ("exception-payload", "active exception"),
            ],
        )

    def test_fingerprints_survive_line_shifts_and_distinguish_branches(self) -> None:
        before = inventory_source(
            """
from agents.logger import logger
def report(secret):
    if enabled:
        logger.error("failed: %s", secret)
    else:
        logger.error("failed: %s", secret)
"""
        )
        after = inventory_source(
            """
from agents.logger import logger


def report(secret):
    unused = None
    if enabled:
        logger.error("failed: %s", secret)
    else:
        logger.error("failed: %s", secret)
"""
        )

        self.assertEqual(
            [item.group_fingerprint for item in before],
            [item.group_fingerprint for item in after],
        )
        self.assertNotEqual(before[0].group_fingerprint, before[1].group_fingerprint)

    def test_identical_sites_form_a_duplicate_group(self) -> None:
        findings = inventory_source(
            """
from agents.logger import logger
def report(secret):
    logger.error("failed: %s", secret)
    logger.error("failed: %s", secret)
"""
        )

        self.assertEqual(findings[0].group_fingerprint, findings[1].group_fingerprint)
        self.assertEqual([item.group_count for item in findings], [2, 2])
        self.assertEqual([item.identity_quality for item in findings], ["duplicate", "duplicate"])
        self.assertEqual([item.fingerprint.rsplit(":", 1)[-1] for item in findings], ["0", "1"])

    def test_discovers_keyword_callback_sinks(self) -> None:
        findings = inventory_source(
            """
import sys
from agents.logger import logger

register(on_error=logger.error, writer=sys.stderr.write)
"""
        )

        self.assertEqual(
            [(item.kind, item.method) for item in findings],
            [("logger-callback", "error"), ("raw-output-callback", "stderr.write")],
        )

    def test_source_collection_ignores_hidden_paths_only_below_the_scan_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / ".cache" / "worktree"
            visible = root / "src" / "visible.py"
            hidden = root / ".generated" / "hidden.py"
            visible.parent.mkdir(parents=True)
            hidden.parent.mkdir()
            visible.touch()
            hidden.touch()

            self.assertEqual(collect_source_files([root]), [visible.resolve()])

    def test_normalizes_path_separators_before_hashing(self) -> None:
        source = "from agents.logger import logger\nlogger.error('failed', secret)\n"
        posix = inventory_source(source, "src/agents/fixture.py")
        windows = inventory_source(source, "src\\agents\\fixture.py")
        self.assertEqual(posix[0].group_fingerprint, windows[0].group_fingerprint)

    def test_compare_reports_new_removed_and_duplicate_count_changes(self) -> None:
        baseline = inventory_source("from agents.logger import logger\nlogger.error('x', secret)\n")
        current = inventory_source(
            """
from agents.logger import logger
logger.error('x', secret)
logger.error('x', secret)
logger.warning('y', secret)
"""
        )
        comparison = compare_findings([item.to_dict() for item in baseline], current)

        self.assertEqual(len(comparison["new"]), 1)
        self.assertEqual(comparison["removed"], [])
        self.assertEqual(len(comparison["countChanged"]), 1)
        self.assertEqual(comparison["countChanged"][0]["before"], 1)
        self.assertEqual(comparison["countChanged"][0]["after"], 2)

    def test_review_ledger_requires_every_dynamic_group_and_exact_count(self) -> None:
        findings = inventory_source(
            """
from agents.logger import logger
logger.error('x', secret)
logger.error('x', secret)
logger.info('ready')
"""
        )
        dynamic_group = findings[0].group_fingerprint
        valid = validate_review_ledger(
            {
                "reviews": [
                    {
                        "group_fingerprint": dynamic_group,
                        "group_count": 2,
                        "disposition": "tool",
                        "evidence": "The argument is the tool input.",
                        "action": "Guard with DONT_LOG_TOOL_DATA.",
                    }
                ]
            },
            findings,
        )
        invalid = validate_review_ledger({"reviews": []}, findings)

        self.assertTrue(valid["valid"])
        self.assertFalse(invalid["valid"])
        self.assertIn("Missing review", invalid["errors"][0])

    def test_summary_separates_unknown_and_raw_candidates(self) -> None:
        findings = inventory_source(
            """
from agents.logger import logger
logger.info('ready')
logger.error('failed', secret)
print(secret)
task.exception()
"""
        )
        summary = summarize(findings)

        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["dynamic"], 3)
        self.assertEqual(summary["rawOutputCalls"], 1)
        self.assertEqual(summary["unknownReceiverCalls"], 1)


if __name__ == "__main__":
    unittest.main()
