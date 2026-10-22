
#!/usr/bin/env python3
"""
Mythri Control-Flow-Focused CodeQL Audit
----------------------------------------
Runs CodeQL's Python extended-quality suite against an existing database,
then produces:
  1. Full SARIF findings
  2. A control-flow-focused Markdown report
  3. A JSON summary

This script does not modify Mythri application source code.

Note:
The control-flow report is a keyword-based triage of findings from the
selected suite. It is not a complete control-flow graph or proof of
program correctness.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HOME = Path.home()

DEFAULT_CODEQL = (
    HOME
    / ".config/Code/User/globalStorage/github.vscode-codeql"
    / "distribution1/codeql/codeql"
)

DEFAULT_DATABASE = HOME / "codeql-databases/mythri-python"

DEFAULT_SUITE = (
    HOME
    / ".codeql/packages/codeql/python-queries/1.8.12"
    / "codeql-suites/python-code-quality-extended.qls"
)

DEFAULT_OUTPUT = HOME / "codeql-databases/mythri-control-flow-reports"

# These keywords help triage relevant findings. They do not prove
# that a finding is a control-flow defect.
CONTROL_FLOW_KEYWORDS = (
    "unreachable",
    "empty except",
    "exception handling",
    "empty catch",
    "redundant condition",
    "condition is always",
    "always true",
    "always false",
    "dead code",
    "useless assignment",
    "return statement",
    "control flow",
    "branch",
    "loop",
    "guard condition",
    "fall through",
)


def run_command(command: list[str]) -> None:
    """Execute a command and stop if CodeQL reports an error."""
    print("\nRunning:")
    print(" ".join(str(part) for part in command))

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )


def load_sarif(path: Path) -> tuple[list[dict], dict[str, dict]]:
    """Load SARIF results and map rule IDs to their metadata."""
    with path.open("r", encoding="utf-8") as file:
        sarif = json.load(file)

    results = []
    rule_metadata = {}

    for run in sarif.get("runs", []):
        driver = run.get("tool", {}).get("driver", {})

        for rule in driver.get("rules", []):
            rule_id = rule.get("id", "")
            rule_metadata[rule_id] = rule

        results.extend(run.get("results", []))

    return results, rule_metadata


def get_rule_text(rule_id: str, rule_metadata: dict) -> str:
    """Combine rule metadata into searchable text."""
    rule = rule_metadata.get(rule_id, {})

    pieces = [rule_id]

    for key in ("name", "shortDescription", "fullDescription", "help"):
        value = rule.get(key, {})

        if isinstance(value, dict):
            pieces.append(value.get("text", ""))
        elif isinstance(value, str):
            pieces.append(value)

    return " ".join(pieces).lower()


def is_control_flow_candidate(
    result: dict,
    rule_metadata: dict,
) -> bool:
    """Identify potentially relevant findings for manual review."""
    rule_id = result.get("ruleId", "")
    message = result.get("message", {}).get("text", "")

    searchable = (
        get_rule_text(rule_id, rule_metadata)
        + " "
        + message.lower()
    )

    return any(
        keyword in searchable
        for keyword in CONTROL_FLOW_KEYWORDS
    )


def finding_details(result: dict, rule_metadata: dict) -> dict:
    """Extract a compact finding record from SARIF."""
    rule_id = result.get("ruleId", "")
    rule = rule_metadata.get(rule_id, {})
    locations = result.get("locations", [])

    uri = "(location unavailable)"
    line = "?"
    column = "?"

    if locations:
        physical = (
            locations[0]
            .get("physicalLocation", {})
        )
        artifact = physical.get("artifactLocation", {})
        region = physical.get("region", {})

        uri = artifact.get("uri", uri)
        line = region.get("startLine", line)
        column = region.get("startColumn", column)

    message = (
        result.get("message", {}).get("text")
        or "No message provided"
    )

    severity = (
        result.get("level")
        or rule.get("defaultConfiguration", {}).get("level")
        or "unspecified"
    )

    return {
        "rule_id": rule_id,
        "rule_name": rule.get("name", rule_id),
        "severity": severity,
        "file": uri,
        "line": line,
        "column": column,
        "message": message.replace("\n", " "),
    }


def write_markdown(
    path: Path,
    findings: list[dict],
    candidates: list[dict],
) -> None:
    """Write a readable report with full and focused findings."""
    lines = [
        "# Mythri Control-Flow-Focused CodeQL Audit",
        "",
        f"- Generated: {datetime.now().astimezone().isoformat()}",
        f"- Total findings in selected suite: {len(findings)}",
        f"- Control-flow candidates by keyword: {len(candidates)}",
        "",
        "> The candidate list is a triage aid, not a definitive classification.",
        "> Review source context before deciding whether a finding is a defect.",
        "",
        "## Control-flow candidates",
        "",
    ]

    if not candidates:
        lines.append(
            "No findings matched the configured keywords. "
            "This does not establish that control-flow defects are absent."
        )
    else:
        for item in candidates:
            lines.extend([
                f"### {item['rule_name']}",
                "",
                f"- Rule: `{item['rule_id']}`",
                f"- Severity: **{item['severity']}**",
                f"- Location: `{item['file']}:{item['line']}`",
                f"- Message: {item['message']}",
                "",
            ])

    lines.extend(["## All findings", ""])

    if not findings:
        lines.append("The selected suite reported no findings.")
    else:
        for item in findings:
            lines.append(
                f"- **{item['severity']}** `{item['file']}:"
                f"{item['line']}` — `{item['rule_id']}`: "
                f"{item['message']}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a control-flow-focused CodeQL audit."
    )
    parser.add_argument(
        "--codeql",
        type=Path,
        default=DEFAULT_CODEQL,
        help="Path to the CodeQL CLI executable.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="Path to an existing CodeQL database.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=DEFAULT_SUITE,
        help="CodeQL query suite to execute.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for timestamped reports.",
    )
    args = parser.parse_args()

    for label, path in (
        ("CodeQL CLI", args.codeql),
        ("CodeQL database", args.database),
        ("Query suite", args.suite),
    ):
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = args.output / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)

    sarif_path = report_dir / "control_flow_full.sarif"
    markdown_path = report_dir / "control_flow_report.md"
    json_path = report_dir / "control_flow_summary.json"

    try:
        run_command([
            str(args.codeql),
            "database", "analyze",
            str(args.database),
            str(args.suite),
            "--format=sarif-latest",
            f"--output={sarif_path}",
            "--threads=0",
        ])

        results, metadata = load_sarif(sarif_path)

        findings = [
            finding_details(result, metadata)
            for result in results
        ]

        candidates = [
            item
            for result, item in zip(results, findings)
            if is_control_flow_candidate(result, metadata)
        ]

        write_markdown(markdown_path, findings, candidates)

        summary = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "database": str(args.database),
            "suite": str(args.suite),
            "total_findings": len(findings),
            "control_flow_candidates": len(candidates),
            "findings_by_severity": {
                severity: sum(
                    1 for item in findings
                    if item["severity"] == severity
                )
                for severity in sorted({
                    item["severity"] for item in findings
                })
            },
            "reports": {
                "sarif": str(sarif_path),
                "markdown": str(markdown_path),
            },
        }

        json_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print("\nAUDIT COMPLETE")
        print(f"All findings: {len(findings)}")
        print(f"Control-flow candidates: {len(candidates)}")
        print(f"SARIF: {sarif_path}")
        print(f"Markdown: {markdown_path}")
        print(f"JSON summary: {json_path}")

        return 0

    except (RuntimeError, OSError, json.JSONDecodeError) as error:
        print(f"\nAUDIT FAILED: {error}", file=sys.stderr)
        print(
            f"Check any partial reports in: {report_dir}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
