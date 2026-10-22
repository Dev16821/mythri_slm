
#!/usr/bin/env python3
"""
Mythri CodeQL Audit Runner
- Rebuilds the Python CodeQL database from the current source.
- Runs security and quality suites sequentially.
- Produces SARIF reports and a consolidated Markdown summary.
- Never edits Mythri source files.
"""

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


HOME = Path.home()
REPO = Path.cwd()
DB = HOME / "codeql-databases" / "mythri-python"
REPORT_DIR = HOME / "codeql-databases" / "mythri-audit-reports"

KNOWN_CLI = (
    HOME
    / ".config/Code/User/globalStorage/github.vscode-codeql"
    / "distribution1/codeql/codeql"
)

SUITES = (
    "python-security-and-quality.qls",
    "python-code-quality-extended.qls",
    "python-security-extended.qls",
)


def find_cli():
    """Find the CodeQL CLI installed by the VS Code extension."""
    candidates = [
        KNOWN_CLI,
        Path(shutil.which("codeql") or "/nonexistent"),
    ]

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    raise RuntimeError(
        "CodeQL CLI was not found. Checked the VS Code extension path "
        "and PATH. Verify the existing CLI installation."
    )


def find_suite_root():
    """Find the installed Python CodeQL query pack."""
    root = HOME / ".codeql/packages/codeql/python-queries"
    if not root.is_dir():
        raise RuntimeError(f"Python CodeQL query pack not found: {root}")

    versions = sorted(
        (
            p for p in root.iterdir()
            if p.is_dir() and (p / "codeql-suites").is_dir()
        ),
        key=lambda p: p.name,
        reverse=True,
    )

    for version in versions:
        suite_dir = version / "codeql-suites"
        if all((suite_dir / name).is_file() for name in SUITES):
            return suite_dir

    raise RuntimeError(
        f"Could not find all required suites under {root}. "
        "Check the installed Python query pack."
    )


def run_command(command, label):
    """Run one command and stop clearly if it fails."""
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    print("$", " ".join(map(str, command)), flush=True)

    result = subprocess.run(
        list(map(str, command)),
        cwd=REPO,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}. "
            "Read the command output above. No source files were changed."
        )


def load_findings(sarif_path):
    """Read CodeQL SARIF results."""
    data = json.loads(sarif_path.read_text(encoding="utf-8"))
    findings = []

    for run in data.get("runs", []):
        rules = {
            rule.get("id"): rule
            for rule in run.get("tool", {}).get("driver", {}).get("rules", [])
        }

        for result in run.get("results", []):
            rule_id = result.get("ruleId", "unknown")
            rule = rules.get(rule_id, {})
            properties = rule.get("properties", {})
            security_severity = properties.get("security-severity")
            level = result.get("level", "warning")

            # CodeQL security-severity is normally a string from 0.0 to 10.0.
            try:
                security_score = float(security_severity)
            except (TypeError, ValueError):
                security_score = None

            locations = result.get("locations", [])
            if not locations:
                locations = [None]

            for location in locations:
                physical = (
                    location.get("physicalLocation", {})
                    if location else {}
                )
                artifact = physical.get("artifactLocation", {})
                region = physical.get("region", {})

                findings.append({
                    "rule": rule_id,
                    "description": (
                        rule.get("shortDescription", {}).get("text", "")
                    ),
                    "message": result.get("message", {}).get("text", ""),
                    "level": level,
                    "security_score": security_score,
                    "path": artifact.get("uri", "unknown"),
                    "line": region.get("startLine", "?"),
                })

    return findings


def main():
    if not (REPO / ".git").exists() and not (REPO / "model").exists():
        raise RuntimeError(
            f"{REPO} does not look like the Mythri repository. "
            "Run this script from ~/project/mythri_slm."
        )

    cli = find_cli()
    suite_dir = find_suite_root()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("MYTHRI CODEQL AUDIT")
    print(f"Repository : {REPO}")
    print(f"CodeQL CLI : {cli}")
    print(f"Database   : {DB}")
    print(f"Report dir : {REPORT_DIR}")
    print("\nSource files will not be modified.")

    # Rebuild the database so the audit reflects the current source tree.
    run_command(
        [
            cli, "database", "create", DB,
            "--language=python",
            f"--source-root={REPO}",
            "--overwrite",
        ],
        "1/4 — Build fresh CodeQL database",
    )

    all_findings = []
    report_paths = []

    for index, suite in enumerate(SUITES, start=2):
        output = REPORT_DIR / suite.replace(".qls", ".sarif")

        run_command(
            [
                cli, "database", "analyze", DB,
                suite_dir / suite,
                "--format=sarif-latest",
                f"--output={output}",
            ],
            f"{index}/4 — Analyze {suite}",
        )

        report_paths.append(output)
        findings = load_findings(output)

        for finding in findings:
            finding["suite"] = suite

        all_findings.extend(findings)
        print(f"\nFindings in this report: {len(findings)}")

    # Keep one Markdown report for convenient review.
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    summary_path = REPORT_DIR / "audit-summary.md"

    counts = Counter(f["rule"] for f in all_findings)
    security_findings = [
        f for f in all_findings if f["security_score"] is not None
    ]

    lines = [
        "# Mythri CodeQL Audit",
        "",
        f"- Generated: {timestamp}",
        f"- Repository: `{REPO}`",
        "- Scope: Python static analysis",
        "- Source files modified: No",
        f"- Total reported locations across suites: {len(all_findings)}",
        f"- Findings with a CodeQL security-severity score: "
        f"{len(security_findings)}",
        "",
        "> Counts across suites may include duplicate findings. "
        "A reported finding is a lead for investigation, not automatic "
        "proof of a runtime defect.",
        "",
        "## Findings by rule",
        "",
    ]

    if counts:
        lines.extend(
            f"- `{rule}`: {count}"
            for rule, count in counts.most_common()
        )
    else:
        lines.append("No findings were reported by the selected suites.")

    lines.extend(["", "## Finding details", ""])

    if all_findings:
        for number, finding in enumerate(all_findings, start=1):
            score = finding["security_score"]
            severity = (
                f"security score {score:g}"
                if score is not None
                else f"CodeQL level: {finding['level']}"
            )
            lines.extend([
                f"### {number}. `{finding['rule']}`",
                f"- Suite: `{finding['suite']}`",
                f"- Location: `{finding['path']}:{finding['line']}`",
                f"- Classification: {severity}",
                f"- Description: {finding['description'] or 'Not supplied'}",
                f"- Message: {finding['message'] or 'Not supplied'}",
                "",
            ])
    else:
        lines.append("No findings to list.")

    lines.extend(["", "## SARIF reports", ""])
    lines.extend(f"- `{path}`" for path in report_paths)

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{'=' * 72}")
    print("AUDIT COMPLETE")
    print(f"{'=' * 72}")
    print(f"Total reported locations: {len(all_findings)}")
    print(f"Security-scored locations: {len(security_findings)}")
    print(f"Summary: {summary_path}")
    print(f"SARIF reports: {REPORT_DIR}")

    if not all_findings:
        print(
            "\nNo findings were reported by these suites. "
            "This does not establish that the code is defect-free."
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAudit interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"\nAUDIT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
