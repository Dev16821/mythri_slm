#!/usr/bin/env python3
"""
Mythri Code Quality Audit
Run with:
    python3 quality_audit.py
or from the Mythri repository root:
    python3 codeql/quality_audit.py

The script uses sensible defaults and does not modify application source files.
It writes an Excel report and logs to ~/codeql-databases/mythri-quality-reports/.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
except ImportError:
    print("Missing dependency: openpyxl")
    print("Install it with: python3 -m pip install --user openpyxl")
    sys.exit(2)


HOME = Path.home()
SCRIPT_DIR = Path(__file__).resolve().parent
# If this script lives in <repo>/codeql/, the repository is its parent.
DEFAULT_REPO = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() in {"codeql", "scripts", "tools"} else SCRIPT_DIR
DEFAULT_CODEQL = HOME / ".config/Code/User/globalStorage/github.vscode-codeql/distribution1/codeql/codeql"
DEFAULT_DB = HOME / "codeql-databases/mythri-python"
DEFAULT_OUT = HOME / "codeql-databases/mythri-quality-reports"


def find_codeql() -> Path | None:
    candidates = [
        DEFAULT_CODEQL,
        HOME / ".config/Code/User/globalStorage/github.vscode-codeql/distribution/codeql",
        HOME / ".config/Code/User/globalStorage/github.vscode-codeql/codeql",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    # Last-resort search inside the known VS Code extension storage.
    root = HOME / ".config/Code/User/globalStorage/github.vscode-codeql"
    if root.exists():
        for candidate in root.glob("**/codeql"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    from shutil import which
    found = which("codeql")
    return Path(found) if found else None


def find_suite() -> Path | None:
    roots = [
        HOME / ".config/Code/User/globalStorage/github.vscode-codeql",
        HOME / ".config/Code/User/globalStorage",
    ]
    for root in roots:
        if root.exists():
            matches = list(root.glob("**/python-code-quality-extended.qls"))
            if matches:
                return matches[0]
    return None


def run_command(command: list[str], log_path: Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False)
        output = proc.stdout or ""
        log_path.write_text(output, encoding="utf-8", errors="replace")
        return proc.returncode, output
    except Exception as exc:
        message = f"Could not run command: {exc}\n"
        log_path.write_text(message, encoding="utf-8")
        return 1, message


def collect_python_files(repo: Path) -> list[Path]:
    excluded = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"}
    results = []
    for path in repo.rglob("*.py"):
        if any(part in excluded for part in path.relative_to(repo).parts):
            continue
        # Do not audit this utility itself as application code if stored under codeql/.
        results.append(path)
    return sorted(results)


def analyse_file(path: Path, repo: Path) -> tuple[list[dict], dict]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return ([{
            "File": str(path.relative_to(repo)),
            "Line": exc.lineno or "",
            "Category": "Syntax error",
            "Detail": exc.msg,
            "Severity": "HIGH",
        }], {"File": str(path.relative_to(repo)), "Lines": 0, "Functions": 0,
             "Classes": 0, "Imports": 0, "Long functions (>50 lines)": 0,
             "Bare except handlers": 0, "Syntax status": "ERROR"})

    lines = source.splitlines()
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    long_functions = []
    findings = []

    for node in functions:
        end = getattr(node, "end_lineno", node.lineno)
        length = end - node.lineno + 1
        if length > 50:
            long_functions.append(node)
            findings.append({
                "File": str(path.relative_to(repo)), "Line": node.lineno,
                "Category": "Long function", "Detail": f"{node.name} spans {length} lines; consider splitting responsibilities.",
                "Severity": "LOW",
            })

    bare_except_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            bare_except_count += 1
            findings.append({
                "File": str(path.relative_to(repo)), "Line": node.lineno,
                "Category": "Bare except", "Detail": "Catches every exception; consider catching specific exception types and logging useful context.",
                "Severity": "MEDIUM",
            })
        elif isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name) and node.type.id == "Exception":
            # Broad Exception is reported as a review candidate, not automatically a defect.
            findings.append({
                "File": str(path.relative_to(repo)), "Line": node.lineno,
                "Category": "Broad exception handler", "Detail": "Review whether this handler should catch narrower exception types.",
                "Severity": "LOW",
            })

    metrics = {
        "File": str(path.relative_to(repo)),
        "Lines": len(lines),
        "Functions": len(functions),
        "Classes": len(classes),
        "Imports": len(imports),
        "Long functions (>50 lines)": len(long_functions),
        "Bare except handlers": bare_except_count,
        "Syntax status": "OK",
    }
    return findings, metrics


def style_sheet(ws):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="244062")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for col_idx, column_cells in enumerate(ws.columns, 1):
        max_len = 0
        for cell in list(column_cells)[:250]:
            value = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, min(len(value), 70))
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max(max_len + 2, 12), 55)


def add_sheet(wb, name: str, headers: list[str], rows: list[dict] | list[list]):
    ws = wb.create_sheet(name)
    ws.append(headers)
    for row in rows:
        if isinstance(row, dict):
            ws.append([row.get(header, "") for header in headers])
        else:
            ws.append(row)
    style_sheet(ws)
    return ws


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a read-only Mythri Python code-quality audit using automatic defaults."
    )
    parser.add_argument("--repo", default=str(DEFAULT_REPO), help="Optional repository path (default: detected from this script).")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Optional output folder.")
    parser.add_argument("--skip-codeql", action="store_true", help="Only run Python AST checks; skip CodeQL.")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not (repo / ".git").exists():
        print(f"ERROR: {repo} does not look like a Git repository (missing .git).")
        print("Run this script from the Mythri repo's codeql folder, or use --repo /path/to/mythri_slm")
        return 2

    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workbook_path = out / f"mythri_quality_audit_{stamp}.xlsx"
    logs_dir = out / "logs"
    logs_dir.mkdir(exist_ok=True)

    py_files = collect_python_files(repo)
    findings, metrics = [], []
    for path in py_files:
        file_findings, file_metrics = analyse_file(path, repo)
        findings.extend(file_findings)
        metrics.append(file_metrics)

    codeql = find_codeql()
    suite = find_suite()
    codeql_status = "Skipped by user" if args.skip_codeql else "Not run"
    codeql_findings = []
    codeql_message = ""

    if not args.skip_codeql:
        if not codeql:
            codeql_status = "CLI not found; AST checks completed"
            codeql_message = "Could not locate CodeQL CLI automatically."
        elif not suite:
            codeql_status = "Query suite not found; AST checks completed"
            codeql_message = "Could not locate python-code-quality-extended.qls automatically."
        elif not DEFAULT_DB.exists():
            codeql_status = "Database not found; AST checks completed"
            codeql_message = f"Expected CodeQL database at {DEFAULT_DB}."
        else:
            sarif_path = out / f"mythri_quality_{stamp}.sarif"
            log_path = logs_dir / f"codeql_{stamp}.log"
            cmd = [str(codeql), "database", "analyze", str(DEFAULT_DB), str(suite),
                   "--format=sarif-latest", f"--output={sarif_path}"]
            rc, output = run_command(cmd, log_path)
            codeql_status = "Completed" if rc == 0 else f"Failed (exit {rc})"
            codeql_message = output[-12000:]
            if rc == 0 and sarif_path.exists():
                try:
                    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
                    rule_names = {}
                    for run in sarif.get("runs", []):
                        for rule in run.get("tool", {}).get("driver", {}).get("rules", []):
                            rule_names[rule.get("id", "")] = rule.get("name") or rule.get("id", "")
                        for result in run.get("results", []):
                            locs = result.get("locations", [])
                            loc = locs[0].get("physicalLocation", {}) if locs else {}
                            artifact = loc.get("artifactLocation", {})
                            region = loc.get("region", {})
                            codeql_findings.append({
                                "Rule": rule_names.get(result.get("ruleId", ""), result.get("ruleId", "")),
                                "Rule ID": result.get("ruleId", ""),
                                "Severity": result.get("level", ""),
                                "Message": result.get("message", {}).get("text", ""),
                                "File": artifact.get("uri", ""),
                                "Line": region.get("startLine", ""),
                            })
                except Exception as exc:
                    codeql_message += f"\nCould not parse SARIF: {exc}"

    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    summary.append(["Metric", "Value"])
    summary.append(["Repository", str(repo)])
    summary.append(["Python files scanned", len(py_files)])
    summary.append(["AST review candidates", len(findings)])
    summary.append(["CodeQL status", codeql_status])
    summary.append(["CodeQL findings", len(codeql_findings)])
    summary.append(["Report generated", datetime.now().isoformat(timespec="seconds")])
    summary.append(["Interpretation", "AST checks are heuristic review prompts, not proof of defects. CodeQL uses the existing database, which may be stale."])
    style_sheet(summary)

    add_sheet(wb, "File Metrics",
              ["File", "Lines", "Functions", "Classes", "Imports", "Long functions (>50 lines)", "Bare except handlers", "Syntax status"],
              metrics)
    add_sheet(wb, "AST Review Candidates",
              ["File", "Line", "Category", "Detail", "Severity"], findings)
    add_sheet(wb, "CodeQL Findings",
              ["Rule", "Rule ID", "Severity", "Message", "File", "Line"], codeql_findings)
    log_rows = [
        {"Item": "CodeQL CLI", "Value": str(codeql) if codeql else "Not found"},
        {"Item": "Query suite", "Value": str(suite) if suite else "Not found"},
        {"Item": "Database", "Value": str(DEFAULT_DB)},
        {"Item": "CodeQL status", "Value": codeql_status},
        {"Item": "Details", "Value": codeql_message[:30000]},
        {"Item": "Scope", "Value": "Python AST review + optional CodeQL quality suite; no application source files are changed."},
    ]
    add_sheet(wb, "Run Details", ["Item", "Value"], log_rows)
    wb.save(workbook_path)

    print("\nMythri quality audit finished.")
    print(f"Repository : {repo}")
    print(f"Python files scanned: {len(py_files)}")
    print(f"AST review candidates: {len(findings)}")
    print(f"CodeQL status: {codeql_status}")
    print(f"Excel report: {workbook_path}")
    print(f"Logs folder : {logs_dir}")
    print("\nNo application source files were modified.")
    if codeql_message and codeql_status != "Completed":
        print(f"\nNote: {codeql_message[:1200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
