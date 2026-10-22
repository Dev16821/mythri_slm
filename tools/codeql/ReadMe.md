# Mythri CodeQL Audit Scripts

This directory contains read-only audit tooling for the Mythri SLM repository. The scripts analyze source code, inspect Git changes, execute CodeQL queries, and generate reports. **The audit scripts are designed not to modify application source files.** They may create or overwrite their own databases, reports, logs, and query templates in designated output locations.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Scripts at a Glance](#2-scripts-at-a-glance)
3. [Individual Audit Scripts](#3-individual-audit-scripts)
4. [Combined Master Report](#4-combined-master-report)
5. [Excel Formatting and Severity](#5-excel-formatting-and-severity)
6. [Workbook Sheets](#6-workbook-sheets)
7. [Recommended Workflow](#7-recommended-workflow)
8. [Reading and Interpreting Results](#8-reading-and-interpreting-results)
9. [Known Issues and Limitations](#9-known-issues-and-limitations)
10. [Quick Reference](#10-quick-reference)

---

## 1. Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.9 or newer |
| `openpyxl` | Install using `python3 -m pip install --user openpyxl` |
| CodeQL CLI | Installed through the VS Code CodeQL extension |
| Python query pack | Installed under `~/.codeql/packages/codeql/python-queries/<version>/` |
| Git | Required by the change audit and quality audit |

Default CodeQL CLI location:

```text
~/.config/Code/User/globalStorage/github.vscode-codeql/distribution1/codeql/codeql
```

Most reports are written under:

```text
~/codeql-databases/
```

Run commands from the Mythri repository root unless otherwise specified:

```bash
cd ~/project/mythri_slm
```

## 2. Scripts at a Glance

| Script | Purpose | Requires CodeQL database? | Main outputs |
|---|---|---|---|
| `codeql_audit.py` | Rebuilds the database and runs three query suites | Creates and refreshes it | SARIF files and Markdown summary |
| `data_flow_audit.py` | Analyzes security findings and source-to-sink flow paths | Yes | Excel workbook and SARIF |
| `control_flow_audit.py` | Filters code-quality findings for control-flow-related candidates | Yes | SARIF, Markdown, and JSON |
| `quality_audit.py` | Performs AST checks and optionally runs the CodeQL quality suite | Optional | Excel workbook, SARIF, and logs |
| `mythri_change_audit.py` | Audits Git changes and generates custom query templates | Only for custom query execution | Excel workbook, Git patch, and `.ql` files |
| `sarif_to_excel.py` | Converts existing SARIF reports into an Excel workbook | No | Excel workbook |
| `main.py` | Coordinates configured audits and combines their outputs | Depends on the configured stages | Timestamped master Excel report and supporting files |

---

## 3. Individual Audit Scripts

### 3.1 `codeql_audit.py`

**Purpose**

Builds a fresh CodeQL database from the current repository source and executes three Python query suites sequentially:

1. `python-security-and-quality.qls`
2. `python-code-quality-extended.qls`
3. `python-security-extended.qls`

It produces a consolidated Markdown summary containing findings grouped by rule, including the suite, location, severity classification, and message.

**Run**

```bash
python3 codeql/codeql_audit.py
```

The script checks that the current directory resembles the Mythri repository by looking for a `.git` or `model` directory. It stops with an error if the repository check fails.

**Inputs**

The CodeQL CLI and installed Python query pack are located automatically.

**Outputs**

```text
~/codeql-databases/
├── mythri-python/
└── mythri-audit-reports/
    ├── python-security-and-quality.sarif
    ├── python-code-quality-extended.sarif
    ├── python-security-extended.sarif
    └── audit-summary.md
```

**Important notes**

- This is the database-rebuilding stage. Run it after source changes before relying on other CodeQL-based audits.
- The database at `~/codeql-databases/mythri-python` is overwritten on each rebuild.
- The query suites overlap, so the same issue may appear in multiple results and inflate raw totals.

### 3.2 `data_flow_audit.py`

**Purpose**

Runs CodeQL queries against an existing database and exports the findings into Excel. It includes step-by-step source-to-sink paths when those paths are supplied in the SARIF results.

The default query suite is `python-security-extended.qls`. An additional `.ql` query or `.qls` suite can be supplied.

**Run**

```bash
python3 codeql/data_flow_audit.py
```

To include a custom query:

```bash
python3 codeql/data_flow_audit.py --custom-query path/to/query.ql
```

**Options**

| Option | Description |
|---|---|
| `--codeql` | Path to the CodeQL CLI |
| `--database` | CodeQL database; defaults to `~/codeql-databases/mythri-python` |
| `--suite` | Query suite; defaults to `python-security-extended.qls` |
| `--custom-query` | Optional additional `.ql` or `.qls` file |
| `--output-dir` | Output root; defaults to `~/codeql-databases/mythri-data-flow-reports` |

**Outputs**

A timestamped directory containing raw `data_flow_<n>.sarif` files and a workbook named:

```text
mythri_data_flow_audit_<timestamp>.xlsx
```

Expected sheets:

- **Executive Summary** — totals, severity counts, and interpretation notes.
- **Data Flow Findings** — findings and their fingerprints.
- **Flow Paths** — individual steps in reported data-flow paths.
- **Rule Summary** — findings grouped by rule.
- **Execution Log** — query execution results.

**Known issue**

In `load_sarif`, the primary location is extracted using:

```python
sarif_location(locations[0].get("physicalLocation", {}))
```

This causes the File and Line columns in **Data Flow Findings** to be blank when `sarif_location()` expects the complete location object.

The intended correction is:

```python
sarif_location(locations[0])
```

This issue should be verified against the current source before applying a fix.

**Additional note**

The default suite path contains a hard-coded query-pack version, `1.8.12`. If the installed version differs, pass `--suite` explicitly.

### 3.3 `control_flow_audit.py`

**Purpose**

Runs `python-code-quality-extended.qls` against the existing database and filters the results for control-flow-related candidates.

Matched keywords include:

- `unreachable`
- `empty except`
- `always true`
- `always false`
- `dead code`
- `useless assignment`
- `loop`
- `branch`
- `fall through`

**Run**

```bash
python3 codeql/control_flow_audit.py
```

**Options**

| Option | Description |
|---|---|
| `--codeql` | Path to the CodeQL CLI |
| `--database` | CodeQL database |
| `--suite` | Query suite; defaults to `python-code-quality-extended.qls` |
| `--output` | Output root; defaults to `~/codeql-databases/mythri-control-flow-reports` |

**Outputs**

A timestamped directory containing:

```text
control_flow_full.sarif
control_flow_report.md
control_flow_summary.json
```

- `control_flow_full.sarif` contains all findings from the executed suite.
- `control_flow_report.md` lists keyword-matched candidates first, followed by all findings.
- `control_flow_summary.json` contains totals and findings grouped by severity.

**Limitations**

This is keyword-based triage, not a control-flow graph. Broad terms such as `loop` and `branch` can match unrelated findings. Inspect the source before treating a candidate as a defect.

### 3.4 `quality_audit.py`

**Purpose**

Performs two types of code-quality analysis.

**Part A — Python AST checks**

Scans Python files in the repository while skipping directories such as `.git`, virtual environments, `__pycache__`, `node_modules`, `build`, and `dist`.

It checks for:

| Check | Assigned severity |
|---|---|
| Python syntax errors | HIGH |
| Bare `except:` handlers | MEDIUM |
| Broad `except Exception` handlers | LOW |
| Functions longer than 50 lines | LOW |

It also collects per-file metrics, including lines, functions, classes, and imports.

**Part B — CodeQL quality analysis**

Optionally executes `python-code-quality-extended.qls` against the existing database if the CLI, suite, and database are available.

**Run**

```bash
python3 codeql/quality_audit.py
```

Run only the AST checks:

```bash
python3 codeql/quality_audit.py --skip-codeql
```

Specify the repository root:

```bash
python3 codeql/quality_audit.py --repo ~/project/mythri_slm
```

**Options**

| Option | Description |
|---|---|
| `--repo` | Repository root; defaults to the parent of the script directory and must contain `.git` |
| `--out` | Output folder; defaults to `~/codeql-databases/mythri-quality-reports` |
| `--skip-codeql` | Runs AST checks without CodeQL |

**Outputs**

```text
mythri_quality_audit_<timestamp>.xlsx
```

The workbook contains:

- **Summary**
- **File Metrics**
- **AST Review Candidates**
- **CodeQL Findings**
- **Run Details**

The output also includes a SARIF file and a `logs/` directory.

**Important notes**

- AST findings are heuristic review prompts, not proof of defects.
- If CodeQL cannot run, the script records the reason in the Summary and Run Details sheets.
- The script does not rebuild the database. Run `codeql_audit.py` first if the source has changed.

### 3.5 `mythri_change_audit.py`

**Purpose**

Performs a read-only Git change audit and prioritizes changed files using path-based rules.

It compares the working tree, including tracked, staged, and untracked files, against a baseline revision. The default baseline is `HEAD`.

**Priority rules**

| Priority | Example path keywords |
|---|---|
| CRITICAL | `safety`, `deterministic_policy`, `risk`, `crisis`, `threshold`, `predictor` |
| HIGH | `inference`, `model`, `multi_task`, `emotion`, `config`, `experiment`, `training`, `dataset` |
| MEDIUM | Test-related paths |
| NORMAL | Other paths |

These priorities are based on path names and are intended to guide triage.

The script also generates three custom CodeQL query templates for potential flows involving:

1. Risk information and policy decisions.
2. Emotion logits and probabilities or labels.
3. Configuration values and implementation settings.

The templates run only when explicitly requested.

**Run**

```bash
python3 codeql/mythri_change_audit.py
```

Compare against an earlier baseline:

```bash
python3 codeql/mythri_change_audit.py --baseline HEAD~3
```

Execute the custom queries:

```bash
python3 codeql/mythri_change_audit.py --run-codeql
```

**Options**

| Option | Description |
|---|---|
| `--repo` | Git repository; defaults to `~/project/mythri_slm` |
| `--out` | Output folder; defaults to `~/codeql-databases/mythri-change-audit` |
| `--codeql` | CodeQL CLI path for custom query execution |
| `--database` | Existing CodeQL database for custom query execution |
| `--run-codeql` | Runs the three custom queries |
| `--baseline` | Git revision for comparison; defaults to `HEAD` |
| `--include-diff-text` | Adds a Diff Details sheet containing source lines |
| `--max-diff-lines` | Maximum lines in the diff sheet; defaults to 20,000 |

**Outputs**

The timestamped output includes:

- `mythri_change_audit_<timestamp>.xlsx`
- `git_diff_<timestamp>.patch`
- `custom_queries/` containing generated `.ql` templates
- One SARIF file per custom query when `--run-codeql` is used

The workbook contains:

- **Executive Summary**
- **Changed Files**
- **Risk Summary**
- **Custom Queries**
- **Custom Query Findings**
- **Execution Log**
- **Diff Details**, when enabled

The patch includes tracked changes and untracked files, subject to the script's per-file size limit of 2 MB.

**Important notes**

- File priorities are triage aids, not security verdicts.
- Diff text and patches can contain sensitive source code. Review them before sharing.
- The custom queries are templates and may need identifier-pattern adjustments, path-query changes, or a `qlpack.yml` file before they compile.
- Query compilation failures should be reviewed in the Custom Queries sheet.

### 3.6 `sarif_to_excel.py`

**Purpose**

Converts existing SARIF files into a standalone Excel workbook. It does not execute CodeQL or rebuild the database.

**Run**

```bash
python3 codeql/sarif_to_excel.py
```

The script takes no arguments.

**Input directory**

```text
~/codeql-databases/mythri-audit-reports/
```

The script imports whichever of these files exist:

```text
mythri-security-results.sarif
mythri-quality-extended.sarif
mythri-security-extended.sarif
python-security-and-quality.sarif
python-code-quality-extended.sarif
python-security-extended.sarif
```

It stops with an error if none are found.

**Output**

```text
~/codeql-databases/mythri-audit-reports/excel/
└── mythri_codeql_<timestamp>.xlsx
```

Workbook sheets:

- **All Findings** — every finding location, including severity, security score, tags, and duplicate indicators.
- **Security Findings** — findings with a security score, security-related rule name, or CWE tag.
- **Summary** — counts grouped by rule and source file.
- **Rule Reference** — one row per rule and report.

**Important note**

All rows are retained, and duplicates are flagged rather than removed. Consequently, the totals may include overlap between suites.

---

## 4. Combined Master Report

### 4.1 Overview

`main.py` consolidates outputs from the configured audit scripts into a single Excel workbook.

Instead of manually opening each report, you can use the master workbook to review security findings, data-flow paths, control-flow candidates, AST metrics, Git changes, and custom query results.

**Run from the repository root:**

```bash
python3 codeql/main.py
```

The exact stages executed depend on the implementation of `main.py` and the availability of their required inputs.

### 4.2 Output Directory

The master report is saved in a new timestamped directory under:

```text
~/codeql-databases/mythri-master-reports/<timestamp>/
```

Expected structure:

```text
<timestamp>/
├── mythri_master_audit_<timestamp>.xlsx
├── data_flow/
│   ├── Raw SARIF results
│   └── Data-flow audit workbook
├── control_flow/
│   ├── Raw SARIF results
│   ├── Markdown report
│   └── JSON summary
├── quality/
│   ├── Quality audit workbook
│   ├── SARIF results
│   └── Logs
└── change/
    ├── Change audit workbook
    ├── Git patch
    └── Custom queries
```

This is the expected layout; the exact files depend on the outputs actually generated by each stage.

A new timestamped directory is created for each run so that previous master reports are preserved.

### 4.3 Finding Deduplication

The master workbook merges repeated findings using the combination of:

- Rule ID
- File
- Line number
- Message

Each unique finding receives a report-local ID, such as `F-0001`.

The **Reported By** column identifies every suite or script that reported the finding. This allows repeated reports of the same issue to be reviewed as a single entry.

Findings are ordered by severity, security score, file path, and line number. Dashboard totals count unique findings.

The master workbook's deduplication behavior differs from `sarif_to_excel.py`, which retains duplicate rows and flags them.

---

## 5. Excel Formatting and Severity

The master workbook uses formatting to make large audit results easier to navigate.

| Element or status | Formatting or behavior |
|---|---|
| Header rows | Dark blue background |
| Frozen headers | Remain visible while scrolling |
| Column filters | Enabled on columns |
| Long text | Wrapped inside cells |
| CRITICAL | Dark red with white text |
| HIGH | Light red |
| MEDIUM | Yellow |
| LOW | Light blue |
| INFO | Grey |
| SUCCESS | Green |
| FAILED | Red |
| SKIPPED / NOT RUN | Grey |
| Text beginning with `=` | Stored as plain text rather than an Excel formula |
| Very long cells | Content exceeding approximately 32,000 characters is truncated |

### 5.1 Security Score Mapping

When CodeQL supplies a security score, the master report uses it to determine severity.

| Security score | Severity |
|---|---|
| 9.0 and above | CRITICAL |
| 7.0–8.9 | HIGH |
| 4.0–6.9 | MEDIUM |
| Below 4.0 | LOW |

### 5.2 SARIF Fallback Mapping

When a security score is unavailable:

| SARIF level | Severity |
|---|---|
| `error` | HIGH |
| `warning` | MEDIUM |
| `note` or recommendation | LOW |
| Other or unrecognized levels | INFO |

A finding with an available security score is classified using the score rather than the fallback SARIF level.

---

## 6. Workbook Sheets

### 6.1 Dashboard

The main summary page contains:

- **Run Information:** Generation time, repository, output directory, CodeQL CLI version, database path, and database modification time.
- **Stage Results:** Each audit stage's status, duration, and details.
- **Findings Overview:** Unique findings, severity totals, findings with security scores, data-flow paths, control-flow candidates, and affected files.
- **Top 10 Rules:** Rules associated with the most findings.
- **Top 10 Hotspot Files:** Files with the largest concentrations of findings.
- **Git Change Context:** Branch, baseline revision, and change counts.
- **Warnings:** Missing results, skipped stages, or database rebuild issues.
- **Sheet Guide:** Workbook navigation and interpretation guidance.

### 6.2 Security Findings

Contains findings with a CodeQL security score or a security/CWE tag. Use this sheet as the primary starting point for security triage.

### 6.3 All Findings

Contains all unique findings, whether security-related or not.

| Column | Meaning |
|---|---|
| ID | Report-local finding ID, such as `F-0001` |
| Severity | CRITICAL, HIGH, MEDIUM, LOW, or INFO |
| Security Score | CodeQL score from 0 to 10, when available |
| CWE | Weakness identifier extracted from rule tags |
| Rule ID / Rule Name | Rule that generated the finding |
| File / Line | Location relative to the repository root |
| Message | Explanation of the finding |
| Reported By | Audit stages that reported the issue |
| Flow Path? | Whether CodeQL supplied an explicit source-to-sink path |
| Description | Short description of the rule |

### 6.4 Rule Summary

Provides one row per rule, including:

- Worst severity
- Number of findings
- Number of affected files
- Whether it is a security rule
- Rule description

Use it to identify the types of findings that dominate the audit.

### 6.5 File Hotspots

Combines CodeQL findings, quality metrics, and Git change information.

| Column | Meaning |
|---|---|
| Findings | Total CodeQL findings |
| Critical, High, Medium, Low | Counts by severity |
| Top Rule | Most frequently occurring rule in the file |
| AST Prompts | Number of quality-audit review prompts |
| Lines | File line count |
| Functions | Number of functions |
| Long Functions | Number of functions exceeding 50 lines |
| Bare Excepts | Number of bare exception handlers |
| Git Status | Change status relative to the baseline |
| Git Priority | Path-based change priority |

A file is included if it has CodeQL findings, AST review prompts, or a CRITICAL/HIGH-priority Git change.

Files are sorted by critical plus high findings, then total findings.

Prioritize files that combine security findings with changes to sensitive components.

### 6.6 Data Flow Paths

Contains individual steps in source-to-sink paths supplied by CodeQL.

| Column | Meaning |
|---|---|
| Finding ID | Links to the finding in All Findings |
| Rule ID / Severity | Associated rule and severity |
| Alert Location | Location where the alert was raised |
| Path # / Step # | Path identifier and step position |
| Step Message | Description of the operation at that step |
| File / Line / Code Snippet | Step location and any available snippet |

If no explicit paths are supplied, the sheet may be empty. This does not establish that the repository has no data-flow problems.

### 6.7 Control Flow Candidates

Contains findings matched by control-flow-related keywords, including unreachable code, empty exception handlers, redundant conditions, and dead code.

The **Matched Keywords** column explains why a result was included.

This is a triage list, not a control-flow graph or proof of incorrect behavior.

### 6.8 Git Changes

Lists files that differ from the selected baseline, including staged and untracked files.

The default baseline is `HEAD`.

| Column | Meaning |
|---|---|
| Working Status | Git status, such as `M`, `A`, or `??` |
| Diff Status | Status reported by the baseline comparison |
| Path / Previous Path | Current path and previous path for renames |
| Priority | CRITICAL, HIGH, MEDIUM, or NORMAL |
| Reason | Explanation of the assigned priority |

### 6.9 Custom Queries

Reports the execution status of the three custom `.ql` templates.

The default status is:

```text
TEMPLATE ONLY - not executed
```

If the queries are executed, the report records their outcomes and available error output.

### 6.10 Custom Query Findings

Appears only when executed custom queries return findings.

### 6.11 AST Metrics

Contains one row per Python file with:

- Lines
- Functions
- Classes
- Imports
- Long functions exceeding 50 lines
- Bare exception handlers
- Syntax status

### 6.12 AST Review

Lists heuristic review prompts for syntax errors, bare `except:` handlers, broad `except Exception` handlers, and long functions.

Prompts include a line number and severity where available. They require manual verification.

### 6.13 Run Log

Contains one row per audit stage with:

- Execution status
- Duration
- Exact command
- Details
- Final lines of output

If a stage fails, a sheet is empty, or expected results are missing, inspect the Run Log first.

---

## 7. Recommended Workflow

### Option A — Run the audits individually

From the repository root:

```bash
cd ~/project/mythri_slm

# 1. Refresh the CodeQL database and generate fresh SARIF
python3 codeql/codeql_audit.py

# 2. Inspect security data-flow results
python3 codeql/data_flow_audit.py

# 3. Inspect control-flow-related candidates
python3 codeql/control_flow_audit.py

# 4. Run AST and optional CodeQL quality checks
python3 codeql/quality_audit.py

# 5. Audit Git changes
python3 codeql/mythri_change_audit.py

# 6. Convert existing SARIF results into a standalone workbook
python3 codeql/sarif_to_excel.py
```

This approach is useful when individual reports need to be inspected or debugged independently.

### Option B — Generate the combined report

```bash
cd ~/project/mythri_slm
python3 codeql/main.py
```

Use this approach when the goal is to consolidate configured audit outputs into one master workbook.

Check the implementation of `main.py` to determine which stages it executes and whether it rebuilds the database. Do not assume that it refreshes the database unless its implementation explicitly does so.

If it does not rebuild the database, run `codeql_audit.py` first whenever source changes must be included in the analysis.

---

## 8. Reading and Interpreting Results

Follow this sequence when reviewing a report:

1. **Start with Dashboard.** Review run information, stage statuses, and warnings.
2. **Verify database freshness.** Check whether the database includes the latest source changes.
3. **Open Security Findings.** Filter by severity and investigate CRITICAL and HIGH results first.
4. **Inspect File Hotspots.** Identify files that combine numerous findings, AST prompts, and sensitive changes.
5. **Trace Data Flow Paths.** Follow the steps associated with a finding ID when a path is available.
6. **Review Git Changes.** Examine modified files and their path-based priorities.
7. **Check Custom Queries.** Distinguish executed queries from templates that were never run.
8. **Inspect Run Log.** Investigate failures, skipped stages, or unexpected empty sheets.

Every finding is a lead for investigation, not proof of a vulnerability or defect.

Likewise, an empty sheet does not prove that the corresponding class of problems is absent.

---

## 9. Known Issues and Limitations

- **Stale databases:** CodeQL-based scripts analyze the existing database. Run `codeql_audit.py` after source changes when fresh results are required.
- **Overlapping query suites:** The same issue can appear in multiple suites and inflate raw finding counts.
- **Missing data-flow locations:** The known `load_sarif` location-extraction issue can leave File and Line columns blank in the data-flow workbook.
- **Unavailable flow paths:** Some findings do not contain explicit source-to-sink paths in their SARIF output.
- **Heuristic control-flow filtering:** Keyword matches can include findings unrelated to actual control-flow defects.
- **Heuristic AST findings:** Long functions and broad exception handlers are review prompts, not automatically confirmed defects.
- **Custom query templates:** Templates may require identifier, path, and pack adjustments before execution.
- **Partial reports:** Failed or skipped stages can result in missing or incomplete workbook sheets.
- **Sensitive diff output:** Patches and diff sheets can contain source code and should be reviewed before sharing.
- **Standalone SARIF conversion:** `sarif_to_excel.py` flags duplicates but retains all rows, so its totals may differ from the deduplicated master report.
- **No findings is not proof of correctness:** Static analysis has coverage and detection limitations and should be complemented by testing and manual review.

---

## 10. Quick Reference

| Task | Command |
|---|---|
| Generate fresh CodeQL results | `python3 codeql/codeql_audit.py` |
| Run data-flow analysis | `python3 codeql/data_flow_audit.py` |
| Run control-flow triage | `python3 codeql/control_flow_audit.py` |
| Run quality audit | `python3 codeql/quality_audit.py` |
| Run AST checks only | `python3 codeql/quality_audit.py --skip-codeql` |
| Run Git change audit | `python3 codeql/mythri_change_audit.py` |
| Execute custom queries | `python3 codeql/mythri_change_audit.py --run-codeql` |
| Convert existing SARIF to Excel | `python3 codeql/sarif_to_excel.py` |
| Generate the combined master report | `python3 codeql/main.py` |
| List master report directories | `ls -lt ~/codeql-databases/mythri-master-reports/` |
| Inspect an output directory | `ls -lah ~/codeql-databases/mythri-master-reports/<timestamp>/` |

Replace `<timestamp>` with the actual directory name generated by the audit.

---

**Purpose:** Provide a consistent, traceable workflow for reviewing Mythri SLM's static-analysis findings, data-flow paths, code-quality indicators, and recent changes.

**Final reminder:** These tools support security auditing and code review. Their reports do not replace manual validation, runtime testing, or a complete security assessment.
