#!/usr/bin/env python3
"""Read-only Git change audit + custom CodeQL query runner for Mythri.

Writes only to the audit output directory. Does not edit application files,
Git index, commits, branches, or CodeQL database contents intentionally.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
except ImportError:
    sys.exit("Missing dependency: python3 -m pip install --user openpyxl")

HOME = Path.home()
DEFAULT_REPO = HOME / "project" / "mythri_slm"
DEFAULT_CODEQL = HOME / ".config/Code/User/globalStorage/github.vscode-codeql/distribution1/codeql/codeql"
DEFAULT_DB = HOME / "codeql-databases/mythri-python"
DEFAULT_OUT = HOME / "codeql-databases/mythri-change-audit"

# These are intentionally broad candidate-flow templates. Identifier matching must
# be adapted after reviewing the real source names; a result is a review lead,
# not proof that the production safety path is correct or incorrect.
QUERY_TEMPLATES = {
"mythri_risk_policy_flow.ql": r'''/**
 * @name Potential risk prediction to safety-policy flow
 * @description Finds possible data flow from risk-related identifiers to policy/threshold-related identifiers.
 * @kind path-problem
 * @problem.severity warning
 * @id mythri/risk-policy-flow
 */
import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.dataflow.new.TaintTracking

module RiskPolicyConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    exists(Name n |
      (n.getId().matches("%risk%") or n.getId().matches("%safety_score%")) and
      source = DataFlow::exprNode(n)
    )
  }
  predicate isSink(DataFlow::Node sink) {
    exists(Name n |
      (n.getId().matches("%policy%") or n.getId().matches("%threshold%") or
       n.getId().matches("%decision%") or n.getId().matches("%risk_type%")) and
      sink = DataFlow::exprNode(n)
    )
  }
}
module RiskPolicyFlow = TaintTracking::Global<RiskPolicyConfig>;

from DataFlow::Node source, DataFlow::Node sink
where RiskPolicyFlow::flowPath(source, sink)
select sink, source, sink, "Potential flow from risk-related data to a policy/decision-related expression; verify the actual call path and safety behavior."
''',
"mythri_emotion_probability_flow.ql": r'''/**
 * @name Potential emotion-logit to probability/label flow
 * @description Finds possible data flow from emotion-logit-related identifiers to probability/threshold/label-related identifiers.
 * @kind path-problem
 * @problem.severity warning
 * @id mythri/emotion-probability-flow
 */
import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.dataflow.new.TaintTracking

module EmotionFlowConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    exists(Name n |
      (n.getId().matches("%logit%") or n.getId().matches("%emotion_scores%")) and
      source = DataFlow::exprNode(n)
    )
  }
  predicate isSink(DataFlow::Node sink) {
    exists(Name n |
      (n.getId().matches("%prob%") or n.getId().matches("%threshold%") or
       n.getId().matches("%label%") or n.getId().matches("%emotion%")) and
      sink = DataFlow::exprNode(n)
    )
  }
}
module EmotionFlow = TaintTracking::Global<EmotionFlowConfig>;

from DataFlow::Node source, DataFlow::Node sink
where EmotionFlow::flowPath(source, sink)
select sink, source, sink, "Potential flow from emotion logits/scores to a probability, threshold, or label expression; verify the actual transformation and output contract."
''',
"mythri_config_consistency_candidates.ql": r'''/**
 * @name Configuration-related values reaching implementation decisions
 * @description Finds possible flow from config/setting identifiers to threshold/model/training decision identifiers.
 * @kind path-problem
 * @problem.severity warning
 * @id mythri/config-implementation-flow
 */
import python
import semmle.python.dataflow.new.DataFlow
import semmle.python.dataflow.new.TaintTracking

module ConfigImplementationConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    exists(Name n |
      (n.getId().matches("%config%") or n.getId().matches("%setting%") or
       n.getId().matches("%yaml%") or n.getId().matches("%params%")) and
      source = DataFlow::exprNode(n)
    )
  }
  predicate isSink(DataFlow::Node sink) {
    exists(Name n |
      (n.getId().matches("%threshold%") or n.getId().matches("%hidden_size%") or
       n.getId().matches("%num_heads%") or n.getId().matches("%learning_rate%") or
       n.getId().matches("%max_length%") or n.getId().matches("%batch_size%")) and
      sink = DataFlow::exprNode(n)
    )
  }
}
module ConfigImplementationFlow = TaintTracking::Global<ConfigImplementationConfig>;

from DataFlow::Node source, DataFlow::Node sink
where ConfigImplementationFlow::flowPath(source, sink)
select sink, source, sink, "Potential flow from a configuration-related identifier to an implementation setting; verify whether the configured value actually controls runtime behavior."
'''
}

RISK_TERMS = {
    "CRITICAL": ("safety", "deterministic_policy", "risk", "crisis", "threshold", "predictor"),
    "HIGH": ("inference", "model", "multi_task", "emotion", "config", "experiment", "training", "dataset"),
}

def run_git(repo: Path, args: list[str], check=True):
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(p.stderr.strip() or f"git {' '.join(args)} failed ({p.returncode})")
    return p

def git_text(repo: Path, args: list[str]) -> str:
    return run_git(repo, args).stdout.strip()

def classify(path: str) -> tuple[str, str]:
    low = path.lower()
    for priority, terms in RISK_TERMS.items():
        if any(t in low for t in terms):
            return priority, "Sensitive Mythri component/configuration; inspect semantics and tests"
    if "/test" in low or low.startswith("test") or "spec" in low:
        return "MEDIUM", "Test-related change; verify coverage and assertions"
    return "NORMAL", "General change; inspect diff and dependencies"

def parse_name_status(text: str):
    rows=[]
    for line in text.splitlines():
        if not line: continue
        parts=line.split("\t")
        status=parts[0]
        if status.startswith("R") or status.startswith("C"):
            old=parts[1] if len(parts)>1 else ""
            new=parts[2] if len(parts)>2 else old
            rows.append((status, new, old))
        else:
            rows.append((status, parts[1] if len(parts)>1 else "", ""))
    return rows

def write_sheet(ws, headers, rows):
    ws.append(headers)
    for row in rows: ws.append(list(row))
    ws.freeze_panes="A2"
    if ws.max_row > 1: ws.auto_filter.ref=ws.dimensions
    for c in ws[1]:
        c.font=Font(bold=True,color="FFFFFF")
        c.fill=PatternFill("solid",fgColor="243B53")
        c.alignment=Alignment(wrap_text=True,vertical="top")
    for col in ws.columns:
        letter=get_column_letter(col[0].column)
        width=min(max(max(len(str(c.value or "")) for c in col)+2,12),58)
        ws.column_dimensions[letter].width=width
    for row in ws.iter_rows(min_row=2):
        for c in row: c.alignment=Alignment(wrap_text=True,vertical="top")

def parse_sarif(path: Path):
    rows=[]
    if not path.exists(): return rows
    try: data=json.loads(path.read_text(encoding="utf-8"))
    except Exception: return rows
    for run in data.get("runs",[]):
        rules={r.get("id"):r.get("shortDescription",{}).get("text","") for r in run.get("tool",{}).get("driver",{}).get("rules",[])}
        for res in run.get("results",[]):
            loc=(res.get("locations") or [{}])[0].get("physicalLocation",{})
            uri=loc.get("artifactLocation",{}).get("uri","")
            line=loc.get("region",{}).get("startLine","")
            rows.append([res.get("ruleId",""),rules.get(res.get("ruleId",""),""),res.get("level",""),uri,line,res.get("message",{}).get("text","")])
    return rows

def main():
    ap=argparse.ArgumentParser(description="Read-only Mythri Git change analysis and optional custom CodeQL queries")
    ap.add_argument("--repo",type=Path,default=DEFAULT_REPO,help=f"Git repository root (default: {DEFAULT_REPO})")
    ap.add_argument("--out",type=Path,default=DEFAULT_OUT,help=f"Output directory for workbook, diffs, queries, logs and SARIF (default: {DEFAULT_OUT})")
    ap.add_argument("--codeql",type=Path,default=DEFAULT_CODEQL,help=f"CodeQL CLI executable (default: {DEFAULT_CODEQL})")
    ap.add_argument("--database",type=Path,default=DEFAULT_DB,help=f"Existing CodeQL database for Mythri (default: {DEFAULT_DB})")
    ap.add_argument("--run-codeql",action="store_true",help="Run the three custom queries against --database; requires a compatible existing CodeQL DB and CodeQL CLI")
    ap.add_argument("--baseline",default="HEAD",help="Git revision to compare against; default HEAD (latest commit on current branch)")
    ap.add_argument("--include-diff-text",action="store_true",help="Include unified diff text in a workbook sheet; may expose source snippets in the report")
    ap.add_argument("--max-diff-lines",type=int,default=20000,help="Maximum diff lines captured for the optional Diff Details sheet (default 20000)")
    args=ap.parse_args()
    repo=args.repo.expanduser().resolve(); out=args.out.expanduser().resolve()
    if not repo.is_dir(): sys.exit(f"Repository directory does not exist: {repo}")
    if args.max_diff_lines < 1: sys.exit("--max-diff-lines must be at least 1")
    out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    log=[]
    try:
        root=Path(git_text(repo,["rev-parse","--show-toplevel"])).resolve()
        if root != repo: log.append(["Git", "NOTE", f"Provided path resolves to Git root {root}"])
        head=git_text(repo,["rev-parse","HEAD"])
        base=git_text(repo,["rev-parse",args.baseline])
        branch=git_text(repo,["branch","--show-current"])
        status=run_git(repo,["status","--porcelain=v1","-uall"]).stdout
        tracked_diff=run_git(repo,["diff","--name-status",base,"--"]).stdout
        staged_diff=run_git(repo,["diff","--cached","--name-status",base,"--"]).stdout
        # HEAD comparison must include tracked staged + unstaged changes; git diff base captures unstaged working tree,
        # while --cached is recorded separately for explicit staging visibility.
        changed=parse_name_status(tracked_diff)
        changed_map={}
        for st,path,old in changed: changed_map[path]=(st,path,old)
        for st,path,old in parse_name_status(staged_diff): changed_map[path]=(st,path,old)
        # Untracked files are not in git diff. Add them explicitly.
        untracked=run_git(repo,["ls-files","--others","--exclude-standard"]).stdout.splitlines()
        for path in untracked: changed_map[path]=( "??", path, "")
        # Status codes from porcelain are more precise for working-tree/staged state.
        status_by_path={}
        for line in status.splitlines():
            code=line[:2]; p=line[3:]
            if " -> " in p: p=p.split(" -> ",1)[1]
            status_by_path[p]=code
        change_rows=[]
        for path,(st,p,old) in sorted(changed_map.items()):
            priority,reason=classify(path)
            change_rows.append([status_by_path.get(path,st),st,path,old,priority,reason])
        diff=run_git(repo,["diff","--no-ext-diff","--no-color",base,"--"]).stdout
        # Append untracked file contents as clearly marked additions for audit context.
        for path in untracked:
            fp=repo/path
            try:
                if fp.is_file() and fp.stat().st_size <= 2_000_000:
                    body=fp.read_text(encoding="utf-8",errors="replace")
                    diff += f"\n--- /dev/null\n+++ b/{path} (UNTRACKED)\n" + "\n".join("+"+x for x in body.splitlines()) + "\n"
            except OSError: pass
        diff_lines=diff.splitlines()
        diff_truncated=len(diff_lines)>args.max_diff_lines
        diff_excerpt="\n".join(diff_lines[:args.max_diff_lines])
        summary=[
            ["Repository",str(repo)],["Current branch",branch or "(detached HEAD)"],
            ["Baseline revision",args.baseline],["Resolved baseline SHA",base],
            ["Current HEAD SHA",head],["Tracked/staged changes",len([r for r in change_rows if r[1] != "??"])],
            ["Untracked files",len(untracked)],["Changed files total",len(change_rows)],
            ["Critical-priority path matches",sum(1 for r in change_rows if r[4]=="CRITICAL")],
            ["High-priority path matches",sum(1 for r in change_rows if r[4]=="HIGH")],
            ["Diff lines captured",min(len(diff_lines),args.max_diff_lines)],["Diff truncated",diff_truncated],
            ["CodeQL custom queries run",False],["Interpretation note","Path-based priority is a triage heuristic, not a defect verdict."],
            ["Untracked note","Untracked file contents are included in diff context when <=2 MB and readable as text."],
        ]
        (out/f"git_diff_{stamp}.patch").write_text(diff,encoding="utf-8")
        # Write custom query templates outside the application repository.
        query_dir=out/"custom_queries"
        query_dir.mkdir(exist_ok=True)
        query_paths=[]
        for name,content in QUERY_TEMPLATES.items():
            qp=query_dir/name; qp.write_text(content,encoding="utf-8"); query_paths.append(qp)
        custom_rows=[]
        sarif_rows=[]
        if args.run_codeql:
            if not args.codeql.is_file(): log.append(["CodeQL","ERROR",f"CLI not found: {args.codeql}"])
            elif not args.database.exists(): log.append(["CodeQL","ERROR",f"Database not found: {args.database}"])
            else:
                for qp in query_paths:
                    sarif=out/f"{qp.stem}_{stamp}.sarif"
                    cmd=[str(args.codeql),"database","analyze",str(args.database),str(qp),"--format=sarif-latest",f"--output={sarif}","--threads=0"]
                    p=subprocess.run(cmd,capture_output=True,text=True)
                    log.append(["CodeQL query",qp.name,f"exit={p.returncode}; {p.stderr[-1500:]}" if p.stderr else f"exit={p.returncode}"])
                    custom_rows.append([qp.name,"COMPILED/RAN" if p.returncode==0 else "FAILED",str(sarif),p.returncode,p.stderr[-3000:]])
                    for r in parse_sarif(sarif): sarif_rows.append([qp.name,*r])
        else:
            for qp in query_paths: custom_rows.append([qp.name,"TEMPLATE ONLY — not executed",str(qp),"", "Use --run-codeql to test against the existing database; identifiers may need adaptation."])
            log.append(["CodeQL","NOT RUN","Pass --run-codeql to execute custom queries."])
        wb=Workbook()
        write_sheet(wb.active,["Metric","Value"],summary); wb.active.title="Executive Summary"
        write_sheet(wb.create_sheet("Changed Files"),["Working Status","Diff Status","Path","Previous Path","Priority","Reason"],change_rows)
        write_sheet(wb.create_sheet("Risk Summary"),["Priority","File Count"],[[k,sum(1 for r in change_rows if r[4]==k)] for k in ("CRITICAL","HIGH","MEDIUM","NORMAL")])
        write_sheet(wb.create_sheet("Custom Queries"),["Query File","Status","Path/SARIF","Exit Code","Notes"],custom_rows)
        write_sheet(wb.create_sheet("Custom Query Findings"),["Query File","Rule ID","Rule Description","Level","File URI","Line","Message"],sarif_rows)
        write_sheet(wb.create_sheet("Execution Log"),["Stage","Status","Details"],log)
        if args.include_diff_text:
            write_sheet(wb.create_sheet("Diff Details"),["Line","Diff Text"],[[i+1,x] for i,x in enumerate(diff_lines[:args.max_diff_lines])])
        xlsx=out/f"mythri_change_audit_{stamp}.xlsx"; wb.save(xlsx)
        print(f"Repository: {repo}")
        print(f"Baseline: {base}\nHEAD: {head}")
        print(f"Changed files: {len(change_rows)} (untracked: {len(untracked)})")
        print(f"Workbook: {xlsx}\nPatch: {out/f'git_diff_{stamp}.patch'}\nCustom queries: {query_dir}")
        if args.run_codeql: print(f"Custom query findings: {len(sarif_rows)}; inspect query status and SARIF before interpreting results.")
    except Exception as exc:
        sys.exit(f"Audit failed: {exc}")

if __name__=="__main__": main()
