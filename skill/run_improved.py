#!/usr/bin/env python3
"""Run Improved Strategy vs all 10 test cases, then compare with previous results."""
import json, sys, os
sys.path.insert(0, '/tmp/MRAgent-ab-test')
from improved_strategy import ImprovedStrategy, judge_answer
from test_data import TEST_CASES

result_file = "/tmp/MRAgent-ab-test/improved_results.json"
results = {}

improved = ImprovedStrategy()

for category, cases in TEST_CASES.items():
    print(f"\n--- {category} ({len(cases)} cases) ---", flush=True)
    for case in cases:
        cid = case["id"]
        is_cross = category == "cross_session"
        conv = case.get("conversations", case.get("conversation", []))

        print(f"\n  [{cid}] {case['name']}", flush=True)
        print(f"  Q: {case['question'][:50]}", flush=True)

        r = improved.answer_cross_session(case["question"], conv) if is_cross else \
            improved.answer(case["question"], conv)
        j = judge_answer(r["answer"], case["answer"])

        results[cid] = {
            "case": {k: v for k, v in case.items() if k not in ("conversation", "conversations")},
            "improved": {**r, "judge": j},
        }

        with open(result_file, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        si = {"CORRECT": "✓", "PARTIAL": "~", "WRONG": "✗"}
        print(f"  Improved [{si.get(j['score'],'?')}] [{j['score']:>7}] {r['answer'][:60]}", flush=True)
        print(f"  entities={r.get('query_entities',[])}", flush=True)
        print(f"  s1={r.get('stage1_turns',0)} s2={r.get('stage2_new_turns',0)} evidence={r.get('total_evidence',0)}", flush=True)
        print(f"  time={r['elapsed_s']}s calls={r['calls']}", flush=True)
        print(f"  GT: {case['answer'][:60]}", flush=True)

# Load previous results for comparison
prev_file = "/tmp/MRAgent-ab-test/results_progress.json"
prev = {}
if os.path.exists(prev_file):
    with open(prev_file) as f:
        prev = json.load(f)

# ===== Report =====
total_cases = sum(len(v) for v in TEST_CASES.values())
scores = {"improved": {"c":0,"p":0,"w":0,"time":0,"calls":0},
          "mragent": {"c":0,"p":0,"w":0,"time":0,"calls":0},
          "flatrag": {"c":0,"p":0,"w":0,"time":0,"calls":0}}

for cat, cases in TEST_CASES.items():
    for case in cases:
        cid = case["id"]
        if cid in results:
            r = results[cid]["improved"]
            sc = r["judge"]["score"]
            scores["improved"][{"CORRECT":"c","PARTIAL":"p","WRONG":"w"}.get(sc,"w")] += 1
            scores["improved"]["time"] += r["elapsed_s"]
            scores["improved"]["calls"] += r["calls"]
        if cid in prev:
            for key, sk in [("mragent","mragent"),("flatrag","flatrag")]:
                r = prev[cid].get(sk,{})
                j = r.get("judge",{})
                sc = j.get("score","WRONG")
                scores[key][{"CORRECT":"c","PARTIAL":"p","WRONG":"w"}.get(sc,"w")] += 1
                scores[key]["time"] += r.get("elapsed_s",0)
                scores[key]["calls"] += r.get("calls",0)

print("\n\n" + "=" * 80)
print("FINAL COMPARISON REPORT")
print("=" * 80)

print(f"\n{'Metric':<35} {'Improved':<18} {'MRAgent':<18} {'Flat RAG':<18}")
print(f"{'─'*90}")
print(f"{'Total cases':<35} {total_cases:<18} {total_cases:<18} {total_cases:<18}")

for key, label in [("c","Correct"),("p","Partial"),("w","Wrong")]:
    print(f"{'  '+label:<35} {scores['improved'][key]:<18} {scores['mragent'][key]:<18} {scores['flatrag'][key]:<18}")

cr = {k: scores[k]['c']/total_cases*100 for k in ['improved','mragent','flatrag']}
cpr = {k: (scores[k]['c']+scores[k]['p'])/total_cases*100 for k in ['improved','mragent','flatrag']}
print(f"{'Correct %':<35} {cr['improved']:.0f}%{'':<14} {cr['mragent']:.0f}%{'':<14} {cr['flatrag']:.0f}%")
print(f"{'Correct+Partial %':<35} {cpr['improved']:.0f}%{'':<14} {cpr['mragent']:.0f}%{'':<14} {cpr['flatrag']:.0f}%")

print(f"\n{'Avg time/case (s)':<35} {scores['improved']['time']/total_cases:.1f}{'':<16} "
      f"{scores['mragent']['time']/total_cases:.1f}{'':<16} {scores['flatrag']['time']/total_cases:.1f}")
print(f"{'Avg LLM calls/case':<35} {scores['improved']['calls']/total_cases:.1f}{'':<16} "
      f"{scores['mragent']['calls']/total_cases:.1f}{'':<16} {scores['flatrag']['calls']/total_cases:.1f}")

# Per-case comparison table
print(f"\n\n{'─'*80}")
print("Per-Case Comparison")
print(f"{'─'*80}")
print(f"{'Case':<8} {'Question':<30} {'Improved':>10} {'MRAgent':>10} {'FlatRAG':>10} {'Time':>6}")
print(f"{'─'*80}")
for cat, cases in TEST_CASES.items():
    for case in cases:
        cid = case["id"]
        q_short = case["question"][:28]
        i_s = results.get(cid,{}).get("improved",{}).get("judge",{}).get("score","—")
        m_s = prev.get(cid,{}).get("mragent",{}).get("judge",{}).get("score","—")
        f_s = prev.get(cid,{}).get("flatrag",{}).get("judge",{}).get("score","—")
        i_t = results.get(cid,{}).get("improved",{}).get("elapsed_s","—")
        i_s_short = {"CORRECT":"✓   ","PARTIAL":"~   ","WRONG":"✗   ","—":"—   "}.get(i_s,i_s)
        m_s_short = {"CORRECT":"✓   ","PARTIAL":"~   ","WRONG":"✗   ","—":"—   "}.get(m_s,m_s)
        f_s_short = {"CORRECT":"✓   ","PARTIAL":"~   ","WRONG":"✗   ","—":"—   "}.get(f_s,f_s)
        print(f"{cid:<8} {q_short:<30} {i_s_short:<10} {m_s_short:<10} {f_s_short:<10} {str(i_t)+'s':>6}")
