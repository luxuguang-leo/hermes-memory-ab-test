#!/usr/bin/env python3
"""
MRAgent A/B Test v3 — stable version (based on v1 smoke test which worked).
Only difference: both strategies use LLM-only retrieval (no embeddings).
"""
import json
import re
import time
import urllib.request
from typing import List, Dict, Any
from collections import defaultdict

OLLAMA_URL = "http://localhost:11434/api"
MODEL = "gemma4:12b-it-q4_K_M"

from test_data import TEST_CASES


def llm_complete(prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int = 512) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    import time as _time
    _time.sleep(1.0)  # rate limit to avoid tunnel overload
    data = json.dumps({
        "model": MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/chat", data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        return json.loads(resp.read())["message"]["content"].strip()
    except Exception as e:
        return f"[OLLAMA_ERROR: {e}]"


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    return {}


# =========================================================
# Strategy A: MRAgent-style (as in v1 smoke test)
# =========================================================
class MRAgentStrategy:
    def __init__(self):
        self.calls = 0

    def answer(self, question: str, conversation: List[str]) -> Dict[str, Any]:
        start = time.time()
        self.calls = 0

        # Step 1: Extract cues
        self.calls += 1
        prompt_cues = f"""Extract key entities, keywords, and any time references from this question.
Return as JSON: {{"cues": ["entity1", "entity2", ...], "time": "time_ref_or_null", "question_type": "fact|temporal|relation|causal"}}

Question: {question}"""
        r1 = llm_complete(prompt_cues)
        parsed = extract_json(r1)
        cues = parsed.get("cues", [])
        # Fallback
        if not cues:
            cues = re.findall(r'[A-Z][a-z]+|\b\w{3,}\b', question)[:5]

        # Step 2: Build entity graph from conversation
        self.calls += 1
        conv_text = "\n".join(f"T{i}: {t}" for i, t in enumerate(conversation))
        prompt_graph = f"""For each turn in this conversation, extract:
1. Named entities (people, places, things, concepts)
2. Semantic tags for each entity

Return JSON:
{{"entities": {{"entity_name": ["tag1", "tag2"]}}, "turns": [{{"idx": 0, "entities_found": ["entity1"], "relevant": true/false}}]}}

Conversation:
{conv_text}"""
        r2 = llm_complete(prompt_graph, system="You are an entity extraction system.")
        graph = extract_json(r2)

        # Step 3: Find matching turns for each cue
        cue_turns = set()
        cue_entities = set()
        entities = graph.get("entities", {})
        turns_info = graph.get("turns", [])

        for ent_name in entities:
            ent_lower = ent_name.lower()
            for cue in cues:
                if cue.lower() in ent_lower or ent_lower in cue.lower():
                    cue_entities.add(ent_name)
                    # Find turns containing this entity
                    for ti in turns_info:
                        if ti.get("idx") is not None:
                            for ef in ti.get("entities_found", []):
                                if ent_name.lower() in ef.lower():
                                    cue_turns.add(ti["idx"])

        # Step 3b: Expand via tags
        related_entities = set()
        for ent in cue_entities:
            tags = entities.get(ent, [])
            for tag in tags:
                for other_ent, other_tags in entities.items():
                    if other_ent != ent and tag in other_tags:
                        related_entities.add(other_ent)
                        for ti in turns_info:
                            for ef in ti.get("entities_found", []):
                                if other_ent.lower() in ef.lower():
                                    cue_turns.add(ti["idx"])

        # Step 4: Collect evidence and answer
        self.calls += 1
        evidence = []
        for ti in sorted(cue_turns):
            if ti < len(conversation):
                evidence.append(f"[T{ti}] {conversation[ti]}")

        if evidence:
            evidence_str = "\n".join(evidence)
            answer_prompt = f"""Based on the conversation evidence, answer the question concisely with specific facts.

Evidence:
{evidence_str}

Question: {question}"""
        else:
            answer_prompt = f"""Based on the conversation, answer the question concisely with specific facts.

Conversation:
{conv_text}

Question: {question}"""

        answer = llm_complete(answer_prompt)

        elapsed = time.time() - start
        return {
            "strategy": "MRAgent (Graph Active)",
            "answer": answer,
            "calls": self.calls,
            "elapsed_s": round(elapsed, 1),
            "entities": len(cue_entities),
            "evidence_turns": len(evidence),
        }

    def answer_cross_session(self, question: str, conversations: List[List[str]]) -> Dict[str, Any]:
        start = time.time()
        self.calls = 0

        flat = []
        for s_idx, session in enumerate(conversations):
            for t_idx, turn in enumerate(session):
                flat.append(f"[Session {s_idx}] {turn}")

        conv_text = "\n".join(flat)

        # Step 1: Extract cues
        self.calls += 1
        prompt_cues = f"""Extract key entities from this question.
Return JSON: {{"cues": ["entity1", "entity2", ...], "time": null}}
Question: {question}"""
        cues = extract_json(llm_complete(prompt_cues)).get("cues", [])
        if not cues:
            cues = re.findall(r'[A-Z][a-z]+|\b\w{3,}\b', question)[:5]

        # Step 2: Build graph
        self.calls += 1
        prompt_graph = f"""For each turn in this multi-session conversation, extract named entities.
Return JSON: {{"entities": {{"entity_name": ["tag1", "tag2"]}}, "turns": [{{"idx": 0, "entities_found": ["entity1"]}}]}}

Conversation:
{conv_text}"""
        graph = extract_json(llm_complete(prompt_graph))

        # Step 3: Find matching turns
        cue_turns = set()
        cue_entities = set()
        entities = graph.get("entities", {})
        turns_info = graph.get("turns", [])

        for ent_name in entities:
            for cue in cues:
                if cue.lower() in ent_name.lower() or ent_name.lower() in cue.lower():
                    cue_entities.add(ent_name)
                    for ti in turns_info:
                        if ti.get("idx") is not None:
                            for ef in ti.get("entities_found", []):
                                if ent_name.lower() in ef.lower():
                                    cue_turns.add(ti["idx"])

        # Step 4: Collect and answer
        self.calls += 1
        evidence = [flat[ti] for ti in sorted(cue_turns) if ti < len(flat)]

        if evidence:
            evidence_str = "\n".join(evidence)
            answer_prompt = f"""Based on the multi-session conversation evidence, answer the question.
Combine facts across sessions.

Evidence:
{evidence_str}

Question: {question}"""
        else:
            answer_prompt = f"""Based on the multi-session conversation, answer the question.
Combine facts across sessions.

Conversation:
{conv_text}

Question: {question}"""

        answer = llm_complete(answer_prompt)

        elapsed = time.time() - start
        return {
            "strategy": "MRAgent (Graph Active)",
            "answer": answer,
            "calls": self.calls,
            "elapsed_s": round(elapsed, 1),
            "entities": len(cue_entities),
            "evidence_turns": len(evidence),
        }


# =========================================================
# Strategy B: Flat RAG
# =========================================================
class FlatRAGStrategy:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.calls = 0

    def _pick_relevant(self, question: str, chunks: List[str]) -> List[str]:
        """Keyword matching + LLM relevance filter."""
        self.calls += 1
        q_words = set(re.findall(r'\b(\w+)\b', question.lower()))
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does',
                     'did', 'has', 'have', 'had', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'what', 'why', 'how',
                     'when', 'where', 'who', 'which', 'of', 'in', 'to', 'for',
                     'on', 'at', 'by', 'with', 'from', 'as', 'into', 'this',
                     'that', 'these', 'those', 'be', 'been', 'being', 'not',
                     'no', 'or', 'and', 'but', 'if', 'so', 'than', 'too',
                     'very', 'just', 'about', 'up', 'out', 'off', 'over'}
        q_words -= stopwords

        scored = []
        for ch in chunks:
            ch_words = set(re.findall(r'\b(\w+)\b', ch.lower()))
            overlap = len(q_words & ch_words)
            scored.append((overlap, ch))

        scored.sort(key=lambda x: -x[0])
        return [ch for _, ch in scored[:self.top_k]]

    def answer(self, question: str, conversation: List[str]) -> Dict[str, Any]:
        start = time.time()
        self.calls = 0

        chunks = [f"T{i}: {t}" for i, t in enumerate(conversation)]
        selected = self._pick_relevant(question, chunks)

        self.calls += 1
        context = "\n".join(selected)
        prompt = f"""Answer the question based on the conversation context. Be concise.

Context:
{context}

Question: {question}"""
        answer = llm_complete(prompt)

        elapsed = time.time() - start
        return {
            "strategy": "Flat RAG",
            "answer": answer,
            "calls": self.calls,
            "elapsed_s": round(elapsed, 1),
            "selected_chunks": len(selected),
        }

    def answer_cross_session(self, question: str, conversations: List[List[str]]) -> Dict[str, Any]:
        start = time.time()
        self.calls = 0

        chunks = []
        for s_idx, session in enumerate(conversations):
            for t_idx, turn in enumerate(session):
                chunks.append(f"S{s_idx}T{t_idx}: {turn}")

        selected = self._pick_relevant(question, chunks)

        self.calls += 1
        context = "\n".join(selected)
        prompt = f"""Answer the question based on the multi-session conversation. Combine facts across sessions.

Context:
{context}

Question: {question}"""
        answer = llm_complete(prompt)

        elapsed = time.time() - start
        return {
            "strategy": "Flat RAG",
            "answer": answer,
            "calls": self.calls,
            "elapsed_s": round(elapsed, 1),
            "selected_chunks": len(selected),
        }


# =========================================================
# Judge
# =========================================================
def judge_answer(prediction: str, ground_truth: str) -> Dict[str, Any]:
    prompt = f"""Grade whether the prediction correctly answers the question.
Compare with the ground truth answer.

Ground truth: {ground_truth}
Prediction: {prediction}

Score:
- CORRECT: Contains the key facts from ground truth
- PARTIAL: Has some relevant info but misses key details
- WRONG: Doesn't match the ground truth

Return JSON: {{"score": "CORRECT|PARTIAL|WRONG", "reason": "brief explanation"}}"""
    result = llm_complete(prompt)
    parsed = extract_json(result)
    return {
        "score": parsed.get("score", "WRONG"),
        "reason": parsed.get("reason", ""),
    }


# =========================================================
# Main
# =========================================================
def run_test():
    print("=" * 70, flush=True)
    print("MRAgent A/B Test v3 — Active Graph vs Flat RAG", flush=True)
    print(f"Model: {MODEL}", flush=True)
    print("=" * 70, flush=True)

    mragent = MRAgentStrategy()
    flat_rag = FlatRAGStrategy()
    all_results = {}
    totals = {"m": {"c": 0, "p": 0, "w": 0, "calls": 0, "time": 0},
              "f": {"c": 0, "p": 0, "w": 0, "calls": 0, "time": 0}}

    for category, cases in TEST_CASES.items():
        print(f"\n--- {category} ({len(cases)} cases) ---", flush=True)
        for case in cases:
            cid = case["id"]
            is_cross = category == "cross_session"
            conv = case.get("conversations", case.get("conversation", []))

            print(f"\n  [{cid}] {case['name']}", flush=True)
            print(f"  Q: {case['question'][:50]}", flush=True)

            r1 = mragent.answer_cross_session(case["question"], conv) if is_cross else \
                 mragent.answer(case["question"], conv)
            r2 = flat_rag.answer_cross_session(case["question"], conv) if is_cross else \
                 flat_rag.answer(case["question"], conv)

            j1 = judge_answer(r1["answer"], case["answer"])
            j2 = judge_answer(r2["answer"], case["answer"])

            all_results[cid] = {
                "case": {k: v for k, v in case.items() if k not in ("conversation","conversations")},
                "mragent": {**r1, "judge": j1},
                "flatrag": {**r2, "judge": j2},
            }

            for key, score in [("m", j1["score"]), ("f", j2["score"])]:
                if score == "CORRECT":
                    totals[key]["c"] += 1
                elif score == "PARTIAL":
                    totals[key]["p"] += 1
                else:
                    totals[key]["w"] += 1
                totals[key]["calls"] += all_results[cid][{"m":"mragent","f":"flatrag"}[key]]["calls"]
                totals[key]["time"] += all_results[cid][{"m":"mragent","f":"flatrag"}[key]]["elapsed_s"]

            si = {"CORRECT": "✓", "PARTIAL": "~", "WRONG": "✗"}
            print(f"  A[MRAgent] {si[j1['score']]} [{j1['score']:>7}] {r1['answer'][:60]}", flush=True)
            print(f"  B[FlatRAG] {si[j2['score']]} [{j2['score']:>7}] {r2['answer'][:60]}", flush=True)
            print(f"  GT: {case['answer'][:60]}", flush=True)
            print(f"  calls: A={r1['calls']}/{r1['elapsed_s']}s  B={r2['calls']}/{r2['elapsed_s']}s", flush=True)

    total_cases = sum(len(v) for v in TEST_CASES.values())

    print("\n\n" + "=" * 70, flush=True)
    print("FINAL REPORT", flush=True)
    print("=" * 70, flush=True)
    print(f"\n{'Metric':<35} {'MRAgent':<20} {'Flat RAG':<20}", flush=True)
    print(f"{'─'*75}", flush=True)
    print(f"{'Total cases':<35} {total_cases:<20} {total_cases:<20}", flush=True)
    for key, label in [("c","Correct"),("p","Partial"),("w","Wrong")]:
        print(f"{label:<35} {totals['m'][key]:<20} {totals['f'][key]:<20}", flush=True)

    cr_m = totals['m']['c']/total_cases*100
    cr_f = totals['f']['c']/total_cases*100
    cpr_m = (totals['m']['c']+totals['m']['p'])/total_cases*100
    cpr_f = (totals['f']['c']+totals['f']['p'])/total_cases*100
    print(f"{'Correct %':<35} {cr_m:.0f}%{'':<16} {cr_f:.0f}%", flush=True)
    print(f"{'Correct+Partial %':<35} {cpr_m:.0f}%{'':<16} {cpr_f:.0f}%", flush=True)
    print(f"\n{'Total LLM calls':<35} {totals['m']['calls']:<20} {totals['f']['calls']:<20}", flush=True)
    print(f"{'Avg calls/case':<35} {totals['m']['calls']/total_cases:.1f}{'':<18} {totals['f']['calls']/total_cases:.1f}", flush=True)
    print(f"{'Total time (s)':<35} {totals['m']['time']:.0f}{'':<18} {totals['f']['time']:.0f}", flush=True)
    print(f"{'Avg time/case (s)':<35} {totals['m']['time']/total_cases:.1f}{'':<18} {totals['f']['time']/total_cases:.1f}", flush=True)

    # Per-category
    print(f"\n\n{'─'*60}", flush=True)
    print("Per-Category Breakdown", flush=True)
    print(f"{'─'*60}", flush=True)
    for cat, cases in TEST_CASES.items():
        cm = {"c":0,"p":0,"w":0}
        cf = {"c":0,"p":0,"w":0}
        for c in cases:
            r = all_results.get(c["id"], {})
            if r:
                sm = r["mragent"]["judge"]["score"]
                sf = r["flatrag"]["judge"]["score"]
                cm[{"CORRECT":"c","PARTIAL":"p","WRONG":"w"}.get(sm,"w")] += 1
                cf[{"CORRECT":"c","PARTIAL":"p","WRONG":"w"}.get(sf,"w")] += 1
        n = len(cases)
        print(f"\n  {cat} ({n}):", flush=True)
        print(f"    MRAgent: {cm['c']}/{n} ✓ +{cm['p']} ~ +{cm['w']} ✗ ({cm['c']/n*100:.0f}%)", flush=True)
        print(f"    FlatRAG: {cf['c']}/{n} ✓ +{cf['p']} ~ +{cf['w']} ✗ ({cf['c']/n*100:.0f}%)", flush=True)

    with open("/tmp/MRAgent-ab-test/results.json","w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults: /tmp/MRAgent-ab-test/results.json", flush=True)


if __name__ == "__main__":
    run_test()
