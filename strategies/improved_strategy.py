#!/usr/bin/env python3
"""
Improved Strategy: Entity-Indexed + Two-Stage Retrieval

Plan 1: Two-stage retrieval — first pass finds results, extracts new keywords, second pass fills gaps
Plan 2: Entity indexing — store entities at index time, query by entity first to narrow search space

Both use jieba for fast Chinese entity extraction (no LLM at query time).
"""
import json
import re
import time
import urllib.request
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import jieba
import jieba.posseg as pseg

OLLAMA_URL = "http://localhost:11434/api"
MODEL = "gemma4:12b-it-q4_K_M"

# Chinese stopwords
STOPWORDS = set("的了在是我有和就不人都一个上也很到说要去你会着没看好自己这他她它她们他们它们那哪些什么怎么为什么哪个谁多少如何何时何地因为所以但是然而虽然尽管如果那么就是还是只是不是没有可以可能应该能够将会已经正在被把对从向跟与及以及或或者还是要么既又一边不但而且不过此外另外总之例如比如比方说这样那样这些那些这里那里这时那时".strip())

# Common stopword patterns
EN_STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does', 'did',
                'has', 'have', 'had', 'will', 'would', 'could', 'should', 'may',
                'might', 'can', 'what', 'why', 'how', 'when', 'where', 'who',
                'which', 'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with',
                'from', 'as', 'into', 'this', 'that', 'these', 'those', 'be',
                'been', 'being', 'not', 'no', 'or', 'and', 'but', 'if', 'so',
                'than', 'too', 'very', 'just', 'about', 'up', 'out', 'off', 'over'}


def llm_complete(prompt: str, system: str = "", max_tokens: int = 512) -> str:
    import time as _time
    _time.sleep(1.0)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    data = json.dumps({
        "model": MODEL, "messages": messages,
        "temperature": 0.0, "max_tokens": max_tokens, "stream": False,
    }).encode()
    # Fresh connection each time — use opener with no proxy
    import urllib.request as _ur
    opener = _ur.build_opener(_ur.ProxyHandler({}))
    json_bytes = data  # data is already encoded
    req = _ur.Request(f"{OLLAMA_URL}/chat", data=json_bytes,
                      headers={"Content-Type": "application/json"})
    try:
        resp = opener.open(req, timeout=120)
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
# Entity Indexer — uses jieba (no LLM at query time)
# =========================================================
class EntityIndex:
    """
    Plan 2 implementation:
    At index time, extract entities from each turn.
    Build: entity_name → set of turn indices
           entity_name → set of semantic tags
    """

    def __init__(self):
        self.entity_to_turns: Dict[str, Set[int]] = defaultdict(set)
        self.turn_to_entities: Dict[int, List[str]] = defaultdict(list)
        self.entity_to_tags: Dict[str, Set[str]] = defaultdict(set)
        self.knowledge_base: Dict[str, List[str]] = defaultdict(list)  # entity → facts
        self.all_turns: List[str] = []

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities with their POS tags.
        Returns: [(entity, pos_tag), ...]
        """
        words = pseg.cut(text)
        entities = []
        seen = set()

        for word, flag in words:
            word = word.strip()
            if len(word) < 2 or word in EN_STOPWORDS or word in STOPWORDS:
                continue
            key = word.lower()
            if key in seen:
                continue
            # English names
            if flag == 'eng' and word[0].isupper():
                seen.add(key)
                entities.append((word, 'en_name'))
                continue
            # Chinese named entities
            if flag.startswith(('nr', 'ns', 'nt', 'nz')):
                seen.add(key)
                entities.append((word, flag))
                continue
            # Content words
            if flag.startswith(('n', 'v', 'a', 'l', 'vn', 'an', 'vd', 'ad', 'zg', 'j')):
                seen.add(key)
                entities.append((word, flag))
                continue

        # Also extract English-capitalized names
        for m in re.finditer(r'\b[A-Z][a-z]+\b', text):
            entities.append((m.group(), 'en_name'))

        # Also extract date patterns
        for m in re.finditer(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', text):
            entities.append((m.group(), 'date'))
        for m in re.finditer(r'\d{1,2}月\d{1,2}日', text):
            entities.append((m.group(), 'date'))

        # Remove duplicates preserving order
        seen = set()
        result = []
        for e, f in entities:
            key = e.lower()
            if key not in seen:
                seen.add(key)
                result.append((e, f))
        return result

    def build(self, conversation: List[str]):
        """Build entity index from conversation turns."""
        self.all_turns = list(conversation)

        for idx, turn in enumerate(conversation):
            entities = self.extract_entities(turn)
            self.turn_to_entities[idx] = [e for e, _ in entities]

            # Assign semantic tags based on POS
            for entity, flag in entities:
                e_lower = entity.lower()
                self.entity_to_turns[e_lower].add(idx)

                # Map POS to semantic tags
                if flag.startswith('nr'):
                    tag = 'person'
                elif flag.startswith('ns'):
                    tag = 'place'
                elif flag.startswith('nt'):
                    tag = 'organization'
                elif flag.startswith('nz') or flag == 'en_name':
                    tag = 'proper_noun'
                elif flag.startswith('v'):
                    tag = 'action'
                elif flag.startswith('a'):
                    tag = 'attribute'
                elif flag == 'date':
                    tag = 'time'
                else:
                    tag = 'concept'

                self.entity_to_tags[e_lower].add(tag)
                self.knowledge_base[e_lower].append(f"T{idx}: {turn}")

    def build_cross_session(self, conversations: List[List[str]]):
        """Build index from multi-session conversations."""
        flat = []
        for s_idx, session in enumerate(conversations):
            for t_idx, turn in enumerate(session):
                flat.append(f"[Session {s_idx}] {turn}")
        self.build(flat)

    def lookup(self, entities: List[str]) -> Set[int]:
        """Find turns containing any of the given entities."""
        matching_turns = set()
        for ent in entities:
            e_lower = ent.lower()
            for e_stored, turns in self.entity_to_turns.items():
                if e_lower in e_stored or e_stored in e_lower:
                    matching_turns.update(turns)
        return matching_turns

    def expand_via_tags(self, entities: List[str]) -> List[str]:
        """Find related entities sharing tags."""
        related = set()
        for ent in entities:
            e_lower = ent.lower()
            tags = set()
            for e_stored, t in self.entity_to_tags.items():
                if e_lower in e_stored or e_stored in e_lower:
                    tags.update(t)
            # Find other entities with same tags
            for tag in tags:
                for e_stored, t in self.entity_to_tags.items():
                    if tag in t and e_stored not in [e.lower() for e in entities]:
                        related.add(e_stored)
        return list(related)

    def stage2_search(self, first_pass_entities: List[str], excluded_turns: Set[int]) -> Set[int]:
        """
        Plan 1 implementation:
        From first pass results, extract new entities, search remaining turns.
        """
        new_turns = set()
        for ent in first_pass_entities:
            e_lower = ent.lower()
            for e_stored, turns in self.entity_to_turns.items():
                # Different entity, but shares some characters → related
                if e_lower != e_stored and (len(set(e_lower) & set(e_stored)) > 2 or
                    any(kw in e_stored or e_stored in kw for kw in e_lower.split())):
                    new_turns.update(turns - excluded_turns)
        return new_turns

    def keyword_search(self, keywords: List[str], excluded_turns: Set[int] = None) -> Set[int]:
        """Fallback: search each turn for keyword matches."""
        if excluded_turns is None:
            excluded_turns = set()
        results = set()
        for kw in keywords:
            kw_lower = kw.lower()
            for idx, turn in enumerate(self.all_turns):
                if idx in excluded_turns:
                    continue
                if kw_lower in turn.lower():
                    results.add(idx)
        return results


# =========================================================
# Improved Strategy: Entity Index + Two-Stage Retrieval
# =========================================================
class ImprovedStrategy:
    """
    Combines Entity Indexing (Plan 2) + Two-Stage Retrieval (Plan 1).

    Storage: build entity index from conversation (done once)
    Query:
      Stage 1: extract entities from question → entity lookup → candidate turns
               + keyword fallback within those turns
      Stage 2: from stage 1 result text, extract new entities → search remaining turns
      Final: merge evidence, answer with LLM
    """

    def __init__(self):
        self.index = None
        self.calls = 0

    def _extract_query_entities(self, question: str) -> List[str]:
        """Extract entities from a question using jieba."""
        entities = set()
        words = pseg.cut(question)
        for word, flag in words:
            word = word.strip()
            if len(word) < 2 or word in EN_STOPWORDS or word in STOPWORDS:
                continue
            key = word.lower()
            if flag == 'eng' and word[0].isupper():
                entities.add(word)
            elif flag.startswith(('nr', 'ns', 'nt', 'nz', 'n', 'v', 'a', 'l', 'vn', 'an', 'vd', 'ad', 'zg', 'j')):
                entities.add(word)

        # English capitalized
        for m in re.finditer(r'\b[A-Z][a-z]+\b', question):
            entities.add(m.group())

        return list(entities)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (for second stage)."""
        words = pseg.cut(text)
        keywords = []
        for word, flag in words:
            word = word.strip()
            if len(word) < 2:
                continue
            if word in EN_STOPWORDS or word in STOPWORDS:
                continue
            if flag.startswith(('nr', 'ns', 'nt', 'nz', 'n', 'v')):
                keywords.append(word)
        return list(set(keywords))

    def answer(self, question: str, conversation: List[str]) -> Dict[str, Any]:
        start = time.time()
        self.calls = 0

        # Build index (simulating "storage time" — once per conversation)
        self.index = EntityIndex()
        self.index.build(conversation)

        # === Stage 1: Entity Lookup ===
        query_entities = self._extract_query_entities(question)
        stage1_turns = set()

        if query_entities:
            stage1_turns = self.index.lookup(query_entities)
            # Also expand via tags (related entities)
            related = self.index.expand_via_tags(query_entities)
            if related:
                stage1_turns.update(self.index.lookup(related))

        # Keyword fallback for turns not matched by entity
        keywords = self._extract_keywords(question)
        if keywords:
            kw_turns = self.index.keyword_search(keywords, excluded_turns=stage1_turns)
            stage1_turns.update(kw_turns)

        # If still no matches, use top keyword-matched turns
        if not stage1_turns and keywords:
            # Score all turns by keyword overlap
            scored = []
            for idx, turn in enumerate(conversation):
                kws = [kw.lower() for kw in keywords]
                score = sum(1 for kw in kws if kw in turn.lower())
                if score > 0:
                    scored.append((score, idx))
            scored.sort(key=lambda x: -x[0])
            stage1_turns = {idx for _, idx in scored[:5]}

        # === Stage 2: Two-Stage Expansion ===
        # From stage 1 results, extract new keywords/entities
        stage2_turns = set()
        if stage1_turns:
            stage1_text = " ".join(conversation[idx] for idx in stage1_turns if idx < len(conversation))
            new_keywords = self._extract_keywords(stage1_text)

            # Remove keywords already used
            used = set(k.lower() for k in keywords)
            unused_kw = [k for k in new_keywords if k.lower() not in used]

            if unused_kw:
                stage2_turns = self.index.keyword_search(unused_kw, excluded_turns=stage1_turns)

            # Entity-based expansion
            new_entities_from_stage1 = self._extract_query_entities(stage1_text)
            new_entity_turns = self.index.lookup(new_entities_from_stage1)
            stage2_turns.update(new_entity_turns - stage1_turns)

        # Merge evidence — limit to top 4 turns to avoid 400 error
        all_turns = stage1_turns | stage2_turns
        sorted_turns = sorted(all_turns) if all_turns else []
        # Prioritize stage1 (entity-matched) turns
        stage1_list = sorted(stage1_turns) if stage1_turns else []
        stage2_list = [t for t in sorted(stage2_turns) if t not in stage1_turns]
        evidence_turns = (stage1_list + stage2_list)[:4]

        # === Answer with LLM ===
        self.calls += 1
        if evidence_turns:
            evidence = [f"[T{idx}] {conversation[idx]}" for idx in evidence_turns if idx < len(conversation)]
            evidence_str = "\n".join(evidence)
            if len(evidence_str) > 600:
                evidence_str = evidence_str[:600] + "...[truncated]"
            prompt = f"""根据以下对话回答问题。用具体事实，简洁准确。

相关对话：
{evidence_str}

问题：{question}"""
        else:
            # Fallback: use truncated conversation (top 4 turns)
            fallback = chr(10).join(conversation[:4])
            if len(fallback) > 600:
                fallback = fallback[:600] + "...[truncated]"
            prompt = f"""根据以下对话回答问题。用具体事实，简洁准确。

对话：
{fallback}

问题：{question}"""

        answer = llm_complete(prompt)
        elapsed = time.time() - start

        return {
            "strategy": "Improved (EntityIdx + TwoStage)",
            "answer": answer,
            "calls": self.calls,
            "elapsed_s": round(elapsed, 1),
            "query_entities": query_entities,
            "stage1_turns": len(stage1_turns),
            "stage2_new_turns": len(stage2_turns),
            "total_evidence": len(evidence_turns),
        }

    def answer_cross_session(self, question: str, conversations: List[List[str]]) -> Dict[str, Any]:
        start = time.time()
        self.calls = 0

        flat = []
        for s_idx, session in enumerate(conversations):
            for t_idx, turn in enumerate(session):
                flat.append(f"[S{s_idx}] {turn}")

        self.index = EntityIndex()
        self.index.build(flat)

        query_entities = self._extract_query_entities(question)
        stage1_turns = set()

        if query_entities:
            stage1_turns = self.index.lookup(query_entities)
            related = self.index.expand_via_tags(query_entities)
            if related:
                stage1_turns.update(self.index.lookup(related))

        keywords = self._extract_keywords(question)
        if keywords:
            kw_turns = self.index.keyword_search(keywords, excluded_turns=stage1_turns)
            stage1_turns.update(kw_turns)

        if not stage1_turns and keywords:
            scored = []
            for idx, turn in enumerate(flat):
                score = sum(1 for kw in keywords if kw.lower() in turn.lower())
                if score > 0:
                    scored.append((score, idx))
            scored.sort(key=lambda x: -x[0])
            stage1_turns = {idx for _, idx in scored[:5]}

        # Stage 2
        stage2_turns = set()
        if stage1_turns:
            stage1_text = " ".join(flat[idx] for idx in stage1_turns if idx < len(flat))
            new_keywords = self._extract_keywords(stage1_text)
            used = set(k.lower() for k in keywords)
            unused_kw = [k for k in new_keywords if k.lower() not in used]

            if unused_kw:
                stage2_turns = self.index.keyword_search(unused_kw, excluded_turns=stage1_turns)
            new_entities = self._extract_query_entities(stage1_text)
            new_entity_turns = self.index.lookup(new_entities)
            stage2_turns.update(new_entity_turns - stage1_turns)
        # Merge evidence — limit to top 4
        all_turns = stage1_turns | stage2_turns
        stage1_list = sorted(stage1_turns) if stage1_turns else []
        stage2_list = [t for t in sorted(stage2_turns) if t not in stage1_turns]
        evidence_turns = (stage1_list + stage2_list)[:4]

        self.calls += 1
        if evidence_turns:
            evidence = [flat[idx] for idx in evidence_turns if idx < len(flat)]
            evidence_str = "\n".join(evidence)
            if len(evidence_str) > 600:
                evidence_str = evidence_str[:600] + "...[truncated]"
            prompt = f"""根据以下跨会话对话回答问题。需要合并多个会话的信息。

相关对话：
{evidence_str}

问题：{question}"""
        else:
            fallback = "\n".join(flat[:4])
            if len(fallback) > 600:
                fallback = fallback[:600] + "...[truncated]"
            prompt = f"""根据对话回答问题。需要合并多个会话的信息。

对话：
{fallback}

问题：{question}"""

        answer = llm_complete(prompt)
        elapsed = time.time() - start

        return {
            "strategy": "Improved (EntityIdx + TwoStage)",
            "answer": answer,
            "calls": self.calls,
            "elapsed_s": round(elapsed, 1),
            "query_entities": query_entities,
            "stage1_turns": len(stage1_turns),
            "stage2_new_turns": len(stage2_turns),
            "total_evidence": len(evidence_turns),
        }


# =========================================================
# Judge
# =========================================================
def judge_answer(prediction: str, ground_truth: str) -> Dict[str, Any]:
    prompt = f"""Evaluate if the prediction correctly answers the question.

Ground truth: {ground_truth}
Prediction: {prediction}

Score: CORRECT (contains all key facts), PARTIAL (some info but missing details), WRONG (doesn't match)

Return JSON: {{"score": "CORRECT|PARTIAL|WRONG", "reason": "brief explanation"}}"""
    result = llm_complete(prompt)
    parsed = extract_json(result)
    return {
        "score": parsed.get("score", "WRONG"),
        "reason": parsed.get("reason", ""),
    }
