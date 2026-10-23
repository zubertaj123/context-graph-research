import sys, re, dataclasses, warnings
from collections import Counter, defaultdict
sys.path.insert(0, '.')
warnings.filterwarnings("ignore")
import spacy
from paper2.data import load_locomo
from paper2 import extract as ex
from paper2.graph import norm
nlp = spacy.load("en_core_web_sm")
convs = load_locomo()
print("Observation fields:", [f.name for f in dataclasses.fields(convs[0].observations[0])])
t0 = convs[0].sessions[0].turns[0]; print("Turn fields:", [f.name for f in dataclasses.fields(t0)])
subj_kind = Counter(); examples = defaultdict(list)
rule_counts = Counter(); per_conv = {}
for conv in convs:
    names = {n.lower(): n for n in conv.speakers}
    sents = []
    for o in conv.observations:
        txt = getattr(o, "text", None) or getattr(o, "sentence", None)
        spk = getattr(o, "speaker", conv.speakers[0])
        if txt: sents.append((txt, spk))
    # existing extraction: non-hub vertices + edges under current rule
    cur_edges = set(); nonhub_deg = Counter()
    new_edges = {"A_relnoun_named": set(), "B_nonspeaker_subject": set(), "C_comention": set()}
    for txt, spk in sents:
        doc = nlp(txt)
        verb = ex._main_verb(doc)
        ns = None
        if verb is not None:
            _, ns = ex._subject(verb, names, spk)
        # classify subject
        if ns is None: kind = "none/parse-fail"
        elif ns.text.lower() in names: kind = "speaker name"
        elif ns.pos_ == "PRON": kind = "pronoun"
        elif any(c.dep_ == "poss" and (c.text.lower() in names or c.text.lower() in ex._POSS_PRON) for c in ns.children):
            kind = "speaker's X (possessive)"
        elif ns.ent_type_ in ("PERSON",): kind = "other PERSON"
        elif ns.ent_type_ in ("ORG", "GPE", "LOC", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART"): kind = f"other NE ({ns.ent_type_})"
        elif ns.pos_ in ("NOUN", "PROPN"): kind = "common noun"
        else: kind = f"other ({ns.pos_})"
        subj_kind[kind] += 1
        if len(examples[kind]) < 4: examples[kind].append(txt[:90])
        # current triples
        triples, _ = ex._extract(txt, spk, conv.speakers)
        prim = triples[0] if triples else None
        nonhubs = [(t[2], t[3]) for t in triples if t[2].lower() not in names]
        for s, p, o, ot in triples:
            cur_edges.add((norm(s), p, norm(o)))
        # Rule A: relation noun subject with appositive name -> (Name, primary pred, obj)
        if ns is not None and prim is not None and ns.pos_ in ("NOUN", "PROPN") and ns.text.lower() not in names:
            head = ns.lemma_.lower()
            if head in ex.RELATION_NOUNS:
                named = ex._named_as(ns) or ex._appos_name(ns)
                if named is not None and named.pos_ == "PROPN":
                    # the complement, with the named person as subject
                    comp = ex._complement(verb, names, named.text, verb.lemma_.lower()) if verb is not None else None
                    if comp and comp[1] and comp[1].lower() != named.text.lower():
                        new_edges["A_relnoun_named"].add((norm(named.text), comp[0], norm(comp[1])))
        # Rule B: non-speaker named-entity or common-noun subject -> (that entity, pred, obj)
        if ns is not None and verb is not None and kind.startswith(("other PERSON", "other NE", "common noun")):
            comp = ex._complement(verb, names, ns.text, verb.lemma_.lower())
            if comp and comp[1] and comp[1].lower() != ns.text.lower() and comp[1].lower() not in names:
                new_edges["B_nonspeaker_subject"].add((norm(ex._phrase(ns)), comp[0], norm(comp[1])))
        # Rule C: co-mention: primary object <-> each mentioned non-hub entity in the same sentence
        if prim is not None:
            ments = [(t[2], t[3]) for t in triples[1:] if t[1] == "mentions" and t[2].lower() not in names]
            for m, mt in ments:
                if norm(m) != norm(prim[2]):
                    new_edges["C_comention"].add((norm(prim[2]), "co_mentioned_with", norm(m)))
    # degree of non-hub vertices under current + each rule
    def nonhub_stats(edges):
        deg = Counter(); nn = 0
        for s, p, o in edges:
            sh, oh = s in names, o in names
            if not sh: deg[s] += 1
            if not oh: deg[o] += 1
            if not sh and not oh: nn += 1
        return len(deg), sum(1 for v in deg.values() if v >= 2), nn
    base = nonhub_stats(cur_edges)
    row = {"sents": len(sents), "cur_nonhub_vertices": base[0], "cur_deg2": base[1], "cur_nn_edges": base[2]}
    for k, es in new_edges.items():
        st = nonhub_stats(cur_edges | es)
        row[k] = (len(es), st[1], st[2])
    allst = nonhub_stats(cur_edges | new_edges["A_relnoun_named"] | new_edges["B_nonspeaker_subject"] | new_edges["C_comention"])
    row["ALL"] = (sum(len(e) for e in new_edges.values()), allst[1], allst[2])
    per_conv[conv.id] = row
    if conv.id == "conv-26":
        for k, es in new_edges.items():
            print(f"  conv-26 sample {k}:", sorted(es)[:6])
print("\n=== grammatical subject of the 2,541 observation sentences (spaCy nsubj) ===")
tot = sum(subj_kind.values())
for k, v in subj_kind.most_common(): print(f"  {k:28} {v:5} {v/tot:6.1%}  e.g. {examples[k][:2]}")
print("\n=== per conversation: current vs with cheap rules. tuple = (new edges, non-hub vertices with deg>=2, non-hub->non-hub edges) ===")
print(f"  {'conv':8} {'sents':>5} {'nonhubV':>7} {'deg2':>5} {'nnE':>4} | {'A relnoun+name':>16} | {'B nonspk subj':>16} | {'C co-mention':>16} | {'ALL':>16}")
for c, r in per_conv.items():
    print(f"  {c:8} {r['sents']:5} {r['cur_nonhub_vertices']:7} {r['cur_deg2']:5} {r['cur_nn_edges']:4} | {str(r['A_relnoun_named']):>16} | {str(r['B_nonspeaker_subject']):>16} | {str(r['C_comention']):>16} | {str(r['ALL']):>16}")
# raw dialogue turns: third-party entity density (what an LLM extractor could see)
print("\n=== raw dialogue turns: named entities that are not the speakers ===")
tc = Counter()
for conv in convs[:3]:
    names = {n.lower() for n in conv.speakers}
    for s in conv.sessions:
        for t in s.turns:
            txt = getattr(t, "text", "") or ""
            ents = {e.text for e in nlp(txt).ents if e.label_ in ("PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT", "WORK_OF_ART", "PRODUCT") and e.text.lower() not in names}
            tc["turns"] += 1; tc["turns_with_>=1_other_NE"] += len(ents) >= 1; tc["turns_with_>=2_other_NE"] += len(ents) >= 2
print("  (first 3 conversations)", dict(tc))
