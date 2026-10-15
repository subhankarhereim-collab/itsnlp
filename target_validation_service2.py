import json
from functools import lru_cache

from graph_builder.repurposing_scoring import relation_sign
from llm_client.llm_client import generate_json_response
from neo4j_client.neo4j_client import get_session
from services.drug_repurposing import _evidence_on, drug_names_in_graph, get_existing_files
from services.target_dashboard_service import get_all_targets
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Target Validation (Drug-Target Interaction Analysis): the "Target Validation" mode of streamlit_app.py.
# One row per target (the targets of the Target Intelligence Dashboard) and per way the drug acts on it.
#   Interaction type   the drug -> target relationships in the papers (INHIBITS -> Inhibitor),
#                      else the target's Effect of Drug, else the LLM
#   Biological effect  the target's links to pathways, combined with the interaction:
#                      the drug inhibits TLR7 + TLR7 activates type I interferon signaling
#                      -> "Reduces type I interferon signaling", else the LLM
# The knowledge graph first; the LLM fills every cell the graph cannot fill, with what is most relevant
# for the drug (an indirect effect, or the target's role when the drug has no known effect on it).
# Names: the drug's other names (e.g. HCQ) count as the drug (the name list), and targets that differ
# only in capital letters are one target.
# get_target_validation(drug) returns a plain dict, so an API can return it as it is.
# ---------------------------------------------------------------------------

DRUG_TARGET_QUERY = """
MATCH (d:Drug)-[r]->(t:Target)
WHERE toLower(d.name) IN $drugs
RETURN t.name AS target, type(r) AS type, properties(r) AS props
"""

TARGET_PATHWAY_QUERY = """
MATCH (t:Target)-[r]->(p:Pathway)
RETURN t.name AS target, p.name AS pathway, type(r) AS type, properties(r) AS props
"""

EFFECT_OF_DRUG_QUERY = """
MATCH (t:Target)
RETURN t.name AS target, t.effect_of_drug AS effect
"""

INTERACTION_TYPES = {"INHIBITS": "Inhibitor", "ACTIVATES": "Activator", "BINDS": "Binder",
                     "REGULATES": "Modulator", "ASSOCIATED_WITH": "Associated"}
EFFECT_OF_DRUG = {"inhibition": ("Inhibitor", -1), "antagonism": ("Antagonist", -1),
                  "downregulation": ("Downregulator", -1), "activation": ("Activator", 1),
                  "agonism": ("Agonist", 1), "upregulation": ("Upregulator", 1), "modulation": ("Modulator", 0)}
EFFECT_WORDS = {-1: "Reduces", 1: "Increases", 0: "Affects"}
FIELDS = ["interaction_type", "biological_effect"]
MAX_PATHWAYS = 3              # pathways shown in one cell, the ones with the most papers first
ROWS_PER_LLM_CALL = 25        # keeps each LLM answer short, so it is not cut off


def get_target_validation(drug):
    """{"drug", "rows": [{"target", "interaction_type", "biological_effect"}], "llm_error"} for the drug.
    A cell is "—" only when the LLM could not be reached (llm_error is then True)."""
    existing_files = get_existing_files()
    relations, pathways, effects = {}, {}, {}
    with get_session() as session:
        # How the drug acts on each target: the drug -> target relationships (also under its other names, e.g. HCQ)
        for record in session.run(DRUG_TARGET_QUERY, drugs=drug_names_in_graph(drug)):
            link = _evidence_on(record["type"], record["props"], existing_files)    # deleted papers are left out
            if relation_sign(record["type"]) is not None and link["sentences"]:      # "did not inhibit" is no interaction
                relations.setdefault(str(record["target"]).lower(), set()).add(record["type"])
        # Downstream: the target's links to pathways (with the number of papers, to show the best-supported first)
        for record in session.run(TARGET_PATHWAY_QUERY):
            link = _evidence_on(record["type"], record["props"], existing_files)
            sign = relation_sign(record["type"])
            if sign is not None and link["sentences"]:
                pathways.setdefault(str(record["target"]).lower(), []).append(
                    (record["pathway"], sign, len(set(link["files"]))))
        # The target's Effect of Drug from the target pipeline (inhibition, activation, agonism, ...)
        for record in session.run(EFFECT_OF_DRUG_QUERY):
            effect = str(record["effect"] or "").strip().lower()
            if effect and effect != "none":
                effects.setdefault(str(record["target"]).lower(), effect)

    rows, seen = [], set()
    for target in get_all_targets():
        key = str(target).lower()
        if not target or key in seen:            # "TLR7" and "tlr7" are one target
            continue
        seen.add(key)
        for interaction, sign in _interactions(relations.get(key, set()), effects.get(key)):
            rows.append({"target": target, "interaction_type": interaction,
                         "biological_effect": _effect_text(sign, pathways.get(key, [])) if interaction else None})

    llm_error = _fill_with_llm(drug, rows, attempt=1)
    if not llm_error:
        llm_error = _fill_with_llm(drug, rows, attempt=2)     # once more, only for cells the first answer left empty
    for row in rows:
        for field in FIELDS:
            row[field] = row[field] or "—"
    return {"drug": drug, "rows": rows, "llm_error": llm_error}


def _interactions(relation_types, effect):
    """(interaction, sign) for each way the drug acts on the target: from the papers' relationships, else from
    the target's Effect of Drug, else one unknown row (None) for the LLM."""
    if relation_types:
        return [(INTERACTION_TYPES.get(relation, relation.replace("_", " ").capitalize()), relation_sign(relation))
                for relation in sorted(relation_types)]
    if effect:
        return [EFFECT_OF_DRUG.get(effect, (effect.capitalize(), 0))]
    return [(None, None)]


def _effect_text(sign, pathway_links):
    """e.g. "Reduces type I interferon signaling": the interaction combined with the target's link to each
    pathway (inhibits + activates = reduces). At most MAX_PATHWAYS pathways, the ones with the most papers first."""
    texts = []
    for pathway, link_sign, _ in sorted(pathway_links, key=lambda link: (-link[2], link[0])):
        text = f"{EFFECT_WORDS[sign * link_sign]} {pathway}"
        if text not in texts:
            texts.append(text)
    return "; ".join(texts[:MAX_PATHWAYS]) or None


def _fill_with_llm(drug, rows, attempt):
    """Asks the LLM only about the rows with empty cells (ROWS_PER_LLM_CALL rows per call) and fills only the
    empty cells. Returns True if an LLM call failed."""
    missing = [row for row in rows if not row["interaction_type"] or not row["biological_effect"]]
    failed = False
    for start in range(0, len(missing), ROWS_PER_LLM_CALL):
        batch = missing[start:start + ROWS_PER_LLM_CALL]
        try:
            answers = ask_llm(drug, tuple((row["target"], row["interaction_type"] or "") for row in batch), attempt)
        except Exception as e:
            logger.error(f"Target Validation: the LLM call for {drug} failed: {e}")
            failed = True
            continue
        for number, row in enumerate(batch, 1):
            answer = answers.get(str(number), {})
            for field in FIELDS:
                row[field] = row[field] or answer.get(field)
    return failed


@lru_cache(maxsize=100)          # each drug and list of targets is asked once (Streamlit runs the page on every click)
def ask_llm(drug, rows, attempt=1):
    """One LLM call. rows: ((target, interaction type or ""), ...); attempt 2 asks again for the cells that
    attempt 1 left empty. Returns {"1": {field: text or None}, ...} (row numbers)."""
    items = [{"id": number, "target": target, "interaction_type": interaction or None}
             for number, (target, interaction) in enumerate(rows, 1)]
    again = " Some fields were left empty before: fill every field this time." if attempt > 1 else ""
    prompt = f"""You are a pharmacology expert analysing the drug "{drug}". These targets come from the research papers on {drug}:
{json.dumps(items, ensure_ascii=False)}

For each item give:
- "interaction_type": how {drug} acts on the target, in one or two words (e.g. Inhibitor, Activator, Agonist,
  Antagonist, Binder, Modulator). If an interaction type is given, keep it. If {drug} acts on the target only
  indirectly, say so in a few words (e.g. Indirect inhibition). If {drug} has no known effect on the target,
  write "No known interaction".
- "biological_effect": in one short sentence, the biological effect that is most relevant to the analysis of
  {drug}: the effect of this interaction (e.g. "Reduces type I interferon production in dendritic cells"), or,
  when there is no known interaction, the role of the target that matters for the use of {drug}.
Fill both fields for every item. Use only established pharmacology and biology.{again}

Return STRICT JSON:
{{"interactions": [{{"id": 1, "interaction_type": "...", "biological_effect": "..."}}]}}"""
    data = json.loads(generate_json_response(prompt, temperature=0))
    return {str(item.get("id")): {field: _clean(item.get(field)) for field in FIELDS}
            for item in data.get("interactions") or [] if isinstance(item, dict)}


def _clean(value):
    """One line of text, or None for an empty or "unknown" answer."""
    text = " ".join(value.split()) if isinstance(value, str) else ""
    return None if text.lower() in ("", "null", "none", "unknown", "not known", "n/a") else text
