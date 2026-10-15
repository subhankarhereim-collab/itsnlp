import json
from functools import lru_cache

from graph_builder.repurposing_scoring import path_direction, path_strength, relation_sign
from llm_client.llm_client import generate_json_response
from neo4j_client.neo4j_client import get_session
from services.disease_synonyms import canonical_name, merge_disease_synonyms
from services.drug_repurposing import (_evidence_on, display_name, drug_names_in_graph, find_paths,
                                       get_existing_files, get_papers)
from services.target_dashboard_service import get_all_targets
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Target Validation (Drug-Target Interaction Analysis): the "Target Validation" mode of streamlit_app.py.
# One row per target (the targets of the Target Intelligence Dashboard) and per way the drug acts on it.
# The knowledge graph first; the LLM fills only the cells the graph cannot fill.
# Each cell says where it comes from: [paper citations], [target details] (the target's Details
# in the Target Intelligence Dashboard) or (LLM).
#   Interaction type   the drug -> target relationships in the papers (INHIBITS -> Inhibitor),
#                      else the target's Effect of Drug, else the LLM
#   Biological effect  the target's links to pathways, combined with the interaction:
#                      the drug inhibits TLR7 + TLR7 activates type I interferon signaling
#                      -> "Reduces type I interferon signaling", else the LLM
#   Disease outcome    the mechanism paths of Drug Repurposing that go through the target
#                      (drug -> target -> ... -> disease), else the target's Curable Diseases, else the LLM
#   Confidence         the target's Confidence Score (as in its Details);
#                      "Unverified (LLM)" when the interaction comes from the LLM
# Names: the drug's other names (e.g. HCQ) and the diseases' other names (e.g. SLE) count as one
# (the name lists), and targets that differ only in capital letters are one target.
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

TARGET_DETAILS_QUERY = """
MATCH (t:Target)
RETURN t.name AS target, t.effect_of_drug AS effect, t.disease_that_can_be_cured AS diseases,
       t.confidenceScore AS confidence
"""

INTERACTION_TYPES = {"INHIBITS": "Inhibitor", "ACTIVATES": "Activator", "BINDS": "Binder",
                     "REGULATES": "Modulator", "ASSOCIATED_WITH": "Associated"}
EFFECT_OF_DRUG = {"inhibition": ("Inhibitor", -1), "antagonism": ("Antagonist", -1),
                  "downregulation": ("Downregulator", -1), "activation": ("Activator", 1),
                  "agonism": ("Agonist", 1), "upregulation": ("Upregulator", 1), "modulation": ("Modulator", 0)}
EFFECT_WORDS = {-1: "Reduces", 1: "Increases", 0: "Affects"}
OUTCOME_WORDS = {-1: "May help", 1: "May worsen", 0: "Linked to"}
LLM_FIELDS = ["interaction_type", "biological_effect", "disease_outcome"]
MAX_ITEMS = 3                 # pathways or diseases shown in one cell, the best supported first
ROWS_PER_LLM_CALL = 25        # keeps each LLM answer short, so it is not cut off


def get_target_validation(drug):
    """{"drug", "rows": [{"target", "interaction_type", "biological_effect", "disease_outcome", "confidence"}],
    "llm_error"} for the drug. A cell that neither the graph nor the LLM can fill is "—"."""
    existing_files, papers = get_existing_files(), get_papers()
    interactions, pathways, details = {}, {}, {}
    with get_session() as session:
        # How the drug acts on each target: the drug -> target relationships (also under its other names, e.g. HCQ)
        for record in session.run(DRUG_TARGET_QUERY, drugs=drug_names_in_graph(drug)):
            link = _evidence_on(record["type"], record["props"], existing_files)    # deleted papers are left out
            if relation_sign(record["type"]) is not None and link["sentences"]:      # "did not inhibit" is no interaction
                interactions.setdefault(str(record["target"]).lower(), {}).setdefault(record["type"], []).extend(link["files"])
        # Downstream: the target's links to pathways
        for record in session.run(TARGET_PATHWAY_QUERY):
            link = _evidence_on(record["type"], record["props"], existing_files)
            sign = relation_sign(record["type"])
            if sign is not None and link["sentences"]:
                pathways.setdefault(str(record["target"]).lower(), []).append(
                    (record["pathway"], sign, _citations(link["files"], papers), len(set(link["files"]))))
        # The target's details from the target pipeline (Effect of Drug, Curable Diseases, Confidence Score)
        for record in session.run(TARGET_DETAILS_QUERY):
            info = details.setdefault(str(record["target"]).lower(), {})
            for field in ["effect", "diseases", "confidence"]:
                if record[field] is not None:
                    info.setdefault(field, record[field])
    outcomes = _disease_outcomes(drug, existing_files, papers)

    rows, seen = [], set()
    for target in get_all_targets():
        key = str(target).lower()
        if not target or key in seen:            # "TLR7" and "tlr7" are one target
            continue
        seen.add(key)
        info = details.get(key, {})
        for interaction, source, sign, relation in _interactions(interactions.get(key, {}), info, papers):
            rows.append({
                "target": target,
                "interaction": interaction,       # without its source, for the LLM (removed below)
                "interaction_type": f"{interaction} {source}" if interaction else None,
                "biological_effect": _effect_text(sign, pathways.get(key, [])) if interaction else None,
                "disease_outcome": outcomes.get((key, relation)) or _curable_text(info.get("diseases")),
                "confidence": str(info["confidence"]) if interaction and "confidence" in info else None,
            })
    llm_error = _fill_with_llm(drug, rows)
    for row in rows:
        del row["interaction"]
        for field in row:
            row[field] = row[field] or "—"
    return {"drug": drug, "rows": rows, "llm_error": llm_error}


def _interactions(found, info, papers):
    """(interaction, source, sign, relationship type) for each way the drug acts on the target:
    from the papers, else from the target's Effect of Drug, else one unknown row for the LLM."""
    if found:                                    # {relationship type: files}
        return [(INTERACTION_TYPES.get(relation, relation.replace("_", " ").capitalize()),
                 _citations(files, papers), relation_sign(relation), relation)
                for relation, files in sorted(found.items())]
    effect = str(info.get("effect") or "").strip().lower()
    if effect and effect != "none":
        name, sign = EFFECT_OF_DRUG.get(effect, (effect.capitalize(), 0))
        return [(name, "[target details]", sign, None)]
    return [(None, None, None, None)]


def _effect_text(sign, pathway_links):
    """e.g. "Reduces type I interferon signaling [Patel et al. (2021)]": the interaction combined with the
    target's link to the pathway (inhibits + activates = reduces). The pathways with the most papers first."""
    links = sorted(pathway_links, key=lambda link: (-link[3], link[0]))
    return _join([f"{EFFECT_WORDS[sign * link_sign]} {pathway} {citations}" for pathway, link_sign, citations, _ in links])


def _disease_outcomes(drug, existing_files, papers):
    """{(target, relationship type): text} from the mechanism paths of Drug Repurposing that start
    drug -> target, e.g. "May help Systemic lupus erythematosus (via type I interferon signaling) [Patel et al. (2021)]".
    One path per disease and direction: the strongest."""
    found = {}
    for path in merge_disease_synonyms(find_paths(drug, existing_files)):
        direction = path_direction(path["links"])
        if path["labels"][1] != "Target" or direction is None:          # None: a step says "no effect"
            continue
        via = f" (via {', '.join(path['names'][2:-1])})" if len(path["names"]) > 3 else ""
        files = [file for link in path["links"][1:] for file in link["files"]]   # the papers after the interaction
        found.setdefault((path["names"][1].lower(), path["links"][0]["type"]), []).append(
            (path_strength(path["links"], direction), display_name(path["names"][-1]), direction, via, files))
    outcomes = {}
    for key, items in found.items():
        texts, shown = [], set()
        for _, disease, direction, via, files in sorted(items, key=lambda item: (-item[0], item[1], item[3])):
            if (disease, direction) not in shown:
                shown.add((disease, direction))
                texts.append(f"{OUTCOME_WORDS[direction]} {disease}{via} {_citations(files, papers)}")
        outcomes[key] = _join(texts)
    return outcomes


def _curable_text(diseases):
    """The target's Curable Diseases, e.g. "May help Rheumatoid arthritis [target details]"."""
    if isinstance(diseases, str):
        diseases = [diseases]
    names = []
    for disease in diseases or []:
        name = display_name(canonical_name(str(disease)) or disease) if disease else ""
        if name and name.lower() not in [known.lower() for known in names]:
            names.append(name)
    return f"May help {_join(names, ', ')} [target details]" if names else None


def _citations(files, papers):
    """e.g. "[Lee et al. (2022); Smith et al. (2020)]" for the papers of the evidence sentences."""
    names = sorted({papers.get(file, {}).get("citation") or file for file in files if file})
    return f"[{'; '.join(names or ['paper without a name'])}]"


def _join(texts, separator="; "):
    """At most MAX_ITEMS texts, e.g. "A; B; C (+2 more)"; None when there is no text."""
    if not texts:
        return None
    more = f" (+{len(texts) - MAX_ITEMS} more)" if len(texts) > MAX_ITEMS else ""
    return separator.join(texts[:MAX_ITEMS]) + more


def _fill_with_llm(drug, rows):
    """Asks the LLM only about the rows with empty cells (ROWS_PER_LLM_CALL rows per call) and fills only
    the empty cells, marked "(LLM)". Returns True if an LLM call failed."""
    missing = [row for row in rows if None in (row["interaction_type"], row["biological_effect"], row["disease_outcome"])]
    failed = False
    for start in range(0, len(missing), ROWS_PER_LLM_CALL):
        batch = missing[start:start + ROWS_PER_LLM_CALL]
        try:
            answers = ask_llm(drug, tuple((row["target"], row["interaction"] or "") for row in batch))
        except Exception as e:
            logger.error(f"Target Validation: the LLM call for {drug} failed: {e}")
            failed = True
            continue
        for number, row in enumerate(batch, 1):
            answer = answers.get(str(number), {})
            if row["interaction_type"] is None and answer.get("interaction_type"):
                row["interaction_type"] = f"{answer['interaction_type']} (LLM)"
                row["confidence"] = "Unverified (LLM)"
            for field in ["biological_effect", "disease_outcome"]:
                if row[field] is None and answer.get(field):
                    row[field] = f"{answer[field]} (LLM)"
    return failed


@lru_cache(maxsize=100)          # each drug and list of targets is asked once (Streamlit runs the page on every click)
def ask_llm(drug, rows):
    """One LLM call. rows: ((target, interaction type or ""), ...). Returns {"1": {field: text or None}, ...}."""
    items = [{"id": number, "target": target, "interaction_type": interaction or None}
             for number, (target, interaction) in enumerate(rows, 1)]
    prompt = f"""You are a pharmacology expert. For the drug "{drug}", describe its interaction with each of these targets:
{json.dumps(items, ensure_ascii=False)}

For each item give:
- "interaction_type": how {drug} acts on the target, in one or two words (e.g. Inhibitor, Activator, Agonist,
  Antagonist, Binder, Modulator). If an interaction type is given, keep it. If {drug} is not known to act on
  the target, write "No known interaction".
- "biological_effect": the downstream biological effect of this interaction, in one short sentence
  (e.g. "Reduces type I interferon production in dendritic cells").
- "disease_outcome": what this effect means for disease, in one short sentence
  (e.g. "May reduce disease activity in systemic lupus erythematosus").
Use only facts you are sure of. Write null for a field you are not sure of, and for the effect and the
outcome when there is no known interaction.

Return STRICT JSON:
{{"interactions": [{{"id": 1, "interaction_type": "...", "biological_effect": "...", "disease_outcome": "..."}}]}}"""
    data = json.loads(generate_json_response(prompt, temperature=0))
    return {str(item.get("id")): {field: _clean(item.get(field)) for field in LLM_FIELDS}
            for item in data.get("interactions") or [] if isinstance(item, dict)}


def _clean(value):
    """One line of text, or None for an empty or "unknown" answer."""
    text = " ".join(value.split()) if isinstance(value, str) else ""
    return None if text.lower() in ("", "null", "none", "unknown", "not known", "n/a") else text
