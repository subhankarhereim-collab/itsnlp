elif mode == "Target Validation":
    from services.drug_repurposing import escape_markdown, get_drugs
    from services.target_validation_service import get_target_validation

    st.header("Target Validation")
    st.caption("Drug-Target Interaction Analysis: how the selected drug acts on each target, "
               "the biological effect and what it means for diseases.")

    # The drug of the uploaded papers: the most-studied drug in the graph (limit 1)
    top_drug = get_drugs()[:1]
    if not top_drug:
        st.warning("No drug found yet. Add papers in Document Upload (upload PDFs or search by drug name), "
                   "then click Generate KG.")
        st.stop()
    drug = st.selectbox("Selected drug", top_drug)
    with st.spinner(f"Loading the Target Validation table of {drug}..."):
        validation = get_target_validation(drug)          # the knowledge graph first, the LLM only for the gaps
    if not validation["rows"]:
        st.warning("No targets found in the knowledge graph.")
        st.stop()

    table = ["| Target | Interaction Type | Biological Effect | Disease Outcome | Confidence |",
             "|---|---|---|---|---|"]
    for row in validation["rows"]:
        cells = [f"**{escape_markdown(row['target'])}**"] + [escape_markdown(row[key]) for key in
                 ["interaction_type", "biological_effect", "disease_outcome", "confidence"]]
        table.append("| " + " | ".join(cells) + " |")
    st.markdown("\n".join(table))
    st.caption("Where each cell comes from: [ ] the papers · [target details] the target's Details in the "
               "Target Intelligence Dashboard · (LLM) the LLM, please check it · — not found. "
               "Confidence is the target's Confidence Score; Unverified (LLM) means the interaction comes from the LLM.")
    if validation["llm_error"]:
        st.info("Some cells could not be filled by the LLM right now, so they show —. Please try again later.")
