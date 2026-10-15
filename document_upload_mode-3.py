if mode == "Document Upload":
    from document_preprocessing.pubmed_service import search_and_download_papers   # PubMed search (your pipeline)

    st.header("Upload biomedical documents or ask questions")

    # Initialize uploader key for resetting after processing
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # ------------------------------------------------
    # ADD PAPERS TO data/: manual upload and/or papers found by drug name
    # (the knowledge graph is built later, with the "Generate KG" button)
    # ------------------------------------------------
    def paper_search(key):
        """Drug name (+ optional minimum year): your PubMed pipeline downloads the best-ranked paper into data/.
        The PubMed and iCite tables (CSV and JSON) are saved in the pubmed_data/ folder of the project."""
        drug_name = st.text_input("Drug name", placeholder="e.g. Hydroxychloroquine", key=f"{key}_drug")
        year_text = st.text_input("Minimum publication year (optional)",
                                  placeholder="e.g. 2015 (leave empty for all years)", key=f"{key}_year")
        result_key = f"{key}_paper_search_result"

        if st.button("Search and download", key=f"{key}_search"):
            drug, year = drug_name.strip(), year_text.strip()
            if not drug:
                st.warning("Enter a drug name before searching PubMed.")
                return
            if year and not (year.isdigit() and len(year) == 4):
                st.warning("Enter the minimum publication year as four digits (for example 2015), or leave it empty.")
                return
            st.session_state.pop(result_key, None)
            files_before = {p.name for p in DATA_DIR.glob("*.pdf")}
            with st.spinner(f"Searching PubMed for {drug}, adding citation counts (iCite), ranking the papers and "
                            "downloading the best-ranked paper ... This can take a few minutes."):
                try:
                    result = search_and_download_papers(drug, data_folder=DATA_DIR,
                                                        minimum_publication_year=int(year) if year else None)
                except Exception:
                    logger.exception("PubMed search or download failed")
                    st.error("Could not complete the PubMed search. Retry or upload PDFs manually.")
                    return

            for download in result["downloads"]:
                fname = Path(download["file_path"]).name
                download["status"] = "Downloaded again (was already in data/)" if fname in files_before else "Downloaded now"
                if fname not in st.session_state.uploaded_file_names:
                    st.session_state.uploaded_file_names.append(fname)
            st.session_state[result_key] = result
            st.rerun()

        # Keep each search result visible across reruns, including when Generate KG is used.
        if result_key in st.session_state:
            result = st.session_state[result_key]
            downloads = result["downloads"]
            since = f", papers from {result['minimum_publication_year']} on" if result["minimum_publication_year"] else ""
            st.caption(f"Last PubMed search: {result['drug']}{since}. {result['records']} PubMed record(s), "
                       f"{result['with_pmcid']} with a free full text in PubMed Central. Ranking: of the 20 most-cited "
                       "papers (iCite), the ones in PubMed Central, newest first.")
            if downloads:
                st.success(f"✅ {len(downloads)} paper(s) on {result['drug']} downloaded into data/.")
            elif not result["records"]:
                st.warning("PubMed found no papers for this drug name. Try another name or upload papers manually.")
            elif not result["top_papers"]:
                st.warning("None of the 20 most-cited papers has a free full text in PubMed Central"
                           + (" from that year on. Try an earlier year" if since else ". Try another drug name")
                           + ", or upload papers manually.")
            else:
                st.warning("The best-ranked papers could not be downloaded right now. Retry or upload PDFs manually.")
            if downloads and len(downloads) < result["asked"]:
                st.info(f"Downloaded {len(downloads)} of the {result['asked']} papers asked for: the other "
                        "best-ranked papers have no PDF that could be downloaded.")
            if downloads:
                st.dataframe([{
                    "File": Path(download["file_path"]).name,
                    "Status": download["status"],
                    "Title": download["paper"]["title"],
                    "Year": download["paper"]["publication_year"],
                    "Citations": int(download["paper"]["citation_count"]),
                    "RCR": round(download["paper"]["rcr_score"], 2),
                    "Journal": download["paper"].get("journal", ""),
                    "PDF from": download["source"],
                    "PubMed": f"https://pubmed.ncbi.nlm.nih.gov/{download['paper']['pmid']}/",
                } for download in downloads],
                    hide_index=True,
                    column_config={"PubMed": st.column_config.LinkColumn("PubMed")})
            if result["csv_files"]:
                st.caption(f"PubMed tables (CSV and JSON) saved in {Path(result['csv_files'][0]).parent}/: "
                           + ", ".join(Path(file).name for file in result["csv_files"]))

    upload_option = st.radio("How do you want to add papers?",
                             ["Manual upload", "Search PubMed by drug name"], horizontal=True)
    uploaded_files = None
    current_key = None

    if upload_option == "Manual upload":
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDFs to build knowledge graph",
            key=f"file_uploader_{st.session_state.uploader_key}"
        )

        if uploaded_files:
            # Don't clear existing PDFs - just add new ones
            with st.spinner("Saving uploaded PDFs to data/ ..."):
                try:
                    saved_paths = save_uploaded_pdfs(uploaded_files)
                    # Append new file names to existing list
                    for fname in [Path(p).name for p in saved_paths]:
                        if fname not in st.session_state.uploaded_file_names:
                            st.session_state.uploaded_file_names.append(fname)
                except Exception as e:
                    st.error(f"Failed to save PDFs: {e}")
                    st.stop()

            # Reset uploader key to clear the file list display (the files are in data/ now)
            st.session_state.uploader_key += 1
            st.rerun()

        # Optional: also add papers from the web, with the same search as the other option
        if st.checkbox("Also add papers from the web (search by drug name)", key="manual_web_search"):
            paper_search("manual")

    else:
        paper_search("pubmed")

    # ------------------------------------------------
    # SHOW PREVIOUS UPLOADS (PERSISTENT)
    # ------------------------------------------------
    if st.session_state.uploaded_file_names:
        st.success(f"✅ Documents uploaded: {len(st.session_state.uploaded_file_names)} file(s)")

        # Display each file with a close button
        for fname in st.session_state.uploaded_file_names:
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.write(f"• {fname}")
            with col2:
                if st.button("✕", key=f"remove_{fname}"):
                    # Remove from session state
                    st.session_state.uploaded_file_names.remove(fname)

                    # Remove the file from data/ directory
                    file_path = DATA_DIR / fname
                    try:
                        if file_path.exists():
                            # IMPORTANT: Unmark BEFORE deleting, so hash can be calculated
                            unmark_file_as_processed(str(file_path))
                            file_path.unlink()
                            logger.info(f"File deleted from data/: {fname}")
                        else:
                            logger.warning(f"File not found in data/ folder: {fname}")
                    except Exception as e:
                        logger.error(f"Failed to delete file {fname} from data/: {str(e)}")
                        st.error(f"⚠️ Could not delete {fname} from disk: {str(e)}")

                    # Delete nodes from this specific file (not the entire graph)
                    try:
                        logger.info(f"Deleting graph nodes for: {fname}")
                        result = delete_nodes_by_source_file(fname)
                        st.success(f"✅ {result['message']}")
                        logger.info(f"Successfully deleted nodes: {result}")
                    except Exception as e:
                        logger.error(f"Failed to delete graph nodes for {fname}: {str(e)}")
                        st.error(f"❌ Failed to delete graph nodes: {str(e)}")

                    # If no more files, reset documents_uploaded flag
                    if not st.session_state.uploaded_file_names:
                        st.session_state.documents_uploaded = False

                    # Reset upload key to prevent triggering reprocessing
                    st.session_state.last_upload_key = None
                    st.rerun()

        st.markdown("---")

    # ------------------------------------------------
    # CLEAR ALL BUTTON (ALWAYS VISIBLE)
    # ------------------------------------------------

    if st.button("🗑️ Clear all documents", type="secondary"):
        st.session_state.uploaded_file_names = []
        st.session_state.documents_uploaded = False
        clear_existing_pdfs()
        wipe_neo4j_graph()
        st.success("✅ All documents cleared from data folder and graph!")
        st.rerun()

    # ------------------------------------------------
    # GENERATE KG: build the knowledge graph from the papers in data/
    # ------------------------------------------------
    st.markdown("---")
    st.subheader("Generate knowledge graph")

    # Papers in data/, and the ones that are not in the knowledge graph yet (the processed-files registry knows)
    cleanup_orphaned_registry_entries()
    all_papers = sorted(p.name for p in DATA_DIR.glob("*.pdf"))
    new_papers = [f for f in all_papers if not is_file_already_processed(str(DATA_DIR / f))]
    st.write(f"📄 Papers in data/: {len(all_papers)} ({len(new_papers)} not in the knowledge graph yet)")

    if st.button("Generate KG", type="primary"):
        if all_papers:
            st.session_state.show_kg_options = True
        else:
            st.warning("No papers in data/ yet. Upload papers or search PubMed first.")
    if not all_papers:
        st.session_state.show_kg_options = False      # nothing to build (for example after "Clear all documents")

    # After "Generate KG": keep the previous KG or delete it
    if st.session_state.get("show_kg_options"):
        st.write("What should happen to the previous knowledge graph?")
        col1, col2 = st.columns(2)
        with col1:
            keep_clicked = st.button("Keep the previous KG and add the new papers")
        with col2:
            delete_clicked = st.button("Delete the previous KG and build it again from all papers")

        if keep_clicked or delete_clicked:
            st.session_state.show_kg_options = False
            # Keep: read only the papers that are not in the KG yet. Delete: read every paper in data/.
            papers_to_read = all_papers if delete_clicked else new_papers

            if not papers_to_read:
                st.session_state.kg_message = "✅ The knowledge graph is up to date: all papers in data/ are already in it."
            else:
                if delete_clicked:
                    st.info(f"🗑️ Deleting the previous knowledge graph, then reading all {len(papers_to_read)} paper(s)...")
                else:
                    st.info(f"📚 Adding {len(papers_to_read)} new paper(s) to the existing knowledge graph...")

                # Create placeholder for status updates
                status_placeholder = st.empty()
                progress_placeholder = st.empty()

                def status_callback(current_file: int, total_files: int, message: str):
                    """Callback function to update Streamlit UI with progress"""
                    progress = current_file / total_files if total_files > 0 else 0
                    progress_placeholder.progress(progress)
                    status_placeholder.info(f"📊 **Status:** {message}")

                try:
                    if delete_clicked:
                        wipe_neo4j_graph()              # same delete as "Clear all documents", but the papers stay
                        for fname in all_papers:        # the KG is empty now, so no paper is in it any more
                            unmark_file_as_processed(str(DATA_DIR / fname))
                    build_kg_from_directory(DATA_DIR, files_to_process=papers_to_read, status_callback=status_callback)
                except Exception as e:
                    status_placeholder.empty()
                    progress_placeholder.empty()
                    st.error(f"❌ Failed to build knowledge graph: {e}")
                    st.stop()

                st.session_state.documents_uploaded = True
                if delete_clicked:
                    st.session_state.kg_message = (f"✅ Previous knowledge graph deleted. "
                                                   f"New knowledge graph built from {len(papers_to_read)} paper(s).")
                else:
                    st.session_state.kg_message = f"✅ {len(papers_to_read)} new paper(s) added to the knowledge graph."

            # Synonyms and abbreviations of the new drug and disease names, from the LLM (only names that
            # are not in the name lists yet). Drug Repurposing then shows one name for each.
            with st.spinner("Updating the drug and disease name lists..."):
                try:
                    sent = update_name_synonyms()
                    if sent["Drug"] + sent["Disease"] > 0:
                        st.session_state.names_message = ("success", (
                            f"✅ Name lists updated: {sent['Drug']} new drug name(s) and {sent['Disease']} new "
                            "disease name(s) added. Different names for the same drug or disease (for example "
                            "HCQ and hydroxychloroquine) are shown as one in Drug Repurposing."))
                    else:
                        st.session_state.names_message = ("success", (
                            "✅ Name lists are up to date: all drug and disease names in the knowledge graph "
                            "were already known."))
                except Exception as e:
                    logger.warning(f"Could not update the drug and disease name lists: {e}")   # details for the developer
                    st.session_state.names_message = ("info", (
                        "ℹ️ The knowledge graph is ready. The drug and disease name lists could not be updated "
                        "this time, so a few names may show separately in Drug Repurposing. They will be updated "
                        "at the next Generate KG."))
            st.rerun()

    # Result of the last "Generate KG": shown once, after the page refresh
    if "kg_message" in st.session_state:
        st.success(st.session_state.pop("kg_message"))
    if "names_message" in st.session_state:
        kind, text = st.session_state.pop("names_message")
        if kind == "success":
            st.success(text)
        else:
            st.info(text)

    # ------------------------------------------------
    # Generate summaries (only for manual uploads)
    # ------------------------------------------------
    if uploaded_files and current_key:
        st.markdown("---")
        st.subheader("Generate summaries")

        if st.button("Generate Summaries for All PDFs"):
            try:
                pdf_paths = [str(p.resolve()) for p in DATA_DIR.glob("*.pdf")]
                if not pdf_paths:
                    st.warning("No PDFs found in data/.")
                else:
                    with st.spinner("Generating summaries using LLM ..."):
                        for p in pdf_paths:
                            summary = generate_summary_from_pdf(p)
                            st.write(f"### Summary of {Path(p).name}:")
                            st.markdown(summary)

                    st.success("Summaries completed.")
            except Exception as e:
                st.error(f"Summary generation failed: {e}")
