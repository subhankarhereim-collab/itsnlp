import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import requests

# Add the project root when this file is run directly from document_preprocessing/.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from document_preprocessing.drug_paper_fetcher import download_pdf

# ======================================================
# SETTINGS FOR THE APP (Document Upload page)
# ======================================================

PAPERS_TO_DOWNLOAD = 1          # papers downloaded per search: change this number to download more
CSV_FOLDER = "pubmed_data"      # PubMed and iCite tables (CSV and JSON); the PDFs go to the data folder

# ======================================================
# FILTER AND RANK PAPERS
# ======================================================

def filter_pubmed_papers(
    csv_or_dataframe,
    citation_shortlist_size=20,
    max_results=5,
    minimum_publication_year=None
):
    """
    Filter and rank PubMed papers.

    Processing order:
    1. Read the enriched PubMed CSV.
    2. Remove records without a valid PMID.
    3. Sort by citation count.
    4. Keep the top citation_shortlist_size papers.
    5. Keep only papers with a PMCID.
    6. Apply an optional minimum publication year.
    7. Rank by year, RCR, NIH percentile and citations.
    8. Return up to max_results papers.

    Parameters
    ----------
    csv_or_dataframe : str, Path, or pandas.DataFrame
        Path to the iCite-enriched PubMed CSV, or an
        existing DataFrame.

    citation_shortlist_size : int
        Number of highly cited papers to consider
        before filtering for PMC availability.

    max_results : int
        Maximum number of papers to return.

    minimum_publication_year : int or None
        If provided, papers older than this year are
        excluded.

    Returns
    -------
    list[dict]
        Ranked list of paper dictionaries.
    """

    if isinstance(csv_or_dataframe, pd.DataFrame):
        df = csv_or_dataframe.copy()

    else:
        df = pd.read_csv(
            csv_or_dataframe,
            dtype={
                "pmid": str,
                "pmcid": str
            }
        )

    if df.empty:
        print("The PubMed dataset is empty.")
        return []

    required_columns = [
        "pmid",
        "pmcid",
        "title",
        "publication_year",
        "citation_count"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The input dataset is missing required "
            f"columns: {missing_columns}"
        )

    # Add optional ranking columns when unavailable.
    if "rcr_score" not in df.columns:
        df["rcr_score"] = 0

    if "nih_percentile" not in df.columns:
        df["nih_percentile"] = 0

    if "pmc_pdf_url" not in df.columns:
        df["pmc_pdf_url"] = ""

    if "pmc_article_url" not in df.columns:
        df["pmc_article_url"] = ""

    # Normalize identifier columns.
    df["pmid"] = (
        df["pmid"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["pmcid"] = (
        df["pmcid"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Normalize numeric ranking fields.
    numeric_columns = [
        "publication_year",
        "citation_count",
        "rcr_score",
        "nih_percentile"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

    # Remove invalid PMID rows and duplicate papers.
    df = df[
        df["pmid"].ne("")
    ].copy()

    df = df.drop_duplicates(
        subset=["pmid"],
        keep="first"
    )

    if df.empty:
        print("No valid PMID records were found.")
        return []

    # Step 1: Shortlist the most-cited articles.
    citation_shortlist = (
        df.sort_values(
            by=[
                "citation_count",
                "rcr_score",
                "nih_percentile"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .head(citation_shortlist_size)
        .copy()
    )

    # Step 2: Keep papers that have a PMCID.
    downloadable = citation_shortlist[
        citation_shortlist["pmcid"].ne("")
    ].copy()

    # Step 3: Apply publication year filter.
    if minimum_publication_year is not None:

        downloadable = downloadable[
            downloadable["publication_year"]
            >= minimum_publication_year
        ].copy()

    if downloadable.empty:

        print(
            "No downloadable papers found "
            "after applying filters."
        )

        return []

    # Generate PMC URLs if missing.
    downloadable["pmc_article_url"] = (
        downloadable["pmc_article_url"]
        .fillna("")
    )

    downloadable["pmc_pdf_url"] = (
        downloadable["pmc_pdf_url"]
        .fillna("")
    )

    missing_article_url = (
        downloadable["pmc_article_url"]
        .str.strip()
        .eq("")
    )

    downloadable.loc[
        missing_article_url,
        "pmc_article_url"
    ] = downloadable.loc[
        missing_article_url,
        "pmcid"
    ].apply(
        lambda x:
        f"https://pmc.ncbi.nlm.nih.gov/articles/{x}/"
    )

    missing_pdf_url = (
        downloadable["pmc_pdf_url"]
        .str.strip()
        .eq("")
    )

    downloadable.loc[
        missing_pdf_url,
        "pmc_pdf_url"
    ] = downloadable.loc[
        missing_pdf_url,
        "pmcid"
    ].apply(
        lambda x:
        f"https://pmc.ncbi.nlm.nih.gov/articles/{x}/pdf/"
    )

    # Step 4: Re-rank downloadable papers.
    ranked = (
        downloadable.sort_values(
            by=[
                "publication_year",
                "rcr_score",
                "citation_count"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .head(max_results)
        .copy()
    )

    selected_columns = [
        "pmid",
        "pmcid",
        "title",
        "publication_year",
        "citation_count",
        "rcr_score",
        "nih_percentile",
        "doi",
        "journal",
        "pmc_article_url",
        "pmc_pdf_url"
    ]

    selected_columns = [
        col
        for col in selected_columns
        if col in ranked.columns
    ]

    ranked = ranked[selected_columns]

    if "publication_year" in ranked.columns:

        ranked["publication_year"] = (
            ranked["publication_year"]
            .astype(int)
        )

    results = ranked.to_dict(
        orient="records"
    )

    print("\n" + "=" * 60)
    print("PAPER FILTER SUMMARY")
    print("=" * 60)

    print(
        f"Total papers: {len(df)}"
    )

    print(
        f"Citation shortlist: "
        f"{len(citation_shortlist)}"
    )

    print(
        f"Downloadable papers: "
        f"{len(downloadable)}"
    )

    print(
        f"Selected papers: "
        f"{len(results)}"
    )

    for i, paper in enumerate(
        results,
        start=1
    ):

        print(
            f"\n{i}. "
            f"{paper.get('title', '')}"
        )

        print(
            f"PMID: "
            f"{paper.get('pmid', '')}"
        )

        print(
            f"PMCID: "
            f"{paper.get('pmcid', '')}"
        )

        print(
            f"Year: "
            f"{paper.get('publication_year', '')}"
        )

        print(
            f"Citations: "
            f"{paper.get('citation_count', 0)}"
        )

        print(
            f"RCR: "
            f"{paper.get('rcr_score', 0)}"
        )

    return results


# ======================================================
# DOWNLOAD FIRST AVAILABLE PAPER
# ======================================================

def download_first_available_paper(
    ranked_papers,
    output_folder="downloaded_papers"
):
    """
    Try papers in ranking order.

    Download the first successful PDF and stop.
    If a paper fails, move to the next one.

    Parameters
    ----------
    ranked_papers : list[dict]
        Output from filter_pubmed_papers()

    output_folder : str
        Folder to store downloaded PDFs

    Returns
    -------
    dict
    """

    output_path = Path(output_folder)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    for rank, paper in enumerate(
        ranked_papers,
        start=1
    ):

        pmcid = str(
            paper.get(
                "pmcid",
                ""
            )
        ).strip()

        title = str(
            paper.get(
                "title",
                ""
            )
        ).strip()

        if not pmcid:

            print(
                f"[Rank {rank}] No PMCID. Skipping."
            )

            continue

        safe_title = re.sub(
            r'[<>:"/\\|?*]',
            "_",
            title
        )

        safe_title = safe_title[:100]

        filename = (
            f"{pmcid}_{safe_title}.pdf"
        )

        pdf_path = (
            output_path / filename
        )

        try:

            print(
                f"\nTrying Rank {rank}"
            )

            print(
                f"PMCID: {pmcid}"
            )

            print(
                f"Title: {title}"
            )

            pdf_content, source = download_pdf(
                {
                    "pmcid": pmcid,
                    "doi": str(
                        paper.get("doi", "")
                        or ""
                    ).strip()
                }
            )

            if not pdf_content:

                print(
                    f"[Rank {rank}] "
                    "No valid PDF available."
                )

                continue

            pdf_path.write_bytes(pdf_content)

            print(
                "\nDownload Successful"
            )

            print(
                f"Source: {source}"
            )

            print(
                f"Saved: {pdf_path}"
            )

            return {
                "success": True,
                "paper": paper,
                "file_path": str(pdf_path),
                "source": source
            }

        except (OSError, requests.RequestException, ValueError) as ex:

            print(
                f"Download failed: {ex}"
            )

            time.sleep(1)

    print(
        "\nNo paper could be downloaded."
    )

    return {
        "success": False
    }


def collect_pubmed_articles(
    search_term,
    output_folder="pubmed_data",
    max_articles=2000,
    contact_email="ufs2k19@gmail.com",
    tool_name="pubmed_article_collector",
    ncbi_api_key=""
):
    """
    Search PubMed, retrieve metadata, map PMID to PMCID,
    and save the results to CSV and JSON.

    Parameters
    ----------
    search_term : str
        Drug name or complete PubMed search expression.

    output_folder : str or Path
        Folder where CSV and JSON files are saved.

    max_articles : int
        Maximum number of PubMed records to retrieve.

    contact_email : str
        Email used to identify the application to NCBI.

    tool_name : str
        Name used to identify the application to NCBI.

    ncbi_api_key : str
        Optional NCBI API key.

    Returns
    -------
    dict
        Contains:
        - dataframe
        - articles
        - csv_file
        - json_file
        - pmc_csv_file
    """

    # ==================================================
    # CONFIGURATION
    # ==================================================

    eutils_base_url = (
        "https://eutils.ncbi.nlm.nih.gov/"
        "entrez/eutils"
    )

    request_delay = 0.4
    max_retries = 5

    search_term = str(search_term).strip()

    if not search_term:
        raise ValueError(
            "Search term cannot be empty."
        )

    output_path = Path(output_folder)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            f"{tool_name}/1.0 "
            f"({contact_email})"
        )
    })

    # ==================================================
    # REQUEST HELPERS
    # ==================================================

    def add_ncbi_params(params):

        updated_params = params.copy()

        updated_params["tool"] = tool_name
        updated_params["email"] = contact_email

        if ncbi_api_key:
            updated_params["api_key"] = (
                ncbi_api_key
            )

        return updated_params

    def safe_json_request(url, params):

        for attempt in range(
            1,
            max_retries + 1
        ):

            try:
                response = session.get(
                    url,
                    params=params,
                    timeout=60
                )

                response.raise_for_status()

                return response.json()

            except (
                requests.RequestException,
                ValueError
            ) as ex:

                print(
                    f"[JSON Retry "
                    f"{attempt}/{max_retries}] "
                    f"{ex}"
                )

                if attempt < max_retries:
                    time.sleep(5 * attempt)

        raise RuntimeError(
            f"Failed JSON request: {url}"
        )

    def safe_xml_request(url, params):

        for attempt in range(
            1,
            max_retries + 1
        ):

            try:
                response = session.get(
                    url,
                    params=params,
                    timeout=120
                )

                response.raise_for_status()

                return response.content

            except requests.RequestException as ex:

                print(
                    f"[XML Retry "
                    f"{attempt}/{max_retries}] "
                    f"{ex}"
                )

                if attempt < max_retries:
                    time.sleep(5 * attempt)

        raise RuntimeError(
            f"Failed XML request: {url}"
        )

    # ==================================================
    # XML HELPERS
    # ==================================================

    def element_text(element):

        if element is None:
            return ""

        return "".join(
            element.itertext()
        ).strip()

    def unique_values(values):

        return list(
            dict.fromkeys(
                value
                for value in values
                if value
            )
        )

    def extract_abstract(article):

        sections = []

        for item in article.findall(
            ".//Abstract/AbstractText"
        ):

            text = element_text(item)

            if not text:
                continue

            label = item.attrib.get(
                "Label",
                ""
            ).strip()

            if label:
                sections.append(
                    f"{label}: {text}"
                )
            else:
                sections.append(text)

        return " ".join(sections)

    def extract_authors(article):

        authors = []

        for author in article.findall(
            ".//AuthorList/Author"
        ):

            collective_name = (
                author.findtext(
                    "CollectiveName",
                    default=""
                ).strip()
            )

            if collective_name:
                authors.append(
                    collective_name
                )
                continue

            first_name = author.findtext(
                "ForeName",
                default=""
            ).strip()

            last_name = author.findtext(
                "LastName",
                default=""
            ).strip()

            initials = author.findtext(
                "Initials",
                default=""
            ).strip()

            name = (
                f"{first_name} {last_name}"
                .strip()
            )

            if not name:
                name = initials

            if name:
                authors.append(name)

        return unique_values(authors)

    def extract_affiliations(article):

        affiliations = []

        for item in article.findall(
            ".//AffiliationInfo/Affiliation"
        ):

            text = element_text(item)

            if text:
                affiliations.append(text)

        return unique_values(affiliations)

    def extract_mesh(article):

        mesh_terms = []

        for heading in article.findall(
            ".//MeshHeading"
        ):

            descriptor = heading.find(
                "DescriptorName"
            )

            text = element_text(descriptor)

            if text:
                mesh_terms.append(text)

        return unique_values(mesh_terms)

    def extract_keywords(article):

        keywords = []

        for keyword in article.findall(
            ".//Keyword"
        ):

            text = element_text(keyword)

            if text:
                keywords.append(text)

        return unique_values(keywords)

    def extract_chemicals(article):

        chemicals = []

        for chemical in article.findall(
            ".//Chemical"
        ):

            substance = chemical.find(
                "NameOfSubstance"
            )

            text = element_text(substance)

            if text:
                chemicals.append(text)

        return unique_values(chemicals)

    def extract_publication_types(article):

        publication_types = []

        for item in article.findall(
            ".//PublicationTypeList/"
            "PublicationType"
        ):

            text = element_text(item)

            if text:
                publication_types.append(
                    text
                )

        return unique_values(
            publication_types
        )

    def extract_publication_year(article):

        year_paths = [
            ".//JournalIssue/PubDate/Year",
            ".//ArticleDate/Year",
            (
                ".//PubMedPubDate"
                "[@PubStatus='pubmed']/Year"
            ),
            (
                ".//PubMedPubDate"
                "[@PubStatus='entrez']/Year"
            )
        ]

        for path in year_paths:

            year = article.findtext(
                path,
                default=""
            ).strip()

            if year:
                return year

        medline_date = article.findtext(
            ".//JournalIssue/PubDate/"
            "MedlineDate",
            default=""
        ).strip()

        year_match = re.search(
            r"\b(?:18|19|20)\d{2}\b",
            medline_date
        )

        if year_match:
            return year_match.group(0)

        return ""

    def extract_article_ids(article):

        identifiers = {
            "pmid": "",
            "pmcid": "",
            "doi": ""
        }

        for article_id in article.findall(
            ".//PubmedData/ArticleIdList/"
            "ArticleId"
        ):

            id_type = article_id.attrib.get(
                "IdType",
                ""
            ).lower()

            value = element_text(article_id)

            if id_type == "pubmed":
                identifiers["pmid"] = value

            elif id_type == "pmc":
                identifiers["pmcid"] = value

            elif id_type == "doi":
                identifiers["doi"] = value

        if not identifiers["pmid"]:

            identifiers["pmid"] = (
                article.findtext(
                    ".//MedlineCitation/PMID",
                    default=""
                ).strip()
            )

        return identifiers

    # ==================================================
    # STEP 1: SEARCH PUBMED
    # ==================================================

    search_url = (
        f"{eutils_base_url}/esearch.fcgi"
    )

    count_params = add_ncbi_params({
        "db": "pubmed",
        "term": search_term,
        "retmode": "json",
        "retmax": 0
    })

    count_data = safe_json_request(
        search_url,
        count_params
    )

    total_count = int(
        count_data["esearchresult"]["count"]
    )

    records_to_fetch = min(
        total_count,
        max_articles
    )

    print(
        f"\nTotal PubMed records: "
        f"{total_count}"
    )

    print(
        f"Records to retrieve: "
        f"{records_to_fetch}"
    )

    if records_to_fetch == 0:

        return {
            "dataframe": pd.DataFrame(),
            "articles": [],
            "csv_file": None,
            "json_file": None,
            "pmc_csv_file": None
        }

    pmids = []
    search_batch_size = 500

    for start in range(
        0,
        records_to_fetch,
        search_batch_size
    ):

        current_batch_size = min(
            search_batch_size,
            records_to_fetch - start
        )

        params = add_ncbi_params({
            "db": "pubmed",
            "term": search_term,
            "retmode": "json",
            "retstart": start,
            "retmax": current_batch_size
        })

        search_data = safe_json_request(
            search_url,
            params
        )

        batch_pmids = search_data.get(
            "esearchresult",
            {}
        ).get(
            "idlist",
            []
        )

        pmids.extend(batch_pmids)

        print(
            f"PMIDs collected: "
            f"{len(pmids)}/"
            f"{records_to_fetch}"
        )

        time.sleep(request_delay)

    pmids = list(
        dict.fromkeys(pmids)
    )

    # ==================================================
    # STEP 2: CONVERT PMID TO PMCID
    # ==================================================

    pmc_mapping = {}
    converter_batch_size = 200
    elink_url = f"{eutils_base_url}/elink.fcgi"

    print(
        "\nChecking PubMed-to-PMC links..."
    )

    for start in range(
        0,
        len(pmids),
        converter_batch_size
    ):

        batch_pmids = pmids[
            start:start + converter_batch_size
        ]

        link_params = add_ncbi_params({
            "dbfrom": "pubmed",
            "db": "pmc",
            "linkname": "pubmed_pmc",
            # A list sends one id= per PMID, so NCBI returns one
            # linkset per PMID (one-to-one PMID -> PMCID mapping).
            "id": batch_pmids,
            "retmode": "json"
        })

        link_data = safe_json_request(
            elink_url,
            link_params
        )

        for linkset in link_data.get("linksets", []):
            pmid = next(
                (
                    str(value).strip()
                    for value in linkset.get("ids", [])
                    if str(value).strip()
                ),
                ""
            )

            if not pmid:
                continue

            pmc_ids = [
                str(pmc_id).strip()
                for database in linkset.get("linksetdbs", [])
                if database.get("linkname") == "pubmed_pmc"
                for pmc_id in database.get("links", [])
                if str(pmc_id).strip()
            ]

            if pmc_ids:
                pmc_mapping[pmid] = {
                    "pmcid": f"PMC{pmc_ids[0]}",
                    "doi": ""
                }

        processed = min(
            start + converter_batch_size,
            len(pmids)
        )

        print(
            f"PMC IDs checked: "
            f"{processed}/{len(pmids)}"
        )

        time.sleep(request_delay)

    # ==================================================
    # STEP 3: FETCH ARTICLE METADATA
    # ==================================================

    fetch_url = (
        f"{eutils_base_url}/efetch.fcgi"
    )

    articles = []
    fetch_batch_size = 100

    for start in range(
        0,
        len(pmids),
        fetch_batch_size
    ):

        batch_pmids = pmids[
            start:start + fetch_batch_size
        ]

        fetch_params = add_ncbi_params({
            "db": "pubmed",
            "id": ",".join(batch_pmids),
            "retmode": "xml"
        })

        xml_content = safe_xml_request(
            fetch_url,
            fetch_params
        )

        try:
            root = ET.fromstring(
                xml_content
            )

        except ET.ParseError as ex:

            print(
                f"XML parsing failed for batch "
                f"{start}: {ex}"
            )

            continue

        for article in root.findall(
            ".//PubmedArticle"
        ):

            try:
                identifiers = extract_article_ids(
                    article
                )

                pmid = identifiers["pmid"]

                converted_ids = pmc_mapping.get(
                    pmid,
                    {}
                )

                pmcid = (
                    converted_ids.get("pmcid")
                    or identifiers["pmcid"]
                )

                doi = (
                    converted_ids.get("doi")
                    or identifiers["doi"]
                )

                title = element_text(
                    article.find(
                        ".//Article/ArticleTitle"
                    )
                )

                journal = element_text(
                    article.find(
                        ".//Article/Journal/Title"
                    )
                )

                is_in_pmc = bool(pmcid)

                pmc_article_url = ""
                pmc_pdf_url = ""

                if pmcid:
                    pmc_article_url = (
                        "https://pmc.ncbi.nlm.nih.gov/"
                        f"articles/{pmcid}/"
                    )

                    pmc_pdf_url = (
                        "https://pmc.ncbi.nlm.nih.gov/"
                        f"articles/{pmcid}/pdf/"
                    )

                articles.append({
                    "search_term": search_term,
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "is_in_pmc": is_in_pmc,
                    "pmc_article_url":
                        pmc_article_url,
                    "pmc_pdf_url":
                        pmc_pdf_url,
                    "title": title,
                    "abstract":
                        extract_abstract(article),
                    "authors": "; ".join(
                        extract_authors(article)
                    ),
                    "affiliations": "; ".join(
                        extract_affiliations(
                            article
                        )
                    ),
                    "journal": journal,
                    "publication_year":
                        extract_publication_year(
                            article
                        ),
                    "doi": doi,
                    "mesh_terms": "; ".join(
                        extract_mesh(article)
                    ),
                    "keywords": "; ".join(
                        extract_keywords(article)
                    ),
                    "chemicals": "; ".join(
                        extract_chemicals(article)
                    ),
                    "publication_types": "; ".join(
                        extract_publication_types(
                            article
                        )
                    )
                })

            except Exception as ex:

                print(
                    f"Article parse error: {ex}"
                )

        processed = min(
            start + fetch_batch_size,
            len(pmids)
        )

        print(
            f"Articles processed: "
            f"{processed}/{len(pmids)}"
        )

        time.sleep(request_delay)

    # ==================================================
    # STEP 4: SAVE OUTPUTS
    # ==================================================

    df = pd.DataFrame(articles)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    safe_search_term = re.sub(
        r'[<>:"/\\|?*]+',
        "_",
        search_term
    )

    safe_search_term = re.sub(
        r"\s+",
        "_",
        safe_search_term.strip()
    )

    safe_search_term = (
        safe_search_term[:120]
        or "pubmed_results"
    )

    csv_file = output_path / (
        f"{safe_search_term}_{timestamp}.csv"
    )

    json_file = output_path / (
        f"{safe_search_term}_{timestamp}.json"
    )

    pmc_csv_file = output_path / (
        f"{safe_search_term}_pmc_only_"
        f"{timestamp}.csv"
    )

    df.to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig"
    )

    with json_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            articles,
            file,
            indent=2,
            ensure_ascii=False
        )

    pmc_df = df[
        df["pmcid"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    pmc_df.to_csv(
        pmc_csv_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("PUBMED COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total articles saved: {len(df)}")
    print(f"Articles with PMCID: {len(pmc_df)}")
    print(f"CSV: {csv_file}")
    print(f"JSON: {json_file}")
    print(f"PMC-only CSV: {pmc_csv_file}")

    return {
        "dataframe": df,
        "articles": articles,
        "csv_file": str(csv_file),
        "json_file": str(json_file),
        "pmc_csv_file": str(pmc_csv_file)
    }



def enrich_with_icite(
    csv_or_dataframe,
    output_file=None
):
    """
    Enrich PubMed articles with iCite metrics.

    Parameters
    ----------
    csv_or_dataframe : str or pd.DataFrame

    output_file : str, optional

    Returns
    -------
    dict
    {
        "dataframe": enriched_df,
        "output_file": output_file
    }
    """

    def get_icite_metrics(pmids):

        url = "https://icite.od.nih.gov/api/pubs"

        params = {
            "pmids": ",".join(
                str(pmid)
                for pmid in pmids
            )
        }

        response = requests.get(
            url,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for article in data.get(
            "data",
            []
        ):

            results.append({
                "pmid": str(
                    article.get(
                        "pmid",
                        ""
                    )
                ),
                "citation_count":
                    article.get(
                        "citation_count",
                        0
                    ),
                "rcr_score":
                    article.get(
                        "relative_citation_ratio",
                        0
                    ),
                "nih_percentile":
                    article.get(
                        "nih_percentile",
                        0
                    )
            })

        return results

    # =========================
    # INPUT
    # =========================

    if isinstance(
        csv_or_dataframe,
        pd.DataFrame
    ):

        df = csv_or_dataframe.copy()

    else:

        df = pd.read_csv(
            csv_or_dataframe,
            dtype={
                "pmid": str,
                "pmcid": str
            }
        )

    print(
        f"\nRows Loaded: {len(df)}"
    )

    pmids = (
        df["pmid"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    print(
        f"Unique PMIDs: {len(pmids)}"
    )

    # =========================
    # ICITE ENRICHMENT
    # =========================

    icite_records = []

    batch_size = 100

    total = len(pmids)

    for start in range(
        0,
        total,
        batch_size
    ):

        batch = pmids[
            start:start + batch_size
        ]

        try:

            records = (
                get_icite_metrics(
                    batch
                )
            )

            icite_records.extend(
                records
            )

            print(
                f"Processed "
                f"{min(start + batch_size, total)}"
                f"/{total}"
            )

        except requests.RequestException as ex:
            raise RuntimeError(
                f"iCite request failed for PMID batch "
                f"{start + 1}-{min(start + batch_size, total)}."
            ) from ex

        time.sleep(1)

    icite_df = pd.DataFrame(
        icite_records,
        columns=[
            "pmid",
            "citation_count",
            "rcr_score",
            "nih_percentile"
        ]
    )

    icite_df["pmid"] = (
        icite_df["pmid"]
        .astype(str)
    )

    df["pmid"] = (
        df["pmid"]
        .astype(str)
    )

    enriched_df = df.merge(
        icite_df,
        on="pmid",
        how="left"
    )

    # =========================
    # SAVE OUTPUT
    # =========================

    if output_file is None:

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_file = (
            f"pubmed_icite_enriched_"
            f"{timestamp}.csv"
        )

    enriched_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nEnriched file saved:"
    )

    print(output_file)

    print(
        f"Rows: {len(enriched_df)}"
    )

    return {
        "dataframe": enriched_df,
        "output_file": output_file
    }


# ======================================================
# APP ENTRY POINT (Document Upload page, or an API)
# ======================================================

def download_available_papers(
    ranked_papers,
    output_folder="data",
    how_many=PAPERS_TO_DOWNLOAD
):
    """
    Download up to how_many papers, best-ranked first:
    download_first_available_paper(), then again from the
    paper after the one just downloaded.

    Returns the list of successful downloads (the dicts of
    download_first_available_paper()).
    """

    downloads = []
    remaining = list(ranked_papers)

    while remaining and len(downloads) < how_many:

        result = download_first_available_paper(
            remaining,
            output_folder=output_folder
        )

        if not result["success"]:
            break

        downloads.append(result)
        remaining = remaining[remaining.index(result["paper"]) + 1:]

    return downloads


def search_and_download_papers(
    drug_name,
    data_folder="data",
    csv_folder=CSV_FOLDER,
    minimum_publication_year=None,
    how_many=PAPERS_TO_DOWNLOAD
):
    """
    The whole pipeline for one drug, without questions on the
    console: search PubMed, add iCite metrics, rank the papers
    and download the best-ranked PDF(s).

    The PDFs go to data_folder; the CSV and JSON files go to
    csv_folder. A paper that is already in data_folder is
    downloaded again (the file is replaced).

    Returns a plain dict (ready for an API): drug,
    minimum_publication_year, asked, records, with_pmcid,
    top_papers, downloads [{file_path, source, paper}], csv_files.
    """

    collection = collect_pubmed_articles(
        search_term=drug_name,
        output_folder=csv_folder
    )

    df = collection["dataframe"]

    result = {
        "drug": drug_name,
        "minimum_publication_year": minimum_publication_year,
        "asked": how_many,
        "records": len(df),
        "with_pmcid": 0,
        "top_papers": [],
        "downloads": [],
        "csv_files": [
            collection[key]
            for key in ("csv_file", "json_file", "pmc_csv_file")
            if collection[key]
        ]
    }

    if df.empty:
        return result

    result["with_pmcid"] = int(
        df["pmcid"].fillna("").astype(str).str.strip().ne("").sum()
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    enrichment = enrich_with_icite(
        df,
        output_file=(
            Path(csv_folder)
            / f"pubmed_icite_enriched_{timestamp}.csv"
        )
    )

    result["csv_files"].append(str(enrichment["output_file"]))

    result["top_papers"] = filter_pubmed_papers(
        csv_or_dataframe=enrichment["dataframe"],
        citation_shortlist_size=20,
        max_results=max(5, how_many),
        minimum_publication_year=minimum_publication_year
    )

    for download in download_available_papers(
        result["top_papers"],
        output_folder=data_folder,
        how_many=how_many
    ):
        result["downloads"].append({
            "file_path": download["file_path"],
            "source": download["source"],
            "paper": download["paper"]
        })

    return result


def main(search_term=None, minimum_publication_year=None):
    """Run PubMed search, iCite enrichment, ranking, and PDF download."""
    project_directory = Path(__file__).resolve().parents[1]     # the project folder (this file is in document_preprocessing/)
    data_directory = project_directory / "data"
    pubmed_data_directory = project_directory / "pubmed_data"
    pubmed_data_directory.mkdir(parents=True, exist_ok=True)

    if search_term is None:
        search_term = input("Enter Drug Name: ").strip()

    if minimum_publication_year is None:
        year_input = input(
            "Minimum publication year (leave blank for no minimum): "
        ).strip()
        if year_input:
            try:
                minimum_publication_year = int(year_input)
            except ValueError as ex:
                raise ValueError(
                    "Minimum publication year must be a four-digit year."
                ) from ex

    collection_result = collect_pubmed_articles(
        search_term=search_term,
        output_folder=pubmed_data_directory
    )

    if collection_result["dataframe"].empty:
        raise RuntimeError(
            f"PubMed returned no articles for {search_term!r}."
        )

    enrichment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    enrichment_result = enrich_with_icite(
        collection_result["dataframe"],
        output_file=(
            pubmed_data_directory
            / f"pubmed_icite_enriched_{enrichment_timestamp}.csv"
        )
    )

    top_papers = filter_pubmed_papers(
        csv_or_dataframe=enrichment_result["dataframe"],
        citation_shortlist_size=20,
        max_results=5,
        minimum_publication_year=minimum_publication_year
    )

    download_result = download_first_available_paper(
        top_papers,
        output_folder=data_directory
    )

    return {
        "collection": collection_result,
        "enrichment": enrichment_result,
        "top_papers": top_papers,
        "download_result": download_result,
        "pubmed_data_directory": str(pubmed_data_directory),
        "pdf_data_directory": str(data_directory)
    }


if __name__ == "__main__":
    main()
