import xml.etree.ElementTree as ET

from med_storm.ingestion.conflict_scraper import ConflictFundingScraper


_SAMPLE_XML = """
<PubmedArticle>
  <MedlineCitation>
    <PMID>12345</PMID>
    <Article>
      <Abstract>
        <AbstractText Label="Conflict of interest">No conflicts of interest declared.</AbstractText>
      </Abstract>
    </Article>
    <GrantList>
      <Grant>
        <Agency>NIH</Agency>
        <GrantID>R01HL123456</GrantID>
      </Grant>
      <Grant>
        <Agency>Pfizer</Agency>
        <GrantID>INVEST-001</GrantID>
      </Grant>
    </GrantList>
  </MedlineCitation>
</PubmedArticle>
"""


def _build_article_node():
    root = ET.fromstring(_SAMPLE_XML)
    return root  # PubmedArticle element


def test_extract_conflict_and_funding():
    article = _build_article_node()
    conflict = ConflictFundingScraper._extract_conflict(article)
    funding = ConflictFundingScraper._extract_funding(article)
    flags = ConflictFundingScraper._flag_risk(conflict, funding)

    assert conflict == "No conflicts of interest declared."
    assert "NIH" in funding and "Pfizer" in funding
    assert flags["industry_sponsored"] is True
    assert flags["no_disclosure"] is False 