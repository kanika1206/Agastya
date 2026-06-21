from __future__ import annotations

STANDARDS: tuple[dict[str, str], ...] = (
    {
        "id": "ISO/IEC 27037:2012",
        "scope": "identification, collection, acquisition and preservation of digital evidence",
    },
    {
        "id": "ISO/IEC 27041:2015",
        "scope": "assurance for digital-evidence investigation methods",
    },
    {
        "id": "ISO/IEC 27042:2015",
        "scope": "analysis and interpretation of digital evidence",
    },
    {
        "id": "ISO/IEC 27043:2015",
        "scope": "incident investigation principles and processes",
    },
    {
        "id": "NIST SP 800-86",
        "scope": "integration of forensic techniques into incident response",
    },
    {
        "id": "eIDAS Regulation (EU) 910/2014",
        "scope": "electronic identification and trust services for evidence integrity",
    },
)


def attach_standards(manifest: dict) -> dict:
    bound = {key: manifest[key] for key in manifest}
    bound["standards"] = [dict(item) for item in STANDARDS]
    return bound
