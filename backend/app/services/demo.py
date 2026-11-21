from functools import lru_cache

import pymupdf

from app.services.hybrid_engine import extract_pdf

NOTICES = {
    "01 Registration circular.pdf": [
        ("Campus Innovation Challenge", "Registration Office | Circular 01",
         "Team registration deadline: 24 November 2026, 5:00 PM IST",
         "Registration closes at the time stated in this circular.",
         "Teams apply through the campus events portal."),
        ("Registration requirement", "Registration Office | Circular 01",
         "Required registration document: Student ID card",
         "Each participant must upload a legible copy."),
    ],
    "02 Participant handbook.pdf": [
        ("Participation rules", "Campus Innovation Challenge | Participant Handbook",
         "Maximum team size: 3 students",
         "All team members must be enrolled students.",
         "Applicants can form teams across departments."),
        ("Session venue", "Campus Innovation Challenge | Participant Handbook",
         "Orientation venue: Seminar Hall B",
         "An introductory briefing takes place before the opening ceremony."),
    ],
    "03 Coordinator notice.pdf": [
        ("Registration reminder", "Campus Innovation Challenge | Coordinator Notice",
         "Team registration deadline: 25 November 2026, 12:00 PM IST",
         "Submit the team application before this cutoff.",
         "This notice provides no reference to a superseded circular."),
        ("Event logistics", "Campus Innovation Challenge | Coordinator Notice",
         "Check-in location: Innovation Lab reception",
         "Bring a student ID for entry."),
    ],
}

EXAMPLES = [
    {"label": "Conflicting deadlines", "query": "What is the team registration deadline?"},
    {"label": "Paraphrased question", "query": "How many people can be in a group?"},
    {"label": "Missing answer", "query": "Is there a registration fee waiver?"},
]


@lru_cache
def demo_pdf(filename: str) -> bytes:
    pages = NOTICES[filename]
    with pymupdf.open() as pdf:
        for title, subtitle, *body in pages:
            page = pdf.new_page()
            page.insert_text((60, 62), "DATASCOUT / FICTIONAL DEMO NOTICE", fontsize=10)
            page.insert_text((60, 107), title, fontsize=25)
            page.insert_text((60, 137), subtitle, fontsize=12)
            page.draw_line((60, 158), (535, 158))
            for i, line in enumerate(body):
                page.insert_text((60, 193 + 25 * i), line, fontsize=12)
        return pdf.tobytes()


@lru_cache
def demo_snapshot():
    all_passages = []
    documents = []
    for filename in NOTICES:
        passages, pages = extract_pdf(demo_pdf(filename), filename)
        all_passages.extend(passages)
        documents.append(dict(filename=filename, chunk_count=len(passages), page_count=pages))
    return all_passages, documents
