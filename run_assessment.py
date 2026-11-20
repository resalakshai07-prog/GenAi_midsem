"""Run the five assigned questions and write their exact answer() outputs."""
from __future__ import annotations

import json
from pathlib import Path

from rag import answer

QUESTIONS = [
    "In which city and at what garage street address did the author and their co-founder assemble their first 50 microprocessor circuit boards in 1984?",
    "What caused the catastrophic thermal failure of the Series-4 prototype during the trade-show demonstration described in Chapter 5?",
    "What was Nvidia Corporation's market capitalization in 2026?",
    "How much emergency bridge financing, in USD, did the author secure by mortgaging their family home during the 1991 semiconductor downturn?",
    "What are the system requirements for running Unreal Engine 5 on a Windows laptop?",
]


def main() -> None:
    outputs = [(question, answer(question)) for question in QUESTIONS]
    report = ["# Assessment Question Outputs", "", "Outputs produced by `rag.answer(question)` using only the assigned PDF.", ""]
    for index, (question, result) in enumerate(outputs, 1):
        report.extend([
            f"## Q{index}",
            "",
            f"**Question:** {question}",
            "",
            "```json",
            json.dumps(result, ensure_ascii=False, indent=2),
            "```",
            "",
        ])
    target = Path(__file__).parent / "ASSESSMENT_OUTPUTS.md"
    target.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()
