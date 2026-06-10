# 🤖 RL Summer School 2026

> Personal notes, code, and resources from the Reinforcement Learning Summer School — June 3–12, 2026.

---

## 📅 Program Overview

| Week | Dates | Focus |
|------|-------|-------|
| Week 1 | June 3–5 (Wed–Fri) | Foundations — MDPs, RL Basics, Policy Gradient, Deep RL |
| Week 2 | June 8–12 (Mon–Fri) | Advanced Topics — Bandits, Multi-agent, Meta RL, RLHF, Robotics |

---


## 👨‍🏫 Speakers

| Speaker | Topic | Affiliation |
|---------|-------|-------------|
| Leif Döring | MDPs & Dynamic Programming | University of Mannheim |
| Shie Mannor | Basics of RL I | Technion & Researcher at Nvidia |
| Vincent François-Lavet | Basics of RL II + Function Approximation | VU Amsterdam |
| Semih Cayci | Policy Gradient | RWTH Aachen University |
| Matteo Hessel | Deep RL | Google DeepMind |
| Tor Lattimore | Stochastic Bandits | Google DeepMind |
| Nicolò Cesa-Bianchi | Adversarial Bandits | Università degli Studi di Milano |
| Claire Vernade | Contextual Bandits | University of Technology Nuremberg |
| Gergely Neu | Performance Difference Lemma & PO Methods | Pompeu Fabra University |
| Giorgia Ramponi | Multi-agent RL | University of Zurich |
| Aviv Tamar | Meta RL | Technion ECE |
| Michal Valko | RL & LLMs | Isara Labs, Inria & MVA |
| Marcello Restelli | Real-world RL | Polytechnic University of Milan |
| Kelly Zhang | RL for Health | Imperial College London |
| Georgia Chalvatzaki | Robotics | Technische Universität Darmstadt |
| Antonin Raffin | Deep RL I & II (practical) | German Aerospace Center |

---

## 🗂️ Repository Structure

```
rl-summer-school-2026/
│
├── lectures/              # Notes + slides per lecture
│   ├── material/
│   └── notes/             # .md file for each lecture
│
├── tutorials/             # Hands-on sessions & notebooks
│
├── src/                   # Source code for tutorials
│   └── RL/                # Main package 
│
├── final-quiz/
│   └── prep-notes.md
│
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---
## Installation
To set up the environment for the RL Summer School tutorials, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone <REPOSITORY_URL>
   cd rl-summer-school-2026
   ```

2. **Create and activate a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

3. **Install Required Packages**:
   ```bash
    pip install -r requirements.txt
    ```
4. **Install the RL Package**:
    ```bash
    pip install -e .
    ```
---


## 📝 Notes Template

Each `notes.md` file follows this structure:

```markdown
# [Topic] — [Speaker], [Date]

## Summary
_overview of the lecture._


## Papers & References
- ...
```

---


<!-- ---

## 🏷️ Legend

| Color | Type |
|-------|------|
| 🟩 Green | Lectures |
| 🟪 Purple | Practical sessions |
| 🟥 Red/Pink | Poster sessions |
| 🟨 Yellow | Final quiz |

--- -->

## 📌 Notes

- All notes are personal and may contain errors — always defer to official materials.
- Slides are included only when shared publicly by the speaker.
---

*Summer School — June 2026*
