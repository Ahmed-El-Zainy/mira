# Design Spec: Mira "Ghost in the Machine" HTML Demo

**Date:** 2026-05-09
**Status:** Draft
**Topic:** Modern HTML Demo for Market Intelligence & Research Agent (Mira)

---

## 1. Overview
A high-fidelity, dynamic HTML demo that visualizes the Mira agent's reasoning and final analysis. It focuses on the "Agent's Mind" concept, contrasting raw technical logs with a polished financial report.

## 2. Visual & Interaction Design
- **Theme:** Cyber-noir / Dark Mode.
- **Color Palette:**
  - Background: `#0a0a0b` (Obsidian)
  - Primary (Terminal): `#00ff41` (Matrix Green)
  - Secondary (Accents): `#00d4ff` (Neon Sapphire)
  - Warning/Negative: `#ff3131` (Pulse Red)
- **Layout:** 
  - **Left Pane (40%):** "The Neural Link" - A scrolling terminal-style log feed.
  - **Right Pane (60%):** "The Analyst Output" - A structured, widget-based dashboard.

## 3. Component Specifications

### 3.1 Neural Link (Terminal)
- **Animation:** Text streams in with a slight flicker and random delays to simulate thought.
- **Content:** Pulls from the `/logs/{job_id}` endpoint.
- **Styling:** Monospace font, glowing text effect, scroll-to-bottom behavior.

### 3.2 Analyst Output (Dashboard)
- **Header:** Ticker, Company Name, and Current Price (with pulse animation on change).
- **Sentiment Dial:** A semi-circle gauge (Chart.js) showing the sentiment score from -1.0 to 1.0.
- **Market Snapshot:** 4-grid card layout for P/E Ratio, Market Cap, and 52-week range.
- **Correlation Heatmap:** A visual grid showing peer correlations (TSLA vs peers).
- **Key Findings:** A list that "types out" character-by-character when the report is finalized.
- **Citations:** A clean list of source links at the bottom.

## 4. Technical Architecture
- **Frontend:** Single-file HTML/JS for simplicity and portability.
  - **CSS:** Tailwind CSS via CDN.
  - **Charts:** Chart.js for data visualization.
  - **Icons:** Lucide-icons or similar via CDN.
- **Data Handling:**
  - `demo.html` will contain a "Demo Mode" toggle.
  - **Live Mode:** Polls the locally running Mira API (`localhost:8000`).
  - **Mock Mode:** Uses the existing `sample_output.json` to showcase the UI immediately without requiring a running backend.

## 5. User Flow
1. User opens `demo.html`.
2. Initial "Handshake" animation (Terminal starts scrolling).
3. Dashboard components fade in as "data is processed".
4. Final report "Synthesised" and displayed with typewriter effects.

## 6. Self-Review
- **Placeholders:** All data points mapped to `sample_output.json` fields.
- **Consistency:** Cyber-noir theme maintained across both panes.
- **Scope:** Focused on visualization; no new backend logic required.
- **Ambiguity:** Defined both Live and Mock modes to ensure the demo always works.
