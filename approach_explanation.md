# Project Methodology and Approach

This document outlines the technical approach, architecture, and design decisions behind the content analysis and planning tool. Our goal is to intelligently extract relevant information from a collection of PDF documents based on a user-defined task, and structure it into a coherent plan.

---

### 1. Core Methodology

The workflow is a two-part process designed to first identify high-level concepts and then drill down into specific, relevant details, ensuring no overlap between the outputs.

#### **Part 1: Structural PDF Content Extraction**

The first challenge is to extract not just text, but also the document's inherent structure. Simply pulling raw text loses the distinction between headings and body paragraphs. Our approach reconstructs this structure using font analysis.

1.  **Library:** We use `PyMuPDF (fitz)` for its speed and detailed access to document elements.
2.  **Font Statistics (`get_font_stats`):** For each page, we calculate the **median font size** and the **dominant font family**. The median is used instead of the mean to create a robust baseline that is not skewed by a few very large or small font sizes (e.g., a massive title or tiny footer text).
3.  **Heading Detection (`is_heading`):** A line of text is identified as a heading if it meets a strict set of criteria:
    - **Structural Rules:** It must be relatively short (under 15 words) and not end with sentence-terminating punctuation.
    - **Case Rules:** It must resemble a title (e.g., Title Case) rather than a regular sentence (Sentence case).
    - **Font Properties:** It must be stylistically distinct. It qualifies if it is either **significantly larger** (at least 20% larger than the median size) OR **bold**, especially if the main body text is not bold.

This heuristic approach allows us to reliably identify the primary heading on each page, which serves as a crucial anchor for semantic analysis.

#### **Part 2: Two-Stage Semantic Analysis**

Once the content is extracted and structured, we perform a two-stage analysis using the `sentence-transformers` library with the `all-mpnet-base-v2` model. This model excels at creating semantically rich vector embeddings for sentences and paragraphs.

**Stage 1: High-Level Section Identification (MMR)**

- **Goal:** Find the top 5 most relevant _sections_ from all documents that give a broad overview of the topic.
- **Process:**
  1.  We create vector embeddings for the user's query and for every _heading_ extracted in Part 1.
  2.  We use **Maximal Marginal Relevance (MMR)** to select the top sections. MMR is a diversification algorithm that optimizes for two things simultaneously:
      - **Relevance:** How similar is a section's heading to the user's query?
      - **Diversity:** How different is a section from the ones already selected?
  3.  This prevents the output from containing five very similar sections. For example, if the query is "beach vacation," MMR will avoid selecting five sections all titled "The Best Beaches," and will instead try to find related but distinct topics like "Beach Activities," "Coastal Dining," etc.
- **Output:** This stage produces the `extracted_sections` part of the final JSON, containing the most important and diverse high-level topics.

**Stage 2: Detailed Subsection Analysis**

- **Goal:** Extract specific, granular, and highly relevant sentences that act as supporting details.
- **Process:**
  1.  To avoid redundancy, we **exclude all pages** that were already selected in Stage 1.
  2.  From the remaining pool of pages, we break down the content into individual sentences.
  3.  We encode all these sentences and compute their cosine similarity to the user's query.
  4.  We identify the top 20 most relevant sentences across all remaining documents.
  5.  To provide better context, we don't just return the single best sentence from a page. Instead, we retrieve a **three-sentence snippet** (the relevant sentence plus the one before and after it).
  6.  To ensure a variety of sources, we only include one such snippet per page, selecting the top 5 unique pages.
- **Output:** This produces the `subsection_analysis` part of the JSON, offering focused, contextualized details that complement the high-level sections.

---

### 2. Repository Structure

The project is organized for clarity, portability, and ease of execution.

```
your_project_name/
│
├── test_case/              # Contains input data for a sample run.
│   ├── documents/
│   └── input.json
│
├── model_cache/            # Local cache for the downloaded ML model (for offline use).
│
├── src/
│   └── main.py             # The core Python script with all the logic.
│
├── download_model.py       # Script to pre-download the model for offline use.
├── Dockerfile              # Instructions to build a portable Docker container.
├── requirements.txt        # Lists all Python dependencies.
└── run.sh                  # The main execution script.
```

---

### 3. Limitations

- **Heading Detection:** The heuristic-based heading detection works well for many documents but may fail on PDFs with non-standard formatting or those that are image-based (scanned documents).
- **Language Support:** The `all-mpnet-base-v2` model is optimized for English. Its performance will be significantly lower for content in other languages.
- **Complex Layouts:** The current implementation processes text linearly. It does not explicitly handle complex layouts like multi-column text, tables, or figures, which could lead to jumbled text extraction.
- **Query Sensitivity:** The quality of the output is highly dependent on the quality and specificity of the input `job_to_be_done` query. Vague queries will yield generic results.

---

### 4. Future Work

- **OCR Integration:** Incorporate an Optical Character Recognition (OCR) engine like Tesseract to process scanned PDFs, making the tool more robust.
- **Advanced Layout Analysis:** Integrate a more sophisticated document layout analysis model (potentially a computer vision model) to correctly parse tables, columns, and other visual elements.
- **LLM-Powered Summarization:** Instead of extracting raw text snippets, the extracted content could be fed to a Large Language Model (LLM) to generate more natural, human-readable summaries for each section.
- **Interactive Frontend:** Develop a simple web interface where a user can upload PDFs, enter a query, and view the structured output in a user-friendly format.
