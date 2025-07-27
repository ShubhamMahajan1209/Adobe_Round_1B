import os
import json
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import torch
import fitz  # PyMuPDF
from collections import Counter
import statistics

# --- Part 1: PDF Extraction Functions (Upgraded Logic) ---

def get_font_stats(page):
    """
    Calculates the median font size and the dominant font name for that size.
    """
    fonts = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                if "spans" in line:
                    for span in line["spans"]:
                        fonts.append((span["size"], span["font"]))
    
    if not fonts:
        return None, None

    font_sizes = [f[0] for f in fonts]
    # Use median for a more robust baseline against outliers
    median_size = statistics.median(font_sizes)
    
    # Find the most common font name at or near the median size
    dominant_fonts = [f[1] for f in fonts if abs(f[0] - median_size) < 0.1]
    if not dominant_fonts: # Fallback if no fonts exactly at median
        dominant_font_name = Counter([f[1] for f in fonts]).most_common(1)[0][0]
    else:
        dominant_font_name = Counter(dominant_fonts).most_common(1)[0][0]

    return median_size, dominant_font_name


def is_heading(span, line_text, median_size, dominant_font):
    """
    Determines if a text span is a heading using stricter font and structure rules.
    """
    # Rule 1: Structural checks. Must be short and not look like a sentence.
    if not line_text or len(line_text.split()) > 15 or line_text.endswith(('.', '?', '!')):
        return False

    # Rule 2: Case check. Must look like a title, not a regular sentence.
    words = line_text.split()
    if len(words) > 2:
        # A simple check for sentence case (most words are lowercase)
        if sum(1 for word in words if word.islower()) > len(words) / 2:
            return False

    # Rule 3: Font property checks. Must be significantly larger OR bold.
    is_larger = span["size"] > (median_size * 1.2) # Must be at least 20% larger than median
    is_bold = "bold" in span["font"].lower() and not ("bold" in dominant_font.lower())
    
    return is_larger or is_bold

def extract_pdf_data(pdf_directory_path):
    """
    Extracts and cleans text from all PDFs, using advanced font analysis for heading detection.
    """
    all_pdfs_data = {}
    if not os.path.isdir(pdf_directory_path):
        print(f"  Error: PDF directory not found at '{pdf_directory_path}'")
        return None

    for filename in os.listdir(pdf_directory_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory_path, filename)
            print(f"  - Processing {filename}...")
            try:
                doc = fitz.open(pdf_path)
                file_data = []
                for page_num, page in enumerate(doc):
                    raw_page_text = page.get_text("text").strip()
                    
                    heading = ""
                    median_size, dominant_font = get_font_stats(page)

                    if median_size:
                        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH)["blocks"]
                        for block in blocks:
                            if "lines" in block:
                                for line in block["lines"]:
                                    line_text = "".join([s["text"] for s in line["spans"]]).strip()
                                    if line_text and "spans" in line and line['spans']:
                                        first_span = line["spans"][0]
                                        if is_heading(first_span, line_text, median_size, dominant_font):
                                            if not heading:
                                                heading = line_text
                    
                    text_normalized = raw_page_text.replace('ﬀ', 'ff')
                    text_no_bullets = re.sub(r'[\*•–-]', '', text_normalized)
                    cleaned_page_text = re.sub(r'\s+', ' ', text_no_bullets).strip()
                    
                    if heading and cleaned_page_text.startswith(heading):
                        cleaned_page_text = cleaned_page_text[len(heading):].lstrip()
                    
                    page_info = {
                        "heading": heading,
                        "page_number": page_num + 1,
                        "details": cleaned_page_text
                    }
                    file_data.append(page_info)
                all_pdfs_data[filename] = file_data
            except Exception as e:
                print(f"    Could not process {filename}. Error: {e}")
    
    return all_pdfs_data

# --- Part 2: Content Analysis Function (MODIFIED) ---

def analyze_and_structure_content(task_input_path, content_data, output_json_path):
    """
    Analyzes content with a two-stage process ensuring no overlap between outputs:
    1. Ranks pages by heading relevance for 'extracted_sections'.
    2. Ranks sentences from REMAINING pages for 'subsection_analysis' and expands context.
    """
    print("  Loading task data for analysis...")
    try:
        with open(task_input_path, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
    except FileNotFoundError as e:
        print(f"    Error: Task input file not found. {e}")
        return

    job_to_be_done = task_data['job_to_be_done']['task']
    print(f"  Using query for analysis: '{job_to_be_done}'")

    # --- OFFLINE MODEL LOADING ---
    # Load the model from the local cache instead of downloading it.
    local_model_path = './model_cache/all-mpnet-base-v2'
    print(f"  Loading sentence-transformer model from local path: {local_model_path}...")
    if not os.path.isdir(local_model_path):
        print(f"  Error: Model not found at '{local_model_path}'.")
        print("  Please run the 'download_model.py' script first with an internet connection.")
        return
        
    model = SentenceTransformer(local_model_path)
    query_embedding = model.encode(job_to_be_done, convert_to_tensor=True)

    all_sections = []
    for doc_name, pages in content_data.items():
        for page in pages:
            if page.get('details', '').strip() and page.get('heading', ''):
                all_sections.append({
                    "document": doc_name, "section_title": page['heading'],
                    "page_number": page['page_number'], "details": page['details'],
                })

    if not all_sections:
        print("    Error: No valid sections with headings found after filtering.")
        return
        
    # --- STAGE 1: Find Top Pages by HEADING Relevance for 'extracted_sections' ---
    print(f"  Stage 1: Analyzing {len(all_sections)} section headings...")
    
    headings = [section['section_title'] for section in all_sections]
    heading_embeddings = model.encode(headings, convert_to_tensor=True)
    heading_relevance_scores = util.cos_sim(query_embedding, heading_embeddings)[0]

    top_k = 5
    lambda_param = 0.5
    top_relevant_idx = torch.topk(heading_relevance_scores, k=1).indices[0].item()
    selected_heading_indices = {top_relevant_idx}
    candidate_indices = [i for i in range(len(all_sections)) if i != top_relevant_idx]

    for _ in range(top_k - 1):
        if not candidate_indices: break
        candidate_relevance = heading_relevance_scores[candidate_indices]
        selected_embeddings = heading_embeddings[list(selected_heading_indices)]
        candidate_embeddings_tensor = heading_embeddings[candidate_indices]
        similarity_to_selected = util.cos_sim(candidate_embeddings_tensor, selected_embeddings)
        max_similarity = torch.max(similarity_to_selected, dim=1).values
        mmr_scores = lambda_param * candidate_relevance - (1 - lambda_param) * max_similarity
        best_candidate_local_idx = torch.argmax(mmr_scores).item()
        best_candidate_global_idx = candidate_indices.pop(best_candidate_local_idx)
        selected_heading_indices.add(best_candidate_global_idx)
    
    top_sections_by_heading = [all_sections[i] for i in selected_heading_indices]

    # --- STAGE 2: Find Top SENTENCES from REMAINING pages for 'subsection_analysis' ---
    print("  Stage 2: Analyzing sentences from remaining pages...")
    
    page_to_sentences = {}
    for i, section in enumerate(all_sections):
        if i not in selected_heading_indices:
            page_key = (section["document"], section["page_number"])
            if page_key not in page_to_sentences:
                sentences = re.split(r'(?<=[.!?])\s+', section['details'])
                cleaned_sentences = [s.strip() for s in sentences if s.strip()]
                if cleaned_sentences:
                    page_to_sentences[page_key] = cleaned_sentences

    candidate_sentences = []
    for page_key, sentences in page_to_sentences.items():
        for idx, sentence_text in enumerate(sentences):
            candidate_sentences.append({
                "text": sentence_text,
                "page_key": page_key,
                "sentence_index": idx
            })
    
    subsection_analysis_output = []
    if candidate_sentences:
        sentence_texts = [s['text'] for s in candidate_sentences]
        sentence_embeddings = model.encode(sentence_texts, convert_to_tensor=True)
        sentence_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]

        top_sentence_indices = torch.topk(sentence_scores, k=min(20, len(candidate_sentences))).indices

        used_pages = set()
        for idx in top_sentence_indices:
            if len(subsection_analysis_output) >= 5:
                break

            sentence_info = candidate_sentences[idx.item()]
            page_key = sentence_info["page_key"]

            if page_key not in used_pages:
                all_page_sentences = page_to_sentences[page_key]
                current_sentence_index = sentence_info["sentence_index"]
                
                start_index = max(0, current_sentence_index - 1)
                end_index = min(len(all_page_sentences), current_sentence_index + 2)
                
                context_snippet = " ".join(all_page_sentences[start_index:end_index])
                
                subsection_analysis_output.append({
                    "document": page_key[0],
                    "refined_text": context_snippet,
                    "page_number": page_key[1]
                })
                used_pages.add(page_key)
    else:
        print("    Warning: No candidate sentences found for subsection analysis.")


    # --- Construct Final JSON ---
    final_output = {
        "metadata": {
            "input_documents": list(content_data.keys()), "persona": task_data['persona']['role'],
            "job_to_be_done": job_to_be_done, "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {"document": s["document"], "section_title": s["section_title"], "importance_rank": i + 1, "page_number": s["page_number"]}
            for i, s in enumerate(top_sections_by_heading)
        ],
        "subsection_analysis": subsection_analysis_output
    }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print(f"  Final analysis complete. Output saved to '{output_json_path}'")

# --- Part 3: Main Execution Block ---

if __name__ == '__main__':
    # These paths are relative to the project root where run.sh is executed
    INPUT_PDF_DIRECTORY = os.path.join('test_case', 'documents')
    TASK_INPUT_FILE = os.path.join('test_case', 'input.json')
    FINAL_OUTPUT_FILE = 'output_test_case.json'

    print("Starting workflow...")
    print("\nStep 1: Extracting text from PDFs...")
    extracted_data = extract_pdf_data(INPUT_PDF_DIRECTORY)

    if extracted_data:
        print("\nStep 2: Analyzing content and generating final plan...")
        analyze_and_structure_content(TASK_INPUT_FILE, extracted_data, FINAL_OUTPUT_FILE)
        print("\nWorkflow complete.")
    else:
        print("\nWorkflow failed: No data was extracted from PDFs.")
