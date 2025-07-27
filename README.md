# Content Analysis and Planning Tool

## 1. Overview

This project is a sophisticated pipeline that automates the process of analyzing a collection of PDF documents. Given a specific task or query, it intelligently extracts the most relevant sections and supporting details, and organizes them into a structured JSON output.

The system first identifies key sections by analyzing document headings for relevance and diversity using a Sentence Transformer model. It then drills down into the remaining content to find specific, contextualized sentences that support the main topics, ensuring a comprehensive yet non-redundant result.

## 2. Dependencies

All required Python libraries are listed in the `requirements.txt` file.

- `sentence-transformers`
- `torch`
- `PyMuPDF`

## 3. Execution Instructions

Follow these steps to set up and run the project.

### Prerequisites

- Git
- Python 3.8+
- Docker (Recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/ShubhamMahajan1209/Adobe_Round_1B.git
cd Adobe_Round_1B
```

### Step 2: Download the Model (Offline Setup)

To run the project without an active internet connection (e.g., in a restricted competition environment), you must first download the machine learning model.

**Run this command once while you have an internet connection:**

```bash
python3 download_model.py
```

This will download the `all-mpnet-base-v2` model (approx. 420 MB) and save it to the `model_cache` directory. The main script is pre-configured to load the model from this local path.

### Step 3: Run using Docker (Recommended)

Using Docker is the most reliable way to run the application, as it encapsulates all dependencies and ensures a consistent environment.

1.  **Build the Docker image:**
    _(Make sure you have completed Step 2 first, as this copies the model into the image)._

    ```bash
    docker build -t content-analyzer .
    ```

2.  **Run the container:**
    This will execute the `run.sh` script inside the container, process the files in `test_case/`, and generate `output_test_case.json` in the project's root directory upon completion (the file will appear after the container finishes).

    ```bash
    docker run --rm -v "$(pwd)":/app content-analyzer
    ```

    _(Note: The `-v "$(pwd)":/app` command mounts the current directory into the container, allowing the output file to be saved directly to your host machine.)_

### Step 4: Run Locally (Alternative)

If you prefer not to use Docker, you can run the script directly in a local Python environment.

1.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the execution script:**
    _(Make sure you have completed Step 2 to download the model)._

    ```bash
    bash run.sh
    ```

The output file, `output_test_case.json`, will be created in the root directory of the project.
