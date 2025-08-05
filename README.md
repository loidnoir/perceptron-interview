# Perceptron Interview Project

## Instruction

- Anaconda or Miniconda installed
- Python `3.11+`

### Installation with Conda

- **Create a new conda environment**

    ```bash
    conda create -n perceptron python=3.11
    ```

- **Activate the environment**

    ```bash
    conda activate perceptron
    ```

- **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

- **Set up environment variables**

    ```bash
    cp .env.example .env
    # Edit .env and add your credentials
    ```

---

### Alternative Installation

```bash
# Create and activate environment
conda create -n perceptron python=3.11
conda activate perceptron

# Install conda packages
conda install pandas numpy pillow scikit-image requests

# Install remaining packages via pip
pip install openai python-dotenv pydantic pydantic-extra-types shapely zstandard
```

## Example

Given `.data/whatisthisthing_submissions.zst` and `.data/whatisthisthing_comments.zst` files in the root directory, we can process the data into CSV files.

### 1. **Processing data**

Process raw zst reddit data files into organized, cleaned CSV files with only required fields and data. The processing is flexible with options to set limitations.

- Process submissions

    ```bash
    python src/processor.py \
        .data/whatisthisthing_submissions.zst \
        .processed/sub.csv \
        --comment-limit=20 \
        --reply-limit=5
    ```

- Process comments

    ```bash
    python src/processor.py \
        .data/whatisthisthing_comments.zst \
        .processed/com.csv \
        --comment-limit=20 \
        --reply-limit=5
    ```

- `.processed/sub.csv` and `processed/com.csv` will be created in the `.processed` directory.

---

### 2. **Generating threads**

Generation script takes submissions and comments from our processed files and using LLM generates a thread analysis with all the content processed ready to be converted to our data structures.

```bash
python src/generator.py \
    .processed/sub.csv \
    .processed/com.csv \
    .generations/ \
    --limit=20
```

- `.generations/thread-XXXXXXXX-XXXXXX.csv` will be created in the `generations` directory.

---

### 3. **Formatting results**

Formatting script takes the generated thread analysis and converts it to our data structures.

```bash
python src/formatter.py \
    .generations/thread-XXXXXXXX-XXXXXX.csv \
    samples/
```

- `samples/` directory will be filled with the generated cases.
