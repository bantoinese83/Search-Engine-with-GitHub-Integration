# Search Engine with GitHub Integration

This project is a search engine that integrates with GitHub repositories. It clones repositories, extracts content, and uses a FAISS vectorstore for efficient document retrieval. Additionally, it leverages the Gemini API to generate answers to queries based on the retrieved documents.

## Features

- **Clone GitHub Repositories**: Clone repositories and upload them to an S3 bucket.
- **Extract Repository Content**: Extract relevant content from cloned repositories.
- **Create FAISS Vectorstore**: Create a FAISS vectorstore for efficient document retrieval.
- **Search and Generate Answers**: Search the vectorstore and generate answers using the Gemini API.
- **Git Integration**: Capture the state of the repository before and after changes.


## Requirements

- Python 3.7+
- boto3
- botocore
- google-generativeai
- langchain
- langchain-community
- langchain-huggingface
- sentence-transformers
- tqdm

## Configuration

Create a `config.py` file with the following variables:

```python
MODEL_NAME = "your_model_name"
GEMINI_API_KEY = "your_gemini_api_key"
AWS_ACCESS_KEY_ID = "your_aws_access_key_id"
AWS_SECRET_ACCESS_KEY = "your_aws_secret_access_key"
```
