import logging
import os
import shutil
import subprocess
from typing import List, Dict

import boto3
import botocore
import google.generativeai as genai
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Import configuration
from config import MODEL_NAME, GEMINI_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

# Initialize the search engine's brain
embedding_model = SentenceTransformer(MODEL_NAME)
genai.configure(api_key=GEMINI_API_KEY)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)


def create_bucket_if_not_exists(bucket_name: str):
    """Create an S3 bucket if it does not exist.

    Args:
        bucket_name (str): The name of the bucket to create.
    """
    try:
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': boto3.session.Session().region_name}
        )
        logging.info(f"Bucket '{bucket_name}' created.")
    except botocore.exceptions.ClientError as error:
        if error.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            logging.info(f"Bucket '{bucket_name}' already exists.")
        else:
            logging.error(f"Error creating bucket: {error}")
            raise


# Fixed bucket name
S3_BUCKET_NAME = "search-engine-github-bucket-s3"

# Create the bucket if it doesn't exist
create_bucket_if_not_exists(S3_BUCKET_NAME)


def upload_to_s3(local_path: str, s3_path: str):
    """Upload a file to an S3 bucket.

    Args:
        local_path (str): Path to the local file.
        s3_path (str): Path in the S3 bucket.
    """
    s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_path)
    logging.info(f"Uploaded {local_path} to s3://{S3_BUCKET_NAME}/{s3_path}")


def download_from_s3(s3_path: str, local_path: str):
    """Download a file from an S3 bucket.

    Args:
        s3_path (str): Path in the S3 bucket.
        local_path (str): Path to the local file.
    """
    s3_client.download_file(S3_BUCKET_NAME, s3_path, local_path)
    logging.info(f"Downloaded s3://{S3_BUCKET_NAME}/{s3_path} to {local_path}")


def clone_repository(repo_url: str, clone_directory: str) -> str:
    """Clone a GitHub repository to a specified directory.

    Args:
        repo_url (str): URL of the GitHub repository.
        clone_directory (str): Directory where the repository will be cloned.

    Returns:
        str: Path to the cloned repository.
    """
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    repo_path = os.path.join(clone_directory, repo_name)
    if not os.path.exists(repo_path):
        subprocess.run(['git', 'clone', repo_url, repo_path], check=True)
        # Upload to S3 only if the repository doesn't exist locally
        zip_path = shutil.make_archive(repo_path, 'zip', repo_path)
        upload_to_s3(zip_path, f"repos/{repo_name}.zip")
    return repo_path


def extract_repository_content(repo_path: str) -> List[Document]:
    """Extract relevant content from a cloned GitHub repository.

    Args:
        repo_path (str): Path to the cloned repository.

    Returns:
        List[Document]: List of Document objects containing the extracted content.
    """
    documents = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.md', '.py', '.java', '.js', '.txt')):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    documents.append(Document(page_content=content, metadata={"file_path": file_path}))
    return documents


def load_repositories(repo_urls: List[str], clone_directory: str) -> List[Document]:
    """Clone repositories and load their content.

    Args:
        repo_urls (List[str]): List of GitHub repository URLs.
        clone_directory (str): Directory where the repositories will be cloned.

    Returns:
        List[Document]: List of Document objects containing the content of the repositories.
    """
    documents = []
    for repo_url in tqdm(repo_urls, desc="Cloning Repositories"):
        repo_path = clone_repository(repo_url, clone_directory)
        documents.extend(extract_repository_content(repo_path))
    return documents


def create_faiss_vectorstore(documents: List[Document]) -> FAISS:
    """Create a FAISS vectorstore for efficient document retrieval.

    Args:
        documents (List[Document]): List of Document objects to be indexed.

    Returns:
        FAISS: FAISS vectorstore object.
    """
    if not documents:
        raise ValueError("No documents to index. Please check the document loading process.")

    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        return FAISS.from_documents(documents, embeddings)
    except Exception as error:
        logging.error(f"Error creating vectorstore: {error}")
        raise


def is_git_repository() -> bool:
    """Check if the current directory is a Git repository.

    Returns:
        bool: True if the current directory is a Git repository, False otherwise.
    """
    try:
        subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False


def search_and_generate_answer(vectorstore: FAISS, query: str, top_k: int = 5, repo_name: str = None) -> Dict[str, str]:
    """Search the vectorstore and generate an answer using Gemini, including before and after changes.

    Args:
        vectorstore (FAISS): FAISS vectorstore object.
        query (str): Query string.
        top_k (int): Number of top documents to retrieve.
        repo_name (str): Name of the repository.

    Returns:
        Dict[str, str]: Dictionary containing the generated answer and before/after changes.
    """
    try:
        if is_git_repository():
            # Capture the state before changes
            subprocess.run(['git', 'add', '.'], check=True)
            subprocess.run(['git', 'commit', '-m', 'Before changes'], check=True)
            before_changes = subprocess.run(['git', 'show', 'HEAD'], capture_output=True, text=True).stdout
        else:
            before_changes = "Not a git repository."

        retriever = vectorstore.as_retriever(search_type="similarity", k=top_k)
        results = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in results])

        generation_config = genai.GenerationConfig(
            temperature=1,
            top_p=0.95,
            top_k=64,
            max_output_tokens=8192,
            response_mime_type="text/plain",
        )

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(f"Question: {query}\nContext: {context}")

        if hasattr(response, 'text'):
            after_changes = response.text
            if is_git_repository():
                # Create directory if it doesn't exist
                changes_dir = 'generated_changes'
                os.makedirs(changes_dir, exist_ok=True)

                # Save the generated changes with the repository name in the file name
                changes_file_path = os.path.join(changes_dir, f'generated_changes_{repo_name}.txt')
                with open(changes_file_path, 'w') as f:
                    f.write(after_changes)

                subprocess.run(['git', 'add', changes_file_path], check=True)
                subprocess.run(['git', 'commit', '-m', 'After changes'], check=True)
                after_changes_diff = subprocess.run(['git', 'diff', 'HEAD~1', 'HEAD'], capture_output=True,
                                                    text=True).stdout
            else:
                after_changes_diff = "Not a git repository."
            return {
                "answer": after_changes,
                "before_changes": before_changes,
                "after_changes": after_changes_diff
            }
        else:
            return {"answer": "Invalid response format.", "before_changes": before_changes, "after_changes": ""}
    except Exception as error:
        logging.error(f"Error searching and answering: {error}")
        return {"answer": "An error occurred while searching and answering your query.", "before_changes": "",
                "after_changes": ""}


# List of GitHub repository URLs
repository_urls = [
    "https://github.com/bantoinese83/car-rental-system.git",
]

# Directory to clone repositories
clone_directory = "cloned_repos"

# Load repositories and their content
documents = load_repositories(repository_urls, clone_directory)

# Create a FAISS vectorstore for efficient document retrieval
vectorstore = create_faiss_vectorstore(documents)

# Search the vectorstore and generate an answer using Gemini
query = ("Refactor the code to improve execution time and optimize memory usage, while maintaining readability and "
         "functionality. Ensure that the changes are well-documented. Provide a detailed explanation of the changes, "
         "including before and after comparisons.")
answer = search_and_generate_answer(vectorstore, query)
print(f"Answer: {answer['answer']}")
print(f"Before Changes: {answer['before_changes']}")
print(f"After Changes: {answer['after_changes']}")
