import unittest
from unittest.mock import patch, MagicMock

import botocore

from search_engine import (
    create_bucket_if_not_exists,
    upload_to_s3,
    download_from_s3,
    clone_repository,
    extract_repository_content,
    load_repositories,
    create_faiss_vectorstore,
    search_and_generate_answer,
    S3_BUCKET_NAME
)
from langchain.schema import Document

class TestSearchEngine(unittest.TestCase):

    @patch('search_engine.s3_client')
    def test_create_bucket_if_not_exists(self, mock_s3_client):
        mock_s3_client.create_bucket.side_effect = botocore.exceptions.ClientError(
            {'Error': {'Code': 'BucketAlreadyOwnedByYou'}}, 'CreateBucket'
        )
        create_bucket_if_not_exists(S3_BUCKET_NAME)
        mock_s3_client.create_bucket.assert_called_once()

    @patch('search_engine.s3_client')
    def test_upload_to_s3(self, mock_s3_client):
        upload_to_s3('local_path', 's3_path')
        mock_s3_client.upload_file.assert_called_once_with('local_path', S3_BUCKET_NAME, 's3_path')

    @patch('search_engine.s3_client')
    def test_download_from_s3(self, mock_s3_client):
        download_from_s3('s3_path', 'local_path')
        mock_s3_client.download_file.assert_called_once_with(S3_BUCKET_NAME, 's3_path', 'local_path')

    @patch('search_engine.subprocess.run')
    @patch('search_engine.upload_to_s3')
    @patch('search_engine.shutil.make_archive')
    def test_clone_repository(self, mock_make_archive, mock_upload_to_s3, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock()
        mock_make_archive.return_value = 'zip_path'
        repo_path = clone_repository('https://github.com/user/repo.git', 'clone_directory')
        self.assertEqual(repo_path, 'clone_directory/repo')
        mock_subprocess_run.assert_called_once_with(['git', 'clone', 'https://github.com/user/repo.git', 'clone_directory/repo'], check=True)
        mock_upload_to_s3.assert_called_once_with('zip_path', 'repos/repo.zip')

    def test_extract_repository_content(self):
        with patch('builtins.open', unittest.mock.mock_open(read_data='file content')):
            documents = extract_repository_content('repo_path')
            self.assertIsInstance(documents, list)
            self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    @patch('search_engine.clone_repository')
    @patch('search_engine.extract_repository_content')
    def test_load_repositories(self, mock_extract_repository_content, mock_clone_repository):
        mock_clone_repository.return_value = 'repo_path'
        mock_extract_repository_content.return_value = [Document(page_content='content', metadata={})]
        documents = load_repositories(['https://github.com/user/repo.git'], 'clone_directory')
        self.assertIsInstance(documents, list)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

    @patch('search_engine.HuggingFaceEmbeddings')
    @patch('search_engine.FAISS')
    def test_create_faiss_vectorstore(self, mock_faiss, mock_hf_embeddings):
        mock_hf_embeddings.return_value = MagicMock()
        mock_faiss.from_documents.return_value = MagicMock()
        documents = [Document(page_content='content', metadata={})]
        vectorstore = create_faiss_vectorstore(documents)
        self.assertIsNotNone(vectorstore)
        mock_faiss.from_documents.assert_called_once()

    @patch('search_engine.is_git_repository', return_value=True)
    @patch('search_engine.subprocess.run')
    @patch('search_engine.genai.GenerativeModel')
    @patch('search_engine.FAISS.as_retriever')
    def test_search_and_generate_answer(self, mock_as_retriever, mock_generative_model, mock_subprocess_run, mock_is_git_repository):
        mock_as_retriever.return_value.invoke.return_value = [Document(page_content='content', metadata={})]
        mock_generative_model.return_value.start_chat.return_value.send_message.return_value = MagicMock(text='generated answer')
        mock_subprocess_run.return_value = MagicMock(stdout='diff output')
        vectorstore = MagicMock()
        result = search_and_generate_answer(vectorstore, 'query')
        self.assertIn('answer', result)
        self.assertIn('before_changes', result)
        self.assertIn('after_changes', result)

if __name__ == '__main__':
    unittest.main()