"""
This set of unit tests is designed to test the file operations that autoGPT has access to.
"""

import os
import random
import string
from pathlib import Path
from tempfile import gettempdir

import pytest

from autogpt.commands.file_operations import (
    append_to_file,
    check_duplicate_operation,
    delete_file,
    download_file,
    ingest_file,
    list_files,
    log_operation,
    read_file,
    split_file,
    write_to_file,
)
from autogpt.config import Config
from autogpt.logs import Logger
from autogpt.memory import get_memory
from autogpt.utils import readable_file_size


@pytest.fixture()
def file_content():
    return "This is a test file.\n"


@pytest.fixture()
def test_file(workspace, file_content):
    test_file = str(workspace.get_path("test_file.txt"))
    with open(test_file, "w") as f:
        f.write(file_content)
    return test_file


@pytest.fixture()
def test_directory(workspace):
    return str(workspace.get_path("test_directory"))


@pytest.fixture()
def test_nested_file(workspace):
    return str(workspace.get_path("nested/test_file.txt"))


def test_check_duplicate_operation(config, test_file):
    log_operation("write", test_file)
    assert check_duplicate_operation("write", test_file) is True


# Test logging a file operation
def test_log_operation(test_file, config):
    file_logger_name = config.file_logger_path
    if os.path.exists(file_logger_name):
        os.remove(file_logger_name)

    log_operation("log_test", test_file)
    with open(config.file_logger_path, "r") as f:
        content = f.read()
    assert f"log_test: {test_file}" in content


# Test splitting a file into chunks
def test_split_file():
    content = "abcdefghij"
    chunks = list(split_file(content, max_length=4, overlap=1))
    expected = ["abcd", "defg", "ghij"]
    assert chunks == expected


def test_read_file(test_file, file_content):
    content = read_file(test_file)
    assert content == file_content


def test_write_to_file(config, test_nested_file):
    new_content = "This is new content.\n"
    write_to_file(test_nested_file, new_content)
    with open(test_nested_file, "r") as f:
        content = f.read()
    assert content == new_content


def test_append_to_file(test_nested_file):
    append_text = "This is appended text.\n"
    write_to_file(test_nested_file, append_text)

    append_to_file(test_nested_file, append_text)

    with open(test_nested_file, "r") as f:
        content_after = f.read()

    assert content_after == append_text + append_text


def test_delete_file(config, test_file):
    delete_file(test_file)
    assert os.path.exists(test_file) is False
    assert delete_file(test_file) == "Error: File has already been deleted."


def test_delete_missing_file(test_file):
    os.remove(test_file)
    try:
        os.remove(test_file)
    except FileNotFoundError as e:
        error_string = str(e)
        assert error_string in delete_file(test_file)
        return
    assert True, "Failed to test delete_file"


def test_list_files(config, workspace, test_directory):
    # Case 1: Create files A and B, search for A, and ensure we don't return A and B
    file_a = workspace.get_path("file_a.txt")
    file_b = workspace.get_path("file_b.txt")

    with open(file_a, "w") as f:
        f.write("This is file A.")

    with open(file_b, "w") as f:
        f.write("This is file B.")

    # Create a subdirectory and place a copy of file_a in it
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    with open(os.path.join(test_directory, file_a.name), "w") as f:
        f.write("This is file A in the subdirectory.")

    files = list_files(str(workspace.root))
    assert file_a.name in files
    assert file_b.name in files
    assert os.path.join(Path(test_directory).name, file_a.name) in files

    # Clean up
    os.remove(file_a)
    os.remove(file_b)
    os.remove(os.path.join(test_directory, file_a.name))
    os.rmdir(test_directory)

    # Case 2: Search for a file that does not exist and make sure we don't throw
    non_existent_file = "non_existent_file.txt"
    files = list_files("")
    assert non_existent_file not in files


def test_download_file():
    url = "https://github.com/Significant-Gravitas/Auto-GPT/archive/refs/tags/v0.2.2.tar.gz"
    local_name = os.path.join(gettempdir(), "auto-gpt.tar.gz")
    size = 365023
    readable_size = readable_file_size(size)
    assert (
        download_file(url, local_name)
        == f'Successfully downloaded and locally stored file: "{local_name}"! (Size: {readable_size})'
    )
    assert os.path.isfile(local_name) is True
    assert os.path.getsize(local_name) == size

    url = "https://github.com/Significant-Gravitas/Auto-GPT/archive/refs/tags/v0.0.0.tar.gz"
    assert "Got an HTTP Error whilst trying to download file" in download_file(
        url, local_name
    )

    url = "https://thiswebsiteiswrong.hmm/v0.0.0.tar.gz"
    assert "Failed to establish a new connection:" in download_file(url, local_name)


@pytest.fixture
def ingest_config():
    class IngestConfig:
        chunks_cnt = 5
        max_len = 10
        overlap = 0
        filename = "file_to_ingest.txt"
        length = chunks_cnt * max_len

    return IngestConfig()


@pytest.fixture
def file_content(ingest_config):
    random.seed(42)
    return "".join(random.choices(string.ascii_letters, k=ingest_config.length))


@pytest.fixture
def file_to_ingest(workspace, ingest_config, file_content):
    file = workspace.get_path(ingest_config.filename)
    with open(file, "w") as f:
        f.write(file_content)
    return file


def test_ingest_file(config, file_to_ingest, ingest_config, file_content, mocker):
    memory = get_memory(config, True)
    mock_logger = mocker.patch.object(Logger, "info")
    mock_memory = mocker.patch.object(memory, "add")

    ingest_file(
        str(file_to_ingest), memory, ingest_config.max_len, ingest_config.overlap
    )

    expected_log_calls = (
        [
            mocker.call(f"Working with file {file_to_ingest}"),
            mocker.call(f"File length: {ingest_config.length} characters"),
        ]
        + [
            mocker.call(
                f"Ingesting chunk {i + 1} / {ingest_config.chunks_cnt} into memory"
            )
            for i in range(ingest_config.chunks_cnt)
        ]
        + [
            mocker.call(
                f"Done ingesting {ingest_config.chunks_cnt} chunks from {file_to_ingest}."
            ),
        ]
    )

    actual_log_calls = mock_logger.call_args_list
    assert actual_log_calls == expected_log_calls

    chunks = list(
        split_file(
            file_content,
            max_length=ingest_config.max_len,
            overlap=ingest_config.overlap,
        )
    )

    expected_memory_calls = [
        mocker.call(
            f"Filename: {file_to_ingest}\n"
            f"Content part#{i + 1}/{ingest_config.chunks_cnt}: {chunk}"
        )
        for i, chunk in enumerate(chunks)
    ]

    actual_memory_calls = mock_memory.call_args_list
    assert actual_memory_calls == expected_memory_calls


def test_ingest_file_error(config, file_to_ingest, ingest_config, mocker):
    mock_logger = mocker.patch.object(Logger, "info")

    ingest_file(str(file_to_ingest), None, ingest_config.max_len, ingest_config.overlap)

    expected_calls = [
        mocker.call(f"Working with file {file_to_ingest}"),
        mocker.call(f"File length: {ingest_config.length} characters"),
        mocker.call("Ingesting chunk 1 / 5 into memory"),
        mocker.call(
            f"Error while ingesting file '{file_to_ingest}': 'NoneType' object has no attribute 'add'"
        ),
    ]

    actual_calls = mock_logger.call_args_list
    assert actual_calls == expected_calls
