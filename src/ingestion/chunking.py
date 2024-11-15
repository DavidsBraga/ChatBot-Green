import html2text
import PyPDF2


def text_to_chunks(text: str, chunk_size:int = 512) -> list[str]:
    """Splits text into chunks of a specified size.

    Args:
        text (str): The text to split into chunks.
        chunk_size (int): The desired size of each chunk (default is 512 words).

    Returns:
        list: A list of text chunks.
    """
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
