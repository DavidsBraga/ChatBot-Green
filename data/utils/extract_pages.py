import PyPDF2

def extract_pages(pdf_path, output_path, page_ranges):
    """Extracts specified pages from a PDF and saves them to a new PDF.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_path (str): The path to the output PDF file.
        page_ranges (list): A list of tuples specifying the page ranges to extract.
                            Each tuple is of the form (start, end), where start
                            and end are 1-based page numbers.

    Returns:
        None
    """

    pdf_reader = PyPDF2.PdfReader(pdf_path)
    pdf_writer = PyPDF2.PdfWriter()

    for start, end in page_ranges:
        for page_num in range(start - 1, end):  # Page numbers are 0-based
            page = pdf_reader.pages[page_num]
            pdf_writer.add_page(page)

    with open(output_path, 'wb') as output_pdf:
        pdf_writer.write(output_pdf)

    print(f"Extracted pages written to {output_path}")

# Example usage:

pdf_path = "../ghg-protocol-revised.pdf"
output_path = "../ghg-protocol-revised-intro.pdf"
page_ranges = [(4, 7)]  # Extracts pages 4-7

extract_pages(pdf_path, output_path, page_ranges)

pdf_path = "../ghg-protocol-revised.pdf"
output_path = "../ghg-protocol-revised-chapter1.pdf"
page_ranges = [(8, 11)]  # Extracts pages 8-11

extract_pages(pdf_path, output_path, page_ranges)

pdf_path = "../ghg-protocol-revised.pdf"
output_path = "../ghg-protocol-revised-glossary.pdf"
page_ranges = [(98, 104)]  # Extracts pages 98-104

extract_pages(pdf_path, output_path, page_ranges)
