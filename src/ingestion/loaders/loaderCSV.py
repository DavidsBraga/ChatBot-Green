import csv
from src.ingestion.loaders.loaderBase import LoaderBase

class LoaderCSV(LoaderBase):

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.products = self._load_products()

    def _load_products(self) -> list:
        """Loads all products from the CSV into a list of dicts."""
        products = []
        with open(self.filepath, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                products.append(row)
        return products

    def extract_text(self) -> str:
        """Returns all products as a readable text block."""
        lines = []
        for p in self.products:
            lines.append(
                f"Product: {p['Product Name']} | Category: {p['Category']} | "
                f"Price: ${p['Price ($)']} | Score: {p['Review Score']} | "
                f"Colors: {p['Colors']} | Countries: {p['Countries']} | "
                f"Description: {p['Description']}"
            )
        return "\n".join(lines)

    def extract_metadata(self):
        return {"total_products": len(self.products)}

    def search_products(self, query: str, top_n: int = 3) -> list:
        """Simple keyword search over product fields."""
        query_words = query.lower().split()
        scored = []
        for p in self.products:
            searchable = (
                f"{p['Product Name']} {p['Category']} {p['Description']} {p['Colors']}"
            ).lower()
            score = sum(1 for word in query_words if word in searchable)
            if score > 0:
                scored.append((score, p))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [p for _, p in scored[:top_n]]

    def format_product_card(self, product: dict) -> str:
        """Formats a product as a terminal-friendly card."""
        return (
            f"\n {product['Product Name']}\n"
            f"     Category : {product['Category']}\n"
            f"     Price    : ${product['Price ($)']}\n"
            f"     Score    : {product['Review Score']}\n"
            f"     Colors   : {product['Colors']}\n"
            f"     Ships to : {product['Countries']}\n"
            f"     {product['Description']}\n"
        )

    def all_keys_have_values(self, metadata, value_check=lambda x: x is not None and x != ''):
        return all(value_check(value) for value in metadata.values())