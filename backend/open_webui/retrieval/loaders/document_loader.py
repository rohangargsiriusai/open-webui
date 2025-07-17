from llama_parse import LlamaParse
from llama_index.core.schema import Document as LlamaIndexDocument  # avoid naming confusion
from langchain_core.documents import Document as LangchainDocument
import os

print("LLAMA_CLOUD_API_KEY is:", os.environ.get("LLAMA_CLOUD_API_KEY"))

class DocumentLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        # Initialize LlamaParse in Balanced Mode (parse with LLM)
        print("Load function called")
        parser = LlamaParse(
            api_key=os.environ["LLAMA_CLOUD_API_KEY"],  # must be set in env
            result_type="markdown",                    # or "text"/"json"
            parse_mode="parse_page_with_llm",          # Balanced Mode {for FAST: fast_mode=True, for PREMIUM: parse_mode="parse_page_with_lvm" }
            verbose=True,
            num_workers=4,
        )

        # Parse the document (sync)
        documents = parser.load_data(self.file_path)

        # Convert each LlamaIndexDocument to LangchainDocument
        result = []
        for doc in documents:
            page_content = getattr(doc, "text", None)
            if page_content is None:
                page_content = getattr(doc, "page_content", "")
            metadata = getattr(doc, "metadata", {})
            metadata["source"] = self.file_path
            metadata["parser"] = "llamaparse"
            metadata["parse_mode"] = "balanced"
            result.append(
                LangchainDocument(page_content=page_content, metadata=metadata)
            )
        return result
