import os
import pandas as pd
import json

class Tools:
    def _resolve_uploaded_file_path(self, file_name: str) -> str:
        """
        Internal helper to resolve a file path from a partial name.
        Returns full path or None.
        """
        base_dir = "/app/backend/data/uploads"

        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith(file_name):
                    return os.path.join(root, f)
        return None

    def load_dataset(
        self, question: str, sample_size: int = None, file_name: str = "dataset.csv"
    ):
        """
        Load a CSV file from Open WebUI storage (partial filename OK), extract rows, and format for analysis.

        Args:
            file_name (str): e.g., 'dataset.csv'
            question (str): Question to answer about the data
            sample_size (int or None): None = full file; otherwise use as nrows

        Returns:
            dict or str: Prompt + data or error string
        """
        found_path = self._resolve_uploaded_file_path(file_name)

        if not found_path:
            return f"❌ File ending with '{file_name}' not found in WebUI storage."

        try:
            df = (
                pd.read_csv(found_path)
                if sample_size is None
                else pd.read_csv(found_path, nrows=sample_size)
            )
            sample = df.to_dict(orient="records")
            result = json.dumps({"question": question, "data": sample})
            print(f"✅ Successfully collected dataset '{file_name}' with {len(sample)} records.")
            return result

        except Exception as e:
            return f"❌ Could not analyze file `{file_name}`: {e}"
