import os
import pandas as pd


class Tools:
    def analyze_wine_club_csv(self, file, question: str) -> str:
        try:
            file_path = file if isinstance(file, str) else file.file_path

            if not file_path.startswith("/tmp/"):
                return "❌ Please upload a CSV file directly when calling this tool."

            print(f"✅ Reading CSV from: {file_path}")
            df = pd.read_csv(file_path)

            sample = df.head(100).to_dict(orient="records")
            instruction = "You are a wine club business analyst. Use the sample data below to answer this question: "

            return {"instruction": instruction + question, "data": sample}

        except Exception as e:
            print(f"❌ Error: {e}")
            return f"❌ Could not load the CSV: {e}"

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

    def load_raw_csv_from_storage(self, file_name: str, max_rows: int = 100) -> str:
        found_path = self._resolve_uploaded_file_path(file_name)

        if not found_path:
            return f"❌ File ending with '{file_name}' not found in uploads directory."

        try:
            df = pd.read_csv(found_path, nrows=max_rows)
            return f"✅ Preview of `{file_name}` ({df.shape[0]} rows × {df.shape[1]} columns):\n\n{df.head(max_rows).to_markdown(index=False)}"
        except Exception as e:
            return f"❌ Failed to read `{file_name}`: {str(e)}"

    def analyze_wine_club_from_knowledge(
        self, file_name: str, question: str, sample_size: int = None
    ):
        """
        Load a CSV file from Open WebUI storage (partial filename OK), extract rows, and format for analysis.

        Args:
            file_name (str): e.g., 'winery-dataset.csv'
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

            instruction = "You are a wine club business analyst. Use the data below to answer this question: "
            return {"instruction": instruction + question, "data": sample}

        except Exception as e:
            return f"❌ Could not analyze file `{file_name}`: {e}"
