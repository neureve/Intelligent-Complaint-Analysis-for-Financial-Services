import pandas as pd
import re

class ComplaintDataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load the complaint dataset from a CSV file."""
        self.df = pd.read_csv(self.filepath)
        return self.df

    def basic_info(self):
        """Print basic information about the dataset."""
        if self.df is not None:
            print("\nDataset Info:")
            print(self.df.info())
            print("\nMissing Values:")
            print(self.df.isnull().sum())
        else:
            print("Data not loaded yet.")

    def narrative_stats(self) -> pd.DataFrame:
        """Return basic stats on the narrative column (lengths, nulls, etc.)."""
        if self.df is None:
            raise ValueError("Data not loaded.")
        df = self.df.copy()
        df['narrative_word_count'] = df['Consumer complaint narrative'].dropna().apply(lambda x: len(str(x).split()))
        return df
    
    def filter_relevant_complaints(self) -> pd.DataFrame:
        """
        Filter dataset to include only relevant products and non-empty narratives.
        Returns the filtered DataFrame.
        """
        target_products = {
            "Credit card",
            "Credit card or prepaid card",
            "Payday loan, title loan, personal loan, or advance loan",
            "Payday loan, title loan, or personal loan",
            "Checking or savings account",
            "Money transfer, virtual currency, or money service",
            "Money transfers"
        }

        df = self.df.copy()

        # Apply filtering
        filtered_df = df[
            df['Product'].isin(target_products) &
            df['Consumer complaint narrative'].notna() &
            (df['Consumer complaint narrative'].str.strip() != "")
        ].copy()

        return filtered_df

    @staticmethod
    def clean_narrative(text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()

        # Remove boilerplate phrases (can add more)
        boilerplates = [
            r"i am writing to file a complaint",
            r"this is regarding",
            r"dear cfpb",
            r"to whom it may concern",
            r"thank you for your time"
        ]
        for phrase in boilerplates:
            text = re.sub(phrase, '', text)

        # Remove special characters and numbers (except spaces and letters)
        text = re.sub(r"[^a-z\s]", " ", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

