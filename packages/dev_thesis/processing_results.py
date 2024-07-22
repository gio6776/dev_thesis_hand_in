import pandas as pd

class ProcessResults():
    def __init__(self, excel_file_path, sheet_name, analysis_type:str):
        self.excel_file_path = excel_file_path
        self.sheet_name = sheet_name
        self.df = pd.read_excel(self.excel_file_path, sheet_name=self.sheet_name)
        # select only the columns that are needed
        if analysis_type == 'baseline':
            self.df = self.df[['chain', 'complexity','correctness', 'error_classification']]
        elif analysis_type == 'langsmith':
            self.df = self.df[['chain', 'complexity','correctness']]

    def get_results(self, baseline):
        if baseline:
            self.df = self.df[self.df['chain'] != 'SMT-NL2BI']
        else:
            self.df = self.df
        return self.df