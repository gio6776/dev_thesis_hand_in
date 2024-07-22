from langsmith import Client
from datetime import datetime, timedelta
import pandas as pd
from langsmith.schemas import Run


class LangsmithUtils:

    def __init__(self):
        try:
            self.client = Client()
            print("Langsmith client initialized successfully")
            print(self.client)
        except Exception as e:
            # Handle the exception here
            print(f"Error initializing Langsmith client: {e}")
        pass




    def get_lansmith_traces(self, project_name, save_to: str):
        runs = list(
        self.client.list_runs(
            project_name=project_name,
            is_root=True,
            )
        )
        def safe_get(obj, *keys, default=None):
            for key in keys:
                if isinstance(obj, dict):
                    obj = obj.get(key, default)
                elif obj is None:
                    return default
                else:
                    return default
            return obj
        
        df = pd.DataFrame(
            [
                {
                    "name": run.name,
                    **run.inputs,
                    "output": safe_get(run.outputs, 'output', 'content'),
                    "feedback_score": safe_get(self.get_feedback_comment(run), 'quantitative_score'),
                    "feedback_comment": safe_get(self.get_feedback_comment(run), 'qualitative_score'),
                    "error": run.error,
                    "latency": (run.end_time - run.start_time).total_seconds()
                    if run.end_time
                    else None,
                    "prompt_tokens": run.prompt_tokens,
                    "completion_tokens": run.completion_tokens,
                    "total_tokens": run.total_tokens,
                }
                for run in runs
            ],
            index=[run.id for run in runs],
        )
        

        # save the dataframe to a excel file using a save_to variable
        df.to_excel(r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\processed_results\langsmith\{}.xlsx'.format(save_to))

        return df
    
    def get_lansmith_clean_traces(self, from_path, save_to):
        # read excel into a pandas dataframe
        df = pd.read_excel(r'{}'.format(from_path))

        # rename columns according to a map
        df.rename(columns={
            'Unnamed: 0': 'id',
            'input': 'user_question',
            'output': 'sql_query',
            'feedback_score': 'correctness',
            'feedback_comment': 'error_classification',
            }, inplace=True)

        # select only the columns needed for the analysis
        df = df[['id','sql_query', 'user_question', 'correctness', 'error_classification']]

        # if the column 'correctness' is equal to 1.0, then "y" if 0.0 then "n", if null then null
        df['correctness'] = df['correctness'].apply(lambda x: 'y' if x == 1.0 else ('n' if x == 0.0 else None))

        # create mappping for the error classification, if erro_classification is column_mapping change to schema_linking, and if text contain

        # select only rows where correctness and error_classification are not null
        # df = df[df['correctness'].notnull() & df['error_classification'].notnull()].reset_index(drop=True)

        
        # reorder the columns
        df = df[['id','sql_query', 'user_question', 'correctness', 'error_classification']]

        # save the dataframe to a excel file using a save_to variable
        df.to_excel(r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\processed_results\langsmith\{}.xlsx'.format(save_to))


        return df
    
    def get_feedback_comment (self, r: Run) -> dict:
        # Extract run IDs  

        # Fetch correctness feedback for the runs
        correctness_feedback = self.client.list_feedback(run_ids=[r.id])

        scores = {}
        
        feedback = [(feedback.comment, feedback.score) for feedback in correctness_feedback]

        if len(feedback) != 0:
            scores['quantitative_score'] = feedback[0][1]
            scores['qualitative_score'] = feedback[0][0]
        else:
            scores['quantitative_score'] = None
            scores['qualitative_score'] = None
        print(scores)
        return scores


    def get_langsmith_client(self):
        return self.client