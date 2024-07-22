from google.cloud import bigquery
from google.oauth2 import service_account
from pathlib import Path
import os
from google.cloud.exceptions import GoogleCloudError, BadRequest
import streamlit as st
import json


class GBQUtils:
    def __init__(self):
        home = str(Path.home())
        credential_path = home + r'\Waternlife\05_Business Intelligence - General\06_BI Team Documents\09_Important docs\01_API KEYS - PROTECTED\giovanni_keys\danish-endurance-analytics-3cc957295117.json'
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
        credentials = service_account.Credentials.from_service_account_file(
            credential_path,
        )
        self.client = bigquery.Client(credentials=credentials)
    def get_client(self):
        return self.client
    
    def get_and_save_field_values_lookup(self, table_id, fetch_all: list, fetch_partial: list):
        ''' Save the distinct values of a table to a json file
        '''
        import pandas as pd

        table_ref = self.client.get_table(table_id)


        distinct_values = {} # Initialize an empty dictionary
        distinct_values['table_id'] = table_id
        distinct_values['table_description'] = table_ref.description
        distinct_values['fields'] = [] 

        # get descripitions
        df = pd.read_csv(r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\table_metadata\table_and_column_descripitions.txt', sep=":", header=None, names=["field", "description"])

        # remove leading and trailing whitespaces
        df["field"] = df["field"].str.strip()
        df["description"] = df["description"].str.strip()

        # convert the dataframe to a dictionary
        dict_descripitions = df.set_index('field').T.to_dict('records')[0]

        for field in table_ref.schema:
            if field.name in fetch_all:
                query = f"SELECT DISTINCT {field.name} FROM {table_id} WHERE {field.name} IS NOT NULL"

            elif field.name in fetch_partial:
                query = f"SELECT DISTINCT {field.name} FROM {table_id} WHERE {field.name} IS NOT NULL LIMIT 20"
            else:
                continue

            query_job_result = self.client.query(query).result() # perform the query

            distinct_values['fields'].append({
                        "field_name": field.name,
                        "field_description": dict_descripitions.get(field.name, "No description available"),
                        "field_dataType": field.field_type,
                        "distinct_values": query_job_result.to_dataframe().iloc[:,0].values.tolist()})
            
            # save dict to json file
            with open(r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\table_metadata\fields_look_up.json', 'w') as f:
                json.dump(distinct_values, f, indent=4)

        return distinct_values

    def get_openAI_table_schema(self, table_id):
        fields = []
        if "`" in table_id:
            table_id = table_id.replace("`", "")
        table_ref = self.client.get_table(table_id)

        for field in table_ref.schema:
            fields.append(field.name)

        # Create the SQL statement
        table_schema_ddl = f'danish-endurance-analytics.nl2sql.amazon_orders{tuple(fields)}'
     
        return table_schema_ddl
    
    def get_simple_ddl_table_schema(self, table_id):
        fields = []
        if "`" in table_id:
            table_id = table_id.replace("`", "")
        table_ref = self.client.get_table(table_id)

        for field in table_ref.schema:
            schema_str = f"{field.name} {field.field_type} {field.description}"
            fields.append(schema_str)

        # Create the SQL statement
        table_schema_ddl = f"CREATE TABLE {table_id}(\n  " + ",\n  ".join(fields) + "\n);"
     
        return table_schema_ddl
    
    def format_distinct_values_as_list (self, query_job_result) -> list:
        
        pass
    
    def format_distinct_values(self, query_job_result):

        ''' Format the results of a query job to a string accepts only one column in the result set'''

        column_name = query_job_result.schema[0].name

        # Initialize an empty string to store the results
        result_string = ""

        # Iterate over each row in the results
        for row in query_job_result:
            # Dynamically access the column value using the column_name
            column_value = getattr(row, column_name)
            # Append the column value to the result string with a newline character
            result_string += f"{column_value}, "

        return result_string

    
    def get_long_context_window_fields(self, table_id, fetch_all: list, fetch_partial: list):
        ''' Get the distinct values for the columns in a table'''
        import pandas as pd

        table_ref = self.client.get_table(table_id)


        distinct_values = {} # Initialize an empty dictionary
        distinct_values['table_id'] = table_id
        distinct_values['table_description'] = table_ref.description
        distinct_values['fields'] = [] 

        # get descripitions
        df = pd.read_csv(r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\table_metadata\table_and_column_descripitions.txt', sep=":", header=None, names=["field", "description"])

        # remove leading and trailing whitespaces
        df["field"] = df["field"].str.strip()
        df["description"] = df["description"].str.strip()

        # convert the dataframe to a dictionary
        dict_descripitions = df.set_index('field').T.to_dict('records')[0]

        for field in table_ref.schema:
            if field.name in fetch_all:
                query = f"SELECT DISTINCT {field.name} FROM {table_id} WHERE {field.name} IS NOT NULL"

            elif field.name in fetch_partial:
                query = f"SELECT DISTINCT {field.name} FROM {table_id} WHERE {field.name} IS NOT NULL LIMIT 20"
            else:
                continue

            query_job_result = self.client.query(query).result() # perform the query

            distinct_values['fields'].append({
                        "field_name": field.name,
                        "field_description": dict_descripitions.get(field.name, "No description available"),
                        "field_dataType": field.field_type,
                        "distinct_values": self.format_distinct_values(query_job_result)})

        return distinct_values
    
    def run_query(self, query):
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            print(f"Query executed successfully. {results.total_rows} rows returned.")
            return results
        except BadRequest as error:
            # BadRequest errors provide more detailed information about query issues
            # Attempting to parse error details for more specific feedback
            err_msg = "A query error occurred: "
            if hasattr(error, 'errors') and error.errors:
                for e in error.errors:
                    err_msg += f"{e['message']} "
                    if 'location' in e:
                        err_msg += f"at location {e['location']}. "
            return err_msg
        except GoogleCloudError as error:
            return f"A Google Cloud error occurred: {error}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"
