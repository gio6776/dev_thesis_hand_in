from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import ChatPromptTemplate
import json
from packages.dev_thesis.gbq_utils import GBQUtils
from openai import OpenAI
from langchain import PromptTemplate
from datetime import datetime
from google.cloud.bigquery.table import RowIterator
import os
import csv
# from packages.dev_thesis.llm_utils import Format

with open(r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\table_metadata\fields_look_up.json', 'r') as f:
    data = json.load(f)

# Initialize the output dictionary
SCHEMA_LINKING_LOOKUP = {}

# Loop through each field in the 'fields' list
for field in data['fields']:
    field_name = field['field_name']
    # Map each distinct value to the field_name
    for value in field['distinct_values']:
        SCHEMA_LINKING_LOOKUP[value] = field_name

with open(r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\prompt_template\5_query_classifier\metrics_layer\growth_metrics.json', 'r') as f:
    GROWTH_METRICS_LOOKUP = json.load(f)

with open(r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\prompt_template\5_query_classifier\metrics_layer\financial_metrics.json', 'r') as f:
    FINANCIAL_METRICS_LOOKUP = json.load(f)

with open(r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\prompt_template\5_query_classifier\metrics_layer\date_range.json', 'r') as f:
    DATERANGE_METRICS_LOOKUP = json.load(f)

# print("Financial Metrics Lookup: ", FINANCIAL_METRICS_LOOKUP)
# print("Growth Metrics Lookup: ", GROWTH_METRICS_LOOKUP)
# print("Date Range Metrics Lookup: ", DATERANGE_METRICS_LOOKUP)
# print("Schema Linking Lookup: ", SCHEMA_LINKING_LOOKUP)


current_dir = r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis'
class LLMUtils:
    def __init__(self):
        try:
            self.client = OpenAI()
            print("Client initialized successfully.")
        except Exception as e:
            print(f"Error initializing client: {str(e)}")
        pass

    def get_lookups(self):
        return SCHEMA_LINKING_LOOKUP, GROWTH_METRICS_LOOKUP, FINANCIAL_METRICS_LOOKUP, DATERANGE_METRICS_LOOKUP

    def invoke_openAI_vanilla(self, model:str, question: str):
        prompt_template = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/1_OpenAIDemo/prompt.txt'))

        table_schema = MetadataLoader.get_all_metadata(os.path.join(current_dir, 'data/de_data/prompt_template/1_OpenAIDemo/table_schema.txt'))

        user_question = question['question']

        # Messages
        messages=[
            {"role": "user", "content": Format.get_prompt_simple(user_question, prompt_template, table_schema)},
        ]

        print(messages)
        
        # API call
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages
        )

        # Response Handling -> Dictionary with SQL query and other metadata
        formatted_response = Format.format_llm_response(response, user_question)

        formatted_response['user_question'] = user_question
        formatted_response['complexity'] = question['complexity']
        formatted_response['model'] = model
        formatted_response['chain'] = 'openAI_chain'


        # Run SQL Query on the database
        if formatted_response.get('sql_query') != None:
            print(formatted_response.get('sql_query'))
            gbq = GBQUtils()
            query_result = gbq.run_query(formatted_response.get('sql_query'))
            if not isinstance(query_result, RowIterator): # procuded an error
                print('## Query Produced an Error ##')
                print("\t {}".format(query_result))
                query_debug = LLMQueryDebug()
                ai_faulty_message = {
                    "role": "assistant",
                    "content": '''{}'''.format(response.choices[0].message.content)
                }
                messages.append(ai_faulty_message)
                query_result = query_debug.invoke_query_debug_chain(messages=messages, error_message=query_result, model = model)      
        else:
            query_result = "No SQL query generated."

        # Log the response
        FrameworkEval.log_response(formatted_response, iterator = query_result)

        return formatted_response, response

    def ivoke_smtNL2BI(self, model:str, question: str):
        '''
        Invoke openai vanilla chain with classfiier
        - Classify the question
        - Format the Prompt and pass it into the normal flow of the openAI chain
        '''
        ## Classify the question ##
        llm_classifier = LLMQueryClassification()
        question_classfied = llm_classifier.invoke_query_classification_chain(user_question=question, model=model)

        # process the classification
        classification_results = llm_classifier.process_classification(question_classfied)

        answerable, reasons_found, reasons_not_found = llm_classifier.decide_if_question_is_answerable(classification_results)
        
        user_question = question['question']
        messages = []
        query_result = 'Not Answerable'
        response = 'Not Answerable'

        if answerable:
            classifier_guidelines = Format.get_classification_guidelines(classification_results)

            ## Run the openAI chain
            sys_message = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/6_TF-SQL/tf-sql_sysmessage.txt'))

            prompt_template = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/6_TF-SQL/tf-sql_prompt.txt'))

            final_prompt = Format.get_prompt_simple_with_classifier(user_question = user_question, prompt_template = prompt_template, classifier_guidelines=classifier_guidelines)

            # Messages
            messages=[
                {"role": "system", "content": sys_message},
                {"role": "user", "content": final_prompt},
            ]
            
            # API call
            response = self.client.chat.completions.create(
                model=model,
                temperature=0,
                messages=messages
            )
            # Response Handling -> Dictionary with SQL query and other metadata
            formatted_response = Format.format_llm_response(response, user_question)

            # Run SQL Query on the database
            if formatted_response.get('sql_query') != None:
                # print(formatted_response.get('sql_query'))
                gbq = GBQUtils()
                query_result = gbq.run_query(formatted_response.get('sql_query'))
                # if not isinstance(query_result, RowIterator): # procuded an error
                #     print('## Query Produced an Error ##')
                #     print("\t {}".format(query_result))
                #     query_debug = LLMQueryDebug()
                #     ai_faulty_message = {
                #         "role": "assistant",
                #         "content": '''{}'''.format(response.choices[0].message.content)
                #     }
                #     messages.append(ai_faulty_message)
                #     query_result = query_debug.invoke_query_debug_chain(messages=messages, error_message=query_result, model = model)      
            else:
                query_result = "No SQL query generated."


        else:
            # if the question is not answerable, return the classification results
            formatted_response = {}
            reseponse = 'Not Answerable'
            formatted_response['id'] = None
            formatted_response['created_at'] = None
            formatted_response['sql_query'] = None
            formatted_response['total_tokens'] = None

        formatted_response['user_question'] = user_question
        formatted_response['classifier_results'] = classification_results
        formatted_response['final_prompt'] = messages[1].get('content') if messages else None
        formatted_response['complexity'] = question['complexity']
        formatted_response['model'] = model
        formatted_response['chain'] = 'SMT-NL2BI'

        # Log the response
        FrameworkEval.log_response(formatted_response, iterator = query_result)

        return formatted_response, response

    def invoke_col_meta_guided_dv(self, model:str, question: str):

        prompt_template = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/4_colMetaGuidedDV/prompt.txt'))

        sys_message = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/4_colMetaGuidedDV/sysmessage.txt'))

        user_question = question['question']

        # Messages
        messages=[
            {"role": "system", "content": sys_message},
            {"role": "user", "content": Format.get_prompt_w_user_question(user_question, prompt_template)},
        ]

        print(messages)
        
        # API call
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages
        )

        # Response Handling -> Dictionary with SQL query and other metadata
        formatted_response = Format.format_llm_response(response, user_question)

        formatted_response['user_question'] = user_question
        formatted_response['complexity'] = question['complexity']
        formatted_response['model'] = model
        formatted_response['chain'] = 'colMetaGuidedDV'


        # Run SQL Query on the database
        if formatted_response.get('sql_query') != None:
            print(formatted_response.get('sql_query'))
            gbq = GBQUtils()
            query_result = gbq.run_query(formatted_response.get('sql_query'))   
        else:
            query_result = "No SQL query generated."

        # Log the response
        FrameworkEval.log_response(formatted_response, iterator = query_result)

        return formatted_response, response

    def invoke_col_meta(self, model:str, question: str):

        prompt_template = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/2_colMeta/prompt.txt'))

        sys_message = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/2_colMeta/sysmessage.txt'))

        user_question = question['question']

        # Messages
        messages=[
            {"role": "system", "content": sys_message},
            {"role": "user", "content": Format.get_prompt_w_user_question(user_question, prompt_template)},
        ]

        print(messages)
        
        # API call
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages
        )

        # Response Handling -> Dictionary with SQL query and other metadata
        formatted_response = Format.format_llm_response(response, user_question)

        formatted_response['user_question'] = user_question
        formatted_response['complexity'] = question['complexity']
        formatted_response['model'] = model
        formatted_response['chain'] = 'explained_chain'


        # Run SQL Query on the database
        if formatted_response.get('sql_query') != None:
            print(formatted_response.get('sql_query'))
            gbq = GBQUtils()
            query_result = gbq.run_query(formatted_response.get('sql_query'))   
        else:
            query_result = "No SQL query generated."

        # Log the response
        FrameworkEval.log_response(formatted_response, iterator = query_result)

        return formatted_response, response

    def invoke_col_meta_guided(self, model:str, question: str):

        prompt_template = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/3_colMetaGuided/prompt.txt'))

        sys_message = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/3_colMetaGuided/sysmessage.txt'))

        user_question = question['question']

        # Messages
        messages=[
            {"role": "system", "content": sys_message},
            {"role": "user", "content": Format.get_prompt_w_user_question(user_question, prompt_template)},
        ]

        print(messages)
        
        # API call
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages
        )

        # Response Handling -> Dictionary with SQL query and other metadata
        formatted_response = Format.format_llm_response(response, user_question)

        formatted_response['user_question'] = user_question
        formatted_response['complexity'] = question['complexity']
        formatted_response['model'] = model
        formatted_response['chain'] = 'explained_chain'


        # Run SQL Query on the database
        if formatted_response.get('sql_query') != None:
            print(formatted_response.get('sql_query'))
            gbq = GBQUtils()
            query_result = gbq.run_query(formatted_response.get('sql_query'))   
        else:
            query_result = "No SQL query generated."

        # Log the response
        FrameworkEval.log_response(formatted_response, iterator = query_result)

        return formatted_response, response



    def load_template_from_file(self, file_path):
        with open(file_path, 'r') as file:
            template_str = file.read()
        return template_str

class LLMQueryClassification(LLMUtils):
    def __init__(self):
        super().__init__()
        pass

    def invoke_query_classification_chain(self, user_question, model = 'gpt-4-turbo'):
        print("##Invoking Query Classification Chain##")

        sys_message_template = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/5_query_classifier/qc_dynamic_system_message.txt'))

        prompt_template = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/5_query_classifier/qc_prompt_template.txt'))

        user_question = user_question['question']

        sys_message = Format.get_sysmessage_classification(prompt_template=sys_message_template)

        prompt =  Format.get_prompt_classification(user_question=user_question, prompt_template=prompt_template)

            # Messages
        messages=[
            {"role": "system", "content": sys_message},
            {"role": "user", "content": prompt},
        ]

        print(messages)

        # API call
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
            response_format={ "type": "json_object" }
        )

        # formatted response to be logged
        formatted_response = {
            "user_question": user_question,
            "prompt": prompt,
            "sys_message": sys_message,
            "id": response.id,
            "created_at": response.created,
            "classification": json.loads(response.choices[0].message.content),
            "model": model
        }

        FrameworkEval.log_response_classifier(formatted_response)


        return json.loads(response.choices[0].message.content)

    def process_classification(self, classification):
        '''
        Process classification for allowed and not_allowed entries and returns their status and details from the lookup.

        Parameters:
        - classification: Dict with 'allowed' and 'not_allowed' keys, each containing categories like financial_metrics, etc.
        
        Returns:
        - Dictionary of classification with status and additional details for metrics.
        '''

        def process_category(items, lookup):
            results = []
            for item in items:
                # print(item)
                if item in lookup:
                    item_details = lookup[item]
                    if lookup == SCHEMA_LINKING_LOOKUP:
                        results.append({item: {'found': 1, 'field_name': item_details}})
                        continue
                    else:
                        results.append({
                            item: {
                                'found': 1,
                                'description': item_details.get('description', 'No description available'),
                                'calculation_guidelines': item_details.get('calculation_guidelines', 'No guidelines available'),
                                'calculation_example': item_details.get('calculation_example', 'No example available')
                            }
                        })
                else:
                    results.append({item: {'found': 0}})
            return results

        results = {}

        # Loop over both 'allowed' and 'not_allowed'
        for status in ['allowed', 'not_allowed']:
            results[status] = {}

            for category, lookup in [
                ('products', SCHEMA_LINKING_LOOKUP),
                ('countries_alpha_2_code', SCHEMA_LINKING_LOOKUP),
                ('financial_metrics', FINANCIAL_METRICS_LOOKUP),
                ('growth_metrics', GROWTH_METRICS_LOOKUP),
                ('date_range', DATERANGE_METRICS_LOOKUP)
            ]:
                items = classification.get(status, {}).get(category, [])
                # print(items)  
                results[status][category] = process_category(items, lookup)

        return results

    def decide_if_question_is_answerable(self, classification: dict):
        '''
        Decide if the question is answerable based on the classification.

        - Returns false if the question contains the following:
            - Not Found Products;
            - Not Found Financial Metrics;
            - Not Found Growth Metrics;
            - Not Found Date Range Metrics;
            - Any Metric on the not allowed key.
        - Returns both FOUND and NOT FOUND metrics in a structured format.
        '''

        def collect_elements(category):
            found_elements = []
            not_found_elements = []
            for items in category:
                for metric, details in items.items():
                    if details['found'] == 0:
                        not_found_elements.append(metric)
                    else:
                        found_elements.append(metric)
            return found_elements, not_found_elements

        reasons_not_found = {}
        reasons_found = {}

        # Check allowed metrics
        for key in ['products', 'financial_metrics', 'growth_metrics', 'date_range', 'countries_alpha_2_code']:
            if key in classification.get('allowed', {}):
                found, not_found = collect_elements(classification['allowed'][key])
                if found:
                    reasons_found[key] = found
                if not_found:
                    reasons_not_found[key] = not_found

        # Check not allowed metrics
        for key in ['products', 'financial_metrics', 'growth_metrics', 'date_range']:
            if key in classification.get('not_allowed', {}):
                _, not_allowed_metrics = collect_elements(classification['not_allowed'][key])
                if not_allowed_metrics or classification['not_allowed'][key]:
                    if key not in reasons_not_found:
                        reasons_not_found[key] = not_allowed_metrics
                    else:
                        reasons_not_found[key].extend(not_allowed_metrics)

        if reasons_not_found:
            return False, reasons_found, reasons_not_found

        return True, reasons_found, None

    
class LLMQueryDebug(LLMUtils):
    def __init__(self):
        super().__init__()
        pass

    def invoke_query_debug_chain(self, messages, error_message, model, user_question: str = None):
        print("##Invoking Query Debug Chain##")

        prompt_template = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/4_query_debug/query_debug_prompt.txt'))
        
        # Messages
        debug_message= {"role": "user", 
                        "content": Format.get_prompt_debug(prompt_template=prompt_template, error_message=error_message)}

        messages.append(debug_message)

        # API call
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages
        )

        # Response Handling -> Dictionary with SQL query and other metadata
        formatted_response = Format.format_llm_response(response, user_question)

        # Run SQL Query on the database
        if formatted_response.get('sql_query') != None:
            print(formatted_response.get('sql_query'))
            gbq = GBQUtils()
            query_result = gbq.run_query(formatted_response.get('sql_query'))    
        else:
            query_result = "No SQL query generated."
        
        return query_result
    
class LLMTabular2NL(LLMUtils):

    def __init__(self):
        super().__init__()
        print(current_dir)
        pass

    def invoke_tabular2sql_chain(self, user_question: str, tabular_response: str):
        print("Invoking Tabular 2 TEXT Chain")
        
        sys_message = self.load_template_from_file(current_dir + os.sep + 'data' + os.sep + 'de_data' + os.sep + 'prompt_template' + os.sep + 'tabular2sql_sysMessage.txt')

        prompt_template = self.load_template_from_file(os.path.join(current_dir, 'data/de_data/prompt_template/tabular2SQL_promptTemplate.txt'))

        # Messages
        messages=[
            {"role": "system", "content": sys_message},
            {"role": "user", "content": Format.get_prompt_tabular2sql(user_question=user_question, prompt_template = prompt_template, tabular_response=tabular_response)},
        ]

        print(messages)

        # API call
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=messages
        )

        # Response Handling -> Dictionary with SQL query and other metadata
        # formatted_response = Format.format_llm_response(response, user_question)

        response = response.choices[0].message.content

        # formatted_response['user_question'] = user_question

        # # Log the response
        # FrameworkEval.log_response(formatted_response, iterator = query_result)

        return response

class Format:
    def __init__(self):
        pass

    @staticmethod
    def get_classification_guidelines(classification:dict):
        '''
            Get the classification guidelines for the classification
        '''
        # Parse the input JSON string to a dictionary
        metrics_info = classification
        
        # Start forming the response string
        response = ""
        
        # Process products under the "allowed" category
        if len(metrics_info['allowed']['products']) > 0:
            response += "###Schema Linking Instructions:###\n"
            for product in metrics_info['allowed']['products']:
                for product_name, details in product.items():
                    response += f"{product_name} maps to the column {details['field_name']}\n"
        
        # Add growth metrics section
        if len(metrics_info['allowed']['growth_metrics']) > 0:
            response += "###Growth Metrics Instructions###\n"
            for metric in metrics_info['allowed']['growth_metrics']:
                for metric_name, details in metric.items():
                    if details['found'] == 1:
                        response += f"{metric_name}:\n{details['calculation_guidelines']}\n {details['calculation_example']}\n"
        
        # Add date_range section
        if len(metrics_info['allowed']['date_range']) > 0:
            response += "###Date Interpretation Instructions:###\n"
            for metric in metrics_info['allowed']['date_range']:
                for metric_name, details in metric.items():
                    if details['found'] == 1:
                        response += f"{metric_name}:\n{details['calculation_guidelines']}\n{details['calculation_example']}\n"
        
                # Add date_range section
        if len(metrics_info['allowed']['financial_metrics']) > 0:
            response += "\n### Financial Metrics instructions###:\n"
            for metric in metrics_info['allowed']['financial_metrics']:
                for metric_name, details in metric.items():
                    if details['found'] == 1:
                        response += f"{metric_name}: \n {details['calculation_guidelines']}\n"
        
        return response
    
    @staticmethod
    def get_sysmessage_classification(prompt_template):
        '''
            Get the formatted prompt for the classification chain
        '''
        from datetime import datetime
        today_date = datetime.today().strftime('%A %d %B %Y')
        ## FORMAT THE PROMPT
        prompt = PromptTemplate(
            input_variables=['dynamic_system_message'],
            template=prompt_template
        )

        dynamic_system_message = ""

        dynamic_system_message += "\n## date_range:\n"
        for key, value in DATERANGE_METRICS_LOOKUP.items():
            dynamic_system_message += f"- {key}: {value['description']}\n"
        
        dynamic_system_message += "\n## financial_metrics:\n"
        for key, value in FINANCIAL_METRICS_LOOKUP.items():
            dynamic_system_message += f"- {key}: {value['description']}\n"

        dynamic_system_message += "\n## growth_metrics:\n"
        for key, value in GROWTH_METRICS_LOOKUP.items():
            dynamic_system_message += f"- {key}: {value['description']}\n"

        prompt_formatted = prompt.format(dynamic_system_message=dynamic_system_message)
        return prompt_formatted

    @staticmethod
    def get_prompt_classification(prompt_template, user_question):
        '''
            Get the formatted prompt for the classification chain
        '''
        ## FORMAT THE PROMPT
        prompt = PromptTemplate(
            input_variables=['user_question'],
            template=prompt_template
        )

        prompt_formatted = prompt.format(
            user_question=user_question,
        )
        return prompt_formatted

    @staticmethod
    def get_prompt_debug(prompt_template, error_message):
        '''
            Get the formatted prompt for the debug chain
        '''
        ## FORMAT THE PROMPT
        prompt = PromptTemplate(
            input_variables=['error_message'],
            template=prompt_template
        )

        prompt_formatted = prompt.format(
            error_message=error_message,
        )
        return prompt_formatted

    @staticmethod
    def get_prompt_simple(user_question, prompt_template, table_schema):
        '''
            Get the formatted prompt for the simple chain
        '''
        ## FORMAT THE PROMPT
        prompt = PromptTemplate(
            input_variables=['user_question', 'table_schema'],
            template=prompt_template
        )

        prompt_formatted = prompt.format(
            user_question=user_question,
            table_schema= table_schema
        )
        return prompt_formatted
    
    @staticmethod
    def get_prompt_w_user_question(user_question, prompt_template):
        '''
            Get the formatted prompt for the simple chain
        '''
        ## FORMAT THE PROMPT
        prompt = PromptTemplate(
            input_variables=['user_question'],
            template=prompt_template
        )

        prompt_formatted = prompt.format(
            user_question=user_question,
        )
        return prompt_formatted

    @staticmethod
    def get_prompt_simple_with_classifier(user_question, classifier_guidelines, prompt_template):
        '''
            Get the formatted prompt for the simple chain
        '''
        ## FORMAT THE PROMPT
        prompt = PromptTemplate(
            input_variables=['user_question', 'classifier_guidelines'],
            template=prompt_template
        )

        prompt_formatted = prompt.format(
            user_question=user_question,
            classifier_guidelines = classifier_guidelines
        )
        return prompt_formatted

    
    @staticmethod
    def get_prompt_tabular2sql(user_question, prompt_template, tabular_response):
        '''
            Get the formatted prompt for the simple chain
        '''
        ## FORMAT THE PROMPT
        prompt = PromptTemplate(
            input_variables=['user_question', 'tabular_response'],
            template=prompt_template
        )

        prompt_formatted = prompt.format(
            user_question=user_question,
            tabular_response= tabular_response
        )
        return prompt_formatted
    
    @staticmethod
    def format_llm_response(response, user_question) -> dict:
        '''
            Format the response from the LLM
        '''
        import re
        pattern = r"```sql\n(.*?)\n```"
        match = re.search(pattern, response.choices[0].message.content, re.DOTALL)
        if match:
            sql_query = match.group(1).strip()
        elif "SELECT" in response.choices[0].message.content:
            sql_query = response.choices[0].message.content
        else:
            sql_query = None
        
        # Create a dictionary to hold the response and the SQL query
        formatted_response = {
            "id": response.id,
            "created_at": response.created,
            "sql_query": sql_query,
            "user_question": user_question,
            "total_tokens": response.usage.total_tokens
        }
        
        return formatted_response
   
    @staticmethod
    def get_fields_lookup(json_data):
        """
        Transforms JSON data into a dictionary where each distinct value is mapped to its field name.

        Args:
        json_data (str): JSON formatted string containing the data structure.

        Returns:
        dict: Dictionary where keys are distinct values and values are corresponding field names.
        """
        # Parse the JSON string into a dictionary
        try:
            with open(json_data, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            print("The file was not found.")
        except json.JSONDecodeError as e:
            print("An error occurred while decoding JSON:", str(e))
        
        # Initialize the output dictionary
        output_dict = {}
        
        # Loop through each field in the 'fields' list
        for field in data['fields']:
            field_name = field['field_name']
            # Map each distinct value to the field_name
            for value in field['distinct_values']:
                output_dict[value] = field_name
                
        return output_dict
        
    
class MetadataLoader:
    def __init__(self):
        pass

    @staticmethod
    def get_all_metadata(path_to_file):
        '''
            Get the metadta for a table from a json file
        '''
        # if it's JSON file load it
        if path_to_file.endswith('.json'):
            with open(path_to_file, 'r') as file:
                file_content = json.load(file)
            return file_content
        if path_to_file.endswith('.txt'):
            with open(path_to_file, 'r') as file:
                file_content = file.read()
            return file_content

    
class FrameworkEval:
    def __init__(self):
        pass

    @staticmethod
    def run_eval(models:list, chains:list, questions:list, hyper_parameters:dict = None):
        '''
            Run the evaluation of the framework
        '''
        for model in models:
            for chain in chains:
                print(f"RUNNIG EVALUATION FOR MODEL: {model} AND CHAIN: {chain}")
                print("=====================================================")
                for question in questions:
                    print(f"\t Question: {question['question']}")
                    llm = LLMUtils()
                    if chain == 'OpenAIDemo':
                        llm.invoke_openAI_vanilla(model = model, question=question)

                    elif chain == 'colMeta':
                        llm.invoke_col_meta(model = model, question=question)

                    elif chain == 'colMetaGuided':
                        llm.invoke_col_meta_guided(model = model, question=question)

                    elif chain == 'colMetaGuidedDV':
                        llm.invoke_col_meta_guided_dv(model = model, question=question)

                    elif chain == 'SMT-NL2BI':
                        llm.ivoke_smtNL2BI(model = model, question=question)

                    elif chain == 'query_classifier_chain':
                        llm = LLMQueryClassification()
                        llm.invoke_query_classification_chain(user_question=question, model=model)



                    else:
                        continue

    @staticmethod
    def log_response(formatted_response: dict, iterator):
        '''
            Log the response from the LLM
        '''
        # Define your file path
        file_path = r"G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\logging\login_response.json"

        # Check if the file exists and read its content if it does
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        query_results_file_path = f"G:/My Drive/Profissional & Acadêmico/Mestrados/DTU/5_thesis/dev_thesis/data/de_data/logging/logging_results/{formatted_response.get('id')}.csv"
        # Save Result to CSV
        FrameworkEval().save_row_iterator_to_csv(iterator, query_results_file_path)

        # open the in read mode
        with open(query_results_file_path, 'r') as file:
            query_results_csv = file.read()

        # check the file
        if "Not Answerable" in query_results_csv:
            formatted_response['automated_result_analysis'] = "not_answerable"
        elif "error" in query_results_csv.lower():
            formatted_response['automated_result_analysis'] = "generated_error"
            formatted_response['error_message'] = query_results_csv.split("\n")[0].replace(". ", ".")
        elif iterator.total_rows == 0:
            formatted_response['automated_result_analysis'] = "empty_result"
        else:
            formatted_response['automated_result_analysis'] = "no_error_found"	

        # add file path to the formatted response
        formatted_response['query_results_file_path'] = query_results_file_path

        # Append the new response to the data array
        data.append(formatted_response)

        # Write the updated data back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

        print("Response logged successfully.")
    
    def log_response_classifier(formatted_response: dict):
        '''
            Log the response from the LLM
        '''
        # Define your file path
        file_path = r"G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\logging\login_response_classifier.json"

        # Check if the file exists and read its content if it does
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # Append the new response to the data array
        data.append(formatted_response)

        # Write the updated data back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

        print("Response logged successfully.")
    
    @staticmethod
    def save_row_iterator_to_csv(iterator, filename):
        """
        Saves the rows returned by a BigQuery query job to a CSV file.
        
        Parameters:
        - query_job: The query job object from BigQuery.
        - filename: The name of the file to save the CSV data.
        """

        with open(filename, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write headers based on the schema of the query result
            if isinstance(iterator, RowIterator):
                headers = [field.name for field in iterator.schema]
                csv_writer.writerow(headers)
                # Write each row from the iterator
                for row in iterator:
                    csv_writer.writerow(row.values())
            else:
                csv_writer.writerow([iterator])

class HelperFunctions:
    def __init__(self):
        pass

    @staticmethod
    def get_classifier_sysmessage_by_id(query_id, json_file_path = r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\logging\login_response_classifier.json'):
        import json
        """
        Reads a JSON file containing queries and returns the SQL query for the specified ID.

        Parameters:
        - json_file_path (str): The path to the JSON file containing the query data.
        - query_id (str): The ID of the desired query.

        Returns:
        - str or None: The SQL query corresponding to the given ID, or None if the ID is not found.
        """
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Search for the query with the specified ID
        for entry in data:
            if entry.get("id") == query_id:
                return entry.get("sys_message")

        # Return None if the ID was not found
        return None
    
    @staticmethod
    def get_classifier_prompt_by_id(query_id, json_file_path = r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\logging\login_response_classifier.json'):
        import json
        """
        Reads a JSON file containing queries and returns the SQL query for the specified ID.

        Parameters:
        - json_file_path (str): The path to the JSON file containing the query data.
        - query_id (str): The ID of the desired query.

        Returns:
        - str or None: The SQL query corresponding to the given ID, or None if the ID is not found.
        """
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Search for the query with the specified ID
        for entry in data:
            if entry.get("id") == query_id:
                return entry.get("prompt")

        # Return None if the ID was not found
        return None

    @staticmethod
    def get_sql_query_by_id(query_id, json_file_path = r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\logging\login_response.json'):
        import json
        """
        Reads a JSON file containing queries and returns the SQL query for the specified ID.

        Parameters:
        - json_file_path (str): The path to the JSON file containing the query data.
        - query_id (str): The ID of the desired query.

        Returns:
        - str or None: The SQL query corresponding to the given ID, or None if the ID is not found.
        """
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Search for the query with the specified ID
        for entry in data:
            if entry.get("id") == query_id:
                return entry.get("sql_query")

        # Return None if the ID was not found
        return None


    @staticmethod
    def get_prompt_by_id(query_id, json_file_path = r'G:\My Drive\Profissional & Acadêmico\Mestrados\DTU\5_thesis\dev_thesis\data\de_data\logging\login_response.json'):
        import json
        """
        Reads a JSON file containing queries and returns the SQL query for the specified ID.

        Parameters:
        - json_file_path (str): The path to the JSON file containing the query data.
        - query_id (str): The ID of the desired query.

        Returns:
        - str or None: The SQL query corresponding to the given ID, or None if the ID is not found.
        """
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Search for the query with the specified ID
        for entry in data:
            if entry.get("id") == query_id:
                return entry.get("final_prompt")

        # Return None if the ID was not found
        return None



                    

