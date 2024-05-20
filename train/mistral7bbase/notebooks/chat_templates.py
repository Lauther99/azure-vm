mistral7binstruct_train_template = """<s>Your task is to Generate a SQL code to the question.
[INST]The query will run on a database with the following schema:
{context}

{relations}

### Answer
Given the database schema, here is the question: '''{question}'''[/INST]
[SQL]{answer}[/SQL]</s>"""

mistral7binstruct_test_template = """<s>Your task is to Generate a SQL code to the question.
[INST]The query will run on a database with the following schema:
{context}

{relations}

### Answer
Given the database schema, here is the question: '''{question}'''[/INST]
"""

# llama38binstruct_train_template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# Your are a very helpfull assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
# Your task is to Generate SQL SERVER 2014 QUERY to the user question. 
# Do not add any other explanation, only SQL code.
# Do not hallucinate or infer new information in the SQL
# The query will run on a database with the following schema:
# {context}

# {relations}

# Given the database schema, write correct SQL QUERY for the question/request: '{question}'<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# [SQL]{answer}[/SQL]<|eot_id|>
# """

# llama38binstruct_test_template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# Your are a very helpfull assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
# Your task is to Generate SQL SERVER 2014 QUERY to the user question. 
# Do not add any other explanation, only SQL code.
# Do not hallucinate or infer new information in the SQL
# The query will run on a database with the following schema:
# {context}

# {relations}

# Given the database schema, write correct SQL QUERY for the question/request: '{question}'<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# [SQL]
# """

llama38binstruct_train_template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Your are a very helpfull assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
Your task is to Generate SQL SERVER 2014 QUERY to the user question. 
Do not add any other explanation, only SQL code.
Do not hallucinate or infer new information in the SQL query.

Use the following past examples only as a reference. Use them to guide you, this are only to guide your answer.
EXAMPLES
{examples}
END OF EXAMPLES

The query will run on a database with the following schema:
SCHEMA
{context}

{relations}
END OF SCHEMA

Given the database schema, write correct SQL QUERY for the question/request: '{question}'<|eot_id|><|start_header_id|>assistant<|end_header_id|>
[SQL]{answer}[/SQL]<|eot_id|>
"""

llama38binstruct_test_template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Your are a very helpfull assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
Your task is to Generate SQL SERVER 2014 QUERY to the user question. 
Do not add any other explanation, only SQL code.
Do not hallucinate or infer new information in the SQL
Use the following past examples only as a reference. Use them to guide you, this are only to guide your answer.
EXAMPLES
{examples}
END OF EXAMPLES

The query will run on a database with the following schema:
SCHEMA
{context}

{relations}
END OF SCHEMA


Given the database schema, write correct SQL QUERY for the question/request: '{question}'<|eot_id|><|start_header_id|>assistant<|end_header_id|>
[SQL]
"""

def get_chat_template(model_name):
    """Always return a tuple (train, test)
Keys needed for templates are: context, relations, question, answer"""

    if model_name == "mistral7b":
        return mistral7binstruct_train_template, mistral7binstruct_test_template
    if model_name == "llama38b":
        return llama38binstruct_train_template, llama38binstruct_test_template
    else:
        return "", ""