import os
import pandas as pd
import openpyxl
import pyperclip


def read_data(sheet):
    # Obtiene la ruta del directorio actual del notebook
    current_dir = os.getcwd()

    # Construye la ruta al archivo Excel utilizando la ruta del notebook
    excel_path = os.path.join(current_dir, "../../fine_tune_data.xlsx")

    # Lee el archivo Excel, selecciona la hoja "data" y las columnas necesarias
    dataframe = pd.read_excel(excel_path, sheet_name=sheet)
    return dataframe


def write_data(sheet_name, tuple: tuple):
    """Parameters e.g.:
sheet_name: "datasets"
tuple: colum_letter, index,data"""

    current_dir = os.getcwd()
    # Construye la ruta al archivo Excel utilizando la ruta del notebook
    excel_path = os.path.join(current_dir, "../data/data.xlsx")
    column_letter, index, data = tuple

    try:
        wb = openpyxl.load_workbook(excel_path)
        sheet = wb[sheet_name]
        cell = f"{column_letter}{index+2}"
        sheet[cell] = data

        wb.save(excel_path)
        print(f"Dato grabado exitosamente en la hoja {sheet_name}, celda {cell}.")

    except Exception as e:
        print(f"Error al grabar el dato en el Excel: {e}")


def get_gpt_relationships_prompt(texts: list[str]):
    """Prompt para chatgpt que busca las relaciones SQL de una consulta SQL.\nEn el prompt le decimos que va a enviarsele una lista de consultas SQL y que su salida debe ser las relaciones que encuentra por medio de los JOINS:
    \nEjm:
    SQL QUERY 5: SELECT Name FROM actor WHERE Age <> 20
    Response: -- No JOIN relationships
    SQL QUERY 6: SELECT T1.Title, T2.Publication_Date FROM book AS T1 JOIN publication AS T2 ON T1.Book_ID = T2.Book_ID
    Response: -- book.Book_ID can be joined with publication.Book_ID
    """

    prompt = """You will be given a list of SQL QUERYS
Your task is to find JOIN relationships in each SQL QUERY.

FOLLOW THIS EXAMPLE FOR EACH SQL QUERY:
SQL QUERY:
SELECT DISTINCT T1.creation FROM department AS T1 JOIN management AS T2 ON T1.department_id = T2.department_id JOIN head AS T3 ON T2.head_id = T3.head_id WHERE T3.born_state = 'Alabama'

ANSWER FORMAT FOR EACH SQL QUERY:
answer = [
...,
'-- department.department_id can be joined with management.department_id, -- management.head_id can be joined with head.head_id',
...
]

YOUR FINAL ANSWER MUST BE A COMMA SEPARATED LIST OF THE ANSWERS OF EACH SQL QUERY
"""
    n = """\nSQL QUERY {}: """
    for index, answer in enumerate(texts):
        prompt += n.format(index + 1) + answer

    return prompt

def get_gpt_comments_prompt(texts: list[str]):
    """Prompt para chatgpt que busca las relaciones SQL de una consulta SQL.\nEn el prompt le decimos que va a enviarsele una lista de consultas SQL y que su salida debe ser las relaciones que encuentra por medio de los JOINS:
    \nEjm:
    SQL QUERY 5: SELECT Name FROM actor WHERE Age <> 20
    Response: -- No JOIN relationships
    SQL QUERY 6: SELECT T1.Title, T2.Publication_Date FROM book AS T1 JOIN publication AS T2 ON T1.Book_ID = T2.Book_ID
    Response: -- book.Book_ID can be joined with publication.Book_ID
    """

    prompt = """You will be given a list of SQL DDLS
Your task is to add a comment per column about the meaning of each one.

EXAMPLE:
DDL 1:
CREATE TABLE head (\n    age INTEGER\n)\n
CREATE TABLE table_name_77 (\n    home_team VARCHAR,\n    away_team VARCHAR\n)
DDL 2:
CREATE TABLE table_14656147_2 (\n    week VARCHAR,\n    record VARCHAR\n)
...

YOUR OUTPUT:
answer = [
    '''
    CREATE TABLE IF NOT EXISTS head (
        age INTEGER -- Respective age
    );
    CREATE TABLE IF NOT EXISTS table_name_77 (
       home_team VARCHAR, -- Name of the home team
       away_team VARCHAR -- Name of the away team
    );
    ''',
    '''
    CREATE TABLE IF NOT EXISTS table_14656147_2 (
        week VARCHAR, -- Week number
        record VARCHAR -- Record information
    );
    ''',
    ...
]
END OF EXAMPLE

DDLS LIST:

"""
    n = """\nDDL {}: """
    for index, answer in enumerate(texts):
        prompt += n.format(index + 1) + answer

    return prompt
