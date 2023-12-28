import streamlit as st
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import HuggingFaceHub, GooglePalm

# conn = st.connection('mysql', type='sql')

import os
from dotenv import load_dotenv

# # for streamlit hosting
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

few_shots = [
    {'Question' : "How many unique vehicles do we have?",
     'SQLQuery' : "SELECT count(DISTINCT Vehicle_Id) FROM team10_vehicle",
     'SQLResult': "[(17,)]",
     'Answer' : "17"},
    {'Question': "How many 5 seaters do we have below 2000000?",
     'SQLQuery':"SELECT count(*) FROM team10_vehicle WHERE Vehicle_Number_Of_Seat = 5 AND Vehicle_Value < 2000000",
     'SQLResult': "[(4,)]",
     'Answer': "4"},
    {'Question' : "Retrieve Customer details whose Claim Amount is less than Coverage Amount and Claim Amount is greater than Sum of (CLAIM_SETTLEMENT_ID, VEHICLE_ID, CLAIM_ID, CUST_ID)" ,
      'SQLQuery': "SELECT DISTINCT cus.* FROM team10_claim c INNER JOIN team10_customer cus ON c.Cust_Id = cus.Cust_Id INNER JOIN team10_claim_settlement cs ON cus.Cust_Id = cs.Cust_Id INNER JOIN team10_coverage cov ON cs.Coverage_Id = cov.Coverage_Id WHERE c.claim_amount>(cs.claim_settlement_id+cs.vehicle_id+cs.claim_id+cs.cust_id) AND c.claim_amount<cov.coverage_amount",
      'SQLResult': "[(152, 'Warden', 'Tomcic', datetime.date(1994, 10, 29), 'M', '21, Rostov St., Russia', 7648683324, 'warden@gmail.com', '83742937', 'Single', 982349634), (153, 'Blathnat', 'Otto', datetime.date(1982, 11, 12), 'F', '9549, Prospect Lane, Livonia', 7628483632, 'ottotto@gmail.com', '12389237', 'Divorced', 98345628), (155, 'Nuria', 'Sigurdsson', datetime.date(1969, 4, 27), 'F', '180, West Court, Westland', 9135566527, 'nuriarox@hotmail.com', '8734234', 'Single', 982374897), (159, 'Homer', 'Kumar', datetime.date(1984, 6, 15), 'M', '6897, Rose Quartz Street', 9879863679, 'homerk@gmail.com', '9472364739', 'Divorced', 82977857), (157, 'Javier', 'Hernandez', datetime.date(1989, 5, 12), 'M', '12, Central Avenue, LA', 9285428612, 'jhernan@yahoo.com', '2271512511', 'Single', 31572420)]",
      'Answer' : "[(152, 'Warden', 'Tomcic', datetime.date(1994, 10, 29), 'M', '21, Rostov St., Russia', 7648683324, 'warden@gmail.com', '83742937', 'Single', 982349634), (153, 'Blathnat', 'Otto', datetime.date(1982, 11, 12), 'F', '9549, Prospect Lane, Livonia', 7628483632, 'ottotto@gmail.com', '12389237', 'Divorced', 98345628), (155, 'Nuria', 'Sigurdsson', datetime.date(1969, 4, 27), 'F', '180, West Court, Westland', 9135566527, 'nuriarox@hotmail.com', '8734234', 'Single', 982374897), (159, 'Homer', 'Kumar', datetime.date(1984, 6, 15), 'M', '6897, Rose Quartz Street', 9879863679, 'homerk@gmail.com', '9472364739', 'Divorced', 82977857), (157, 'Javier', 'Hernandez', datetime.date(1989, 5, 12), 'M', '12, Central Avenue, LA', 9285428612, 'jhernan@yahoo.com', '2271512511', 'Single', 31572420)]"},
    {'Question': "Select Customers who have more than one Vehicle, where the premium for one of the Vehicles is not paid and it is involved in accident",
     'SQLQuery' : "SELECT DISTINCT (Vehicle_Id) FROM TEAM10_VEHICLE AS T1 INNER JOIN TEAM10_CUSTOMER AS T2 ON T1.Cust_Id = T2.Cust_Id INNER JOIN TEAM10_PREMIUM_PAYMENT_RECEIPT AS T3 ON T2.Cust_Id = T3.Cust_Id INNER JOIN TEAM10_PREMIUM_PAYMENT AS T4 ON T3.Premium_Payment_ID = T4.Premium_Payment_Id WHERE T1.Vehicle_Number < T4.Premium_Payment_Amount AND Policy_Number = Policy_Id",
     'SQLResult': "[(557001,), (557014,), (557016,), (557007,)]",
     'Answer' : "[(557001,), (557014,), (557016,), (557007,)]"
     },
     {'Question': "Retrieve Customer and Vehicle details who has been involved in an incident and claim status is pending",
     'SQLQuery' : "SELECT * FROM TEAM10_VEHICLE NATURAL JOIN TEAM10_CUSTOMER WHERE Policy_Id IN (SELECT Policy_Number FROM TEAM10_INSURANCE_POLICY WHERE Agreement_ID IN (SELECT Agreement_ID FROM TEAM10_CLAIM WHERE claim_status = 'PENDING'))",
     'SQLResult': "[(156, 557008, '2022156/100/2', None, 'AP12HS1234', 2250000, 'Sedan', 1500, 4, 'Honda', 132642372, 139103882, '1487', 'M18395HG216', 'Ramond', 'Sanchez', datetime.date(1993, 12, 21), 'M', '10, Downing Street, England', 9169428111, 'ramsan@gmail.com', '2174311682', 'Married', 39164821), (158, 557012, '2021158/100/2', 'DSA06735', 'MH34HJ3401', 1200000, 'SUV', 3000, 7, 'Mahindra', 843535178, 312344582, '1950', 'M4378XH211', 'Kiara', 'Advani', datetime.date(1999, 10, 1), 'F', '561, Church Street, Hyderabad', 9776833668, 'kiaraad@gmail.com', '29723614893', 'Married', 65637863), (159, 557014, '2021159/100/1', 'DSB06459', 'AS11GF3872', 1900000, 'SUV', 1000, 5, 'Mahindra', 898765178, 374684582, '1160', 'M8075XH211', 'Homer', 'Kumar', datetime.date(1984, 6, 15), 'M', '6897, Rose Quartz Street', 9879863679, 'homerk@gmail.com', '9472364739', 'Divorced', 82977857), (160, 557015, '2020160/100/1', 'DSB06456', 'LA03LM9183', 1700000, 'SUV', 4000, 7, 'Toyota', 123455178, 365478582, '1690', 'M7568XH211', 'Sara', 'Ujjwal', datetime.date(1998, 2, 18), 'F', '283, MG Road, Delhi', 8798637829, 'ujjwalsara@gmail.com', '5843643572', 'Married', 32142862), (155, 557005, '2021155/100/1', 'DSC08539', 'TS34RG1873', 200000, 'Mini', 800, 4, 'Maruti', 825564417, 322533223, '1655', 'M38725TH183', 'Nuria', 'Sigurdsson', datetime.date(1969, 4, 27), 'F', '180, West Court, Westland', 9135566527, 'nuriarox@hotmail.com', '8734234', 'Single', 982374897)]",
     'Answer' : "[(156, 557008, '2022156/100/2', None, 'AP12HS1234', 2250000, 'Sedan', 1500, 4, 'Honda', 132642372, 139103882, '1487', 'M18395HG216', 'Ramond', 'Sanchez', datetime.date(1993, 12, 21), 'M', '10, Downing Street, England', 9169428111, 'ramsan@gmail.com', '2174311682', 'Married', 39164821), (158, 557012, '2021158/100/2', 'DSA06735', 'MH34HJ3401', 1200000, 'SUV', 3000, 7, 'Mahindra', 843535178, 312344582, '1950', 'M4378XH211', 'Kiara', 'Advani', datetime.date(1999, 10, 1), 'F', '561, Church Street, Hyderabad', 9776833668, 'kiaraad@gmail.com', '29723614893', 'Married', 65637863), (159, 557014, '2021159/100/1', 'DSB06459', 'AS11GF3872', 1900000, 'SUV', 1000, 5, 'Mahindra', 898765178, 374684582, '1160', 'M8075XH211', 'Homer', 'Kumar', datetime.date(1984, 6, 15), 'M', '6897, Rose Quartz Street', 9879863679, 'homerk@gmail.com', '9472364739', 'Divorced', 82977857), (160, 557015, '2020160/100/1', 'DSB06456', 'LA03LM9183', 1700000, 'SUV', 4000, 7, 'Toyota', 123455178, 365478582, '1690', 'M7568XH211', 'Sara', 'Ujjwal', datetime.date(1998, 2, 18), 'F', '283, MG Road, Delhi', 8798637829, 'ujjwalsara@gmail.com', '5843643572', 'Married', 32142862), (155, 557005, '2021155/100/1', 'DSC08539', 'TS34RG1873', 200000, 'Mini', 800, 4, 'Maruti', 825564417, 322533223, '1655', 'M38725TH183', 'Nuria', 'Sigurdsson', datetime.date(1969, 4, 27), 'F', '180, West Court, Westland', 9135566527, 'nuriarox@hotmail.com', '8734234', 'Single', 982374897)]"
     },
     {'Question': "How many Sedans do we have registered in Karnataka?" ,
     'SQLQuery' : "SELECT COUNT(*) FROM team10_vehicle WHERE Vehicle_Type = 'Sedan' AND Vehicle_Registration_Number LIKE 'KA%'",
     'SQLResult': "[(3,)]",
     'Answer': "3",
     }
]

def get_few_shot_db_chain():
    db_user = "root"
    db_password = "sqlpasswd"
    db_host = "localhost"
    db_name = "vehicle_insurance"
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)
    # llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha",model_kwargs={'temperature':0.1,'max-length':512})
    llm = GooglePalm(temperature=0.1)


    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )
    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".
    
    Use the following format:
    
    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    No pre-amble.
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return chain

def handle_user_input(user_question):
    chain = get_few_shot_db_chain()
    response = chain.run(user_question)

    st.header("Answer")
    st.write(response)

def main():
    load_dotenv()
    st.set_page_config(page_title="QA with vehicle database", page_icon=":car:", layout="wide")

    st.header("Talk to your database")
    st.image('TEAM 10 - VEHICLE INSURANCE DBMS PROJECT - PDM.svg', caption='Database diagram')
    st.subheader("You can try some queries like:")
    st.markdown(
    """
    - How many 5 seaters do we have above 2000000?
    - Retrieve insurance policy details if it expires after 2025.
    - How many Sedans do we have registered in Karnataka?
    """
    )
    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:40px;
    }
    </style>
    ''', unsafe_allow_html=True)

    st.subheader("QA System")
    user_question = st.text_input("Ask a question about the vehicle databse:")

    if user_question:
        handle_user_input(user_question)
        

if __name__ == "__main__":
    main()