# textToSql on Vehicle insurance database

Working on the text to SQL i.e this will convert natural language text into SQL commands to act on the vehicle insurance database that was previosuly created by my group for a college project at: https://github.com/adithyaGHegde/DBMS-PROJECT-TEAM-10-/tree/main

This uses the following:
  * Google gen AI chat or Zephyr model
  * Hugging face embeddings
  * Streamlit for UI
  * Langchain framework
  * Chromadb as a vector store

## Instructions

1. Download and install MySQL: https://dev.mysql.com/downloads/mysql/
   
2. Setup MySQL and remember your root username and password, and set it in the .env file

3.Clone this repository to your local machine using:
```bash
  git clone https://github.com/adithyaGHegde/textToSql-LLMs-Langchain.git
```

4.Navigate to the project directory:
```bash
  cd textToSql-LLMs-Langchain
```

5. Install the requirements
```bash
pip install -r requirements.txt
```

6. In MySQL use the database-creation.sql file in this repo and run the entire code to create the vehicle insurance table

7. Set your HUGGINGFACEHUB_API_TOKEN in the .env file, or your Google API key

8. Run to use the QA application
```bash
streamlit run app.py
```
