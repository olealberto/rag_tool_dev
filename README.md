# rag_tool_dev
Retrieval-Augmented Generation system for matching RFP requirements to historical NIH grants and generating evidence-based support for grant proposals.


**Main python scripts are:**

*application_assistant* --- real pdf query tool for grant sections

*query_pipeline* --- custom query for grant sections

*run_evaluation* --- model evaluation on imported grant pdfs


**Project is divided into 5 phases**

Phase One --- Assesing chunking strategy

Phase Two --- Implementing PubMedBERT on NIH API data

Phase Three --- Embeddings on user data

Phase Four --- Optimizing Hybrid Semantic/Keyword search on user dataset

Phase Five --- Knowledge Graph Layer for additional search results
