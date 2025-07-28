Approach Explanation: Persona-Driven Document Intelligence
1. Core Philosophy
The fundamental goal of this project is to create a generic and intelligent system that can understand a user's specific needs and extract relevant information from a diverse collection of PDF documents. The architecture is built on a modular, offline-first pipeline that prioritizes relevance, accuracy, and performance while adhering to strict constraints on model size and execution time.

The core challenge is that a single, broad query (e.g., "plan a trip") is often too general to yield diverse results from a varied document set. Our solution overcomes this by implementing an intelligent Multi-Query Retrieval strategy, which mimics how a human expert would research a topic.

2. System Architecture & Pipeline
Our system follows a robust, multi-stage process for each collection it analyzes:

Stage 1: Document Ingestion and Granular Parsing
Tool: PyMuPDF (fitz)

Method: Instead of treating entire pages or unreliable heuristic-based sections as our units of analysis, we parse each PDF into granular, structural text blocks. This approach provides a much finer level of detail, allowing the retrieval system to pinpoint very specific paragraphs or sections of text. We filter out blocks with fewer than 100 characters to eliminate noise from headers, footers, and stray text.

Stage 2: Dynamic Topic Modeling for Query Expansion
Tool: scikit-learn (TfidfVectorizer)

Method: To ensure our search is not limited to the user's initial query, we first perform a quick analysis of the entire document collection to discover its primary themes. Using TF-IDF (Term Frequency-Inverse Document Frequency), we automatically identify the most important and representative keywords and phrases (n-grams) from the corpus. These discovered topics (e.g., "fillable forms," "coastal adventures," "gluten-free") are used to create additional, highly relevant search queries.

Stage 3: Multi-Query Semantic Retrieval
Tool: sentence-transformers (all-mpnet-base-v2) and faiss-cpu

Method: This is the core of our intelligence layer.

Indexing: All parsed text blocks are encoded into high-dimensional vectors using the powerful all-mpnet-base-v2 model and are indexed in a FAISS vector store for incredibly fast similarity searches.

Querying: The system executes multiple searches against this index: one for the user's main query (persona + job-to-be-done) and one for each of the dynamically discovered topics from Stage 2.

Aggregation: The results from all searches are collected, and duplicates are removed. This multi-pronged approach guarantees that we retrieve a diverse set of relevant sections from across all provided documents, not just the ones that match the most general terms.

Stage 4: AI-Powered Analysis and Titling
Tool: transformers (t5-small)

Method: For each of the top-ranked text blocks retrieved in the previous stage, we use the t5-small model to perform two key tasks:

Summarization (refined_text): The model generates a concise, abstractive summary of the text block, capturing its most important information.

Title Generation (section_title): To provide a clean and highly relevant headline, we then ask the model to generate a short, descriptive title based on the clean summary it just created. This two-step process results in titles that are perfectly aligned with the extracted subsection analysis.

This comprehensive, multi-stage architecture ensures that the final output is not only accurate and compliant with all constraints but also genuinely intelligent and adaptable to any document collection and user request.