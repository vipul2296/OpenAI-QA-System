from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

def query_rag(user_query):
  """
  Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
  Args:
    - query_text (str): The text to query the RAG system with.
  Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
  """
  # YOU MUST - Use same embedding function as before
  embedding_function = OpenAIEmbeddings(model='text-embedding-3-small')

  # Prepare the database
  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
  
  # Retrieving the context from the DB using similarity search
  results = db.similarity_search_with_relevance_scores(user_query, k=3)

  # Check if there are any matching results or if the relevance score is too low
  if len(results) == 0:
    print(f"Unable to find matching results.")

  # Combine context from matching documents
  context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])

  PROMPT_TEMPLATE = "Please answer the following question based on the information provided below. If the answer is not found in the content, please reply with 'don't know'.\n\n{context}\n\nQuestion: {question}"
    
  # Create prompt template using context and query text
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=user_query)
  
  # Initialize OpenAI chat model
  model = ChatOpenAI(model='gpt-4o-mini')

  # Generate response text based on the prompt
  response = model.invoke(prompt)
  print(f"LLM Response: {response.content}")
   # Get sources of the matching documents
#   sources = [doc.metadata.get("source", None) for doc, _score in results]
 
  # Format and return response including generated text and sources
#   formatted_response = f"Response: {response_text}\nSources: {sources}"
  return response, response.content