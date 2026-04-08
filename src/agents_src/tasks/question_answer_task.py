from crewai import Task
from pydantic import BaseModel

from src.agents_src.agents.question_answer_agent import qa_agent

class AnswerStructure(BaseModel):
    answer:str
    sources : list[str]
    tool_used: str
    rationale: str

qa_task = Task(
    agent=qa_agent,
    name="Question Answer Task",
    description="""
    Answer the user query "{user_query}" using a Retrieval-Augmented Generation(RAG) pipeline.
    chat_history:"{chat_history}"
    
    Instructions:
    -Retrieve relevant context from the document store
    -Priortize evidence that directly addresses the query
    -Synthesis a clear, accurate answer grounded in the retrieved sorces or chat_history
    -If the query cannot be answered from the knowledge sorces or chat_history, don not generate your own response.
    Insted, stat clearly that the knowledge source does not contain the required information.
    -Provide transparency by including references , tool usage, and reasoning steps
    """,
    expected_output="""
    A structured JSON object with the following fields:
    {
    "answer":"Direct response is the query (1-3 paragraphs, clear  and accurate).
                if no answer is found, return :'The knowledge sorce does not contain the required information.'",
    "sources":["List of documents titles, sections, or citations used(empty list if none)"],
    "tool_used": "None of the retrieval/analysis tool invoked(e.g...RAG Retriever, VectorDB, ChatHistory,etc.)",
    "rationals": "Brief explanation of why this answer was chosen , or why no relevant information was found"
    }
    """,
    output_pydantic=AnswerStructure,
)