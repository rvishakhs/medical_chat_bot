prompt_template = """
Use the following pieces of information to answer the user's question. Remeber you are dealing with a medical related 
questions so try to get as much as details and provide them in a simple and understanble way.
If you don't know the answer, Just say that you don't know. don't make the answer.

context: {context}
question: {question}

Only return the helpfull answer below and nothing else. 
Helpful answer
"""