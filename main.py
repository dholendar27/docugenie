import os

from fastapi import FastAPI, UploadFile, HTTPException, status

from chain import convert_into_embeddings, chat
from schema import Question

app = FastAPI()

@app.post('/upload-files/', status_code=status.HTTP_200_OK)
def upload_file(file: UploadFile):
    try:
        contents = file.file.read()
        filepath = os.path.join('../files', file.filename)
        with open(filepath,'wb') as f:
            f.write(contents)
            convert_into_embeddings(filepath)
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Something went wrong')
    finally:
        file.file.close()
    return {"message": f"Successfully uploaded {file.filename}"}

@app.post('/ask', status_code=status.HTTP_200_OK)
def ask(question:Question):
    response = chat(question.question)
    return {"response":response}