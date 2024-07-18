import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import pickle as pk
import numpy as np

# load the model
with open("lin_model.pkl", "rb") as file:
    lin_model = pk.load(file)

# create app instance 
app = FastAPI()

# Configuring CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

#  a pydantic class for the request
class StudentPerformanceRequest(BaseModel):
    hours_studied: float = Field(gt=0, lt=24)
    previous_scores: float = Field(gt=0, lt=100)
    extracurricular_activities: str = Field(pattern="^(Yes|No)$")
    sleep_hours: float = Field(gt=0, lt=24)
    sample_question_papers_practiced: int = Field(gt=0, lt=1000)

# creating a test route
@app.get("/class")
async def get_greet():
    return 'Greetings'

# creating a root rout
@app.get("/", status_code=status.HTTP_200_OK)
async def get_hello():
    return {"You're": "welcome"}

# making the prediction route
@app.post('/predict', status_code=status.HTTP_200_OK)
async def make_prediction(request: StudentPerformanceRequest):
    try:
        extracurricular = 1 if request.extracurricular_activities == 'Yes' else 0
        input_data = np.array([[request.hours_studied, request.previous_scores, extracurricular, request.sleep_hours, request.sample_question_papers_practiced]])
        prediction = lin_model.predict(input_data)
        return {"predicted_performance_index": round(prediction[0], 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
