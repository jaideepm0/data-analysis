# data analysis agent

## Description
This is a data analysis system that can perform various data analysis tasks such as data cleaning, transformation, and visualization. <br>
This is an MULTI AGENT SYSTEM as:
 - Orchestrator: the main agent that coordinates the other coding agents
 - Coding Agent: the agent that performs the coding tasks
 - Merging Agent: the agent that merges the results of the coding agents
 - Schema Agent: the agent that defines the Initial structure of the data


## Setup
 - Requires full environment with complete support to `uv` The Python runner
 - `OPENAI_API_KEY` must be set in the environment

## Usage
- run the `app.py` using `uvicorn` server as
```bash
uvicorn app:app --reload
```
- The API will be available at `http://localhost:8000`
- You can use the API to send data analysis tasks and receive results
- Querying the API sample
```bash
curl 'http://localhost:8000/api' \
-F 'questions.txt=@questions.txt' \
-F 'file.file=@file.file'
```
- Can send any no of files to the endpoint
- The API will return the results of the data analysis tasks in JSON format mentioned in the `questions.txt` file

## Deployment
- Currently, the system is deployed to using `ngrok` as found very hard to replicate a full environment with `uv` support and all the core libraries
