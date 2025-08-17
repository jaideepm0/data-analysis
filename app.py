# /// script
# requires-python = '>=3.13'
# dependencies = ['openai', 'fastapi', 'uvicorn', 'python-dotenv', 'chardet', 'asyncio']
# ///

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
import re
import chardet
import json
import time
import random
from string import Template
import pathlib

load_dotenv()

openai_model = "o3-mini"
anthropic_model = "claude-sonnet-4-20250514"


openai_client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)

anthropic_client = AsyncOpenAI(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://api.anthropic.com/v1/"
)

UPLOAD_DIR = pathlib.Path('/home/jaideep/app/uploads')
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = pathlib.Path('/home/jaideep/app/outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator_system_prompt = '''
# Role
You are an **Orchestrator LLM**. Your goal is to coordinate multiple LLM AGENTS to solve a data analysis workflow efficiently and correctly.

# GOAL:
Reading the user’s requirements, you will break down the workflow into smaller tasks and assign each task to a specific AGENT based on the data sources they need to work with.
<RULES>
- schedule the tasks based on the data sources an AGENT per data source to perform analysis
- DON'T ADD OR REMOVE any INFORMATION that is provided in the REQUIREMENTS but make it more clear and concise.
- **For a data source ONLY 1 AGENT SHOULD be assigned to it.**
- Use ONLY the FILES provided by the user as data sources.
</RULES>

<REASONING>
- Carefully **interpret the user’s requirements** and think step by step.
- **Break down the workflow** into smaller, clear tasks for individual agents.
</REASONING>
'''

coding_system_prompt = Template('''
<Role>
You are a **Data Analysis LLM** that works step by step to solve the problem. You **only write fenced code blocks** and never ask the user for input. All execution results, including errors, will be fed back to you for iteration.
</Role>

<code_standards>
- Write idiomatic, elegant Python; full runnable script each turn (no diffs/patches).
- Cache raw data as early as possible (e.g., HTML, API JSON, CSV). Prefer `parquet` over CSV/JSON for large tabular data (while caching).
- Cache transformed datasets at major pipeline steps (cleaned, merged).
- Always use previously cached intermediate results instead of re-computing/re-downloading.
- Use effective approaches to handle large datasets. as streaming, chunking, or database storage.
- Pick the best libraries for the task.
- Dependencies in the `uv` script header must match imports exactly; no unused deps; prefer stdlib when sufficient.
- Logging to stdout is minimal (progress/errors); do not print the final JSON payload.
- When generating plots, do not call interactive display (e.g., `plt.show()`); produce base64 strings if needed.
</code_standards>

<intent>
- **data sourcing** (scraping, API, or file-based); clean and prepare data if any required; analyze, visualize data
- Output results as a JSON file $JSON_PATH with the analysis results as requested
</intent>

<NOTE>
- The stdout and stderr of the code will be captured and returned to you.
- uv, the Python runner, is ALREADY installed. in the environment.
- Always list the dependencies at the top of the code using `script-inline` metadata. as
```python
# /// script
# dependencies = ["duckdb", "pandas", "httpx", ... ]  # write the actual dependencies here based on your code
# ///
{code_here}
```
</NOTE>

<io_policy>
- Always set timeouts and retries with backoff for external I/O.
- Use descriptive identifiers (e.g., User-Agent for HTTP).
- Respect provider limits; avoid excessive parallelism.
- On recoverable errors (network, file, cache, DB), retry or fallback gracefully.
</io_policy>

<output>
Respond only with a fenced code block. as
```python
{script_here}
```
OR
If the task is fully complete, respond ONLY with:
```Hello World!```
(fenced, exact, no variations).
- Never return any other text or foramt.
- Do not include anything outside the fenced code block. and additional explanations or comments.
</output>

<hints>
# Prefer when applicable (optional, not mandatory):
- Web: `httpx`, `playwright`, `lxml`, `beautifulsoup4`
- Inspecting structures: `lxml`, `jmespath`
- Tables/processing: `duckdb`, `pandas` (For large-scale structured data → prefer DuckDB for queries instead of pandas-only.)
- Graphs: `networkx`
- Audio: `faster-whisper`
- JSON writing/encoding: `json`, `base64`
</hints>
''')


response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "orchestrator_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "agents": {
                    "type": "array",
                    "description": "A list of agents (LLMs) that collaboratively complete the workflow.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The complete prompt merging agent-specific instructions with user goals."
                            },
                            "data_sources": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["local", "url"]},
                                        "path": {"type": "string"}
                                    },
                                    "required": ["type", "path"],
                                    "additionalProperties": False
                                },
                                "description": "Sources (local file paths or URLs) this agent needs."
                            },
                            "expected_output": {
                                "type": "string",
                                "description": "A description of the expected JSON output format for this agent's work. The schema itself may vary."
                            }
                        },
                        "required": ["prompt", "data_sources", "expected_output"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["agents"],
            "additionalProperties": False
        }
    }
}

generator_prompt = '''
You are an **LLM Schema Generator**. Your task is to generate a JSON schema based on:

- REQUIREMENTS provided by the user.
- Collected agent outputs (answers) generated by the LLM agents.

Rules:
- The response JSON must strictly adhere to the format mentioned by the user.
- If the collected answers are insufficient to generate the schema:
    - Generate a dummy schema.
    - Fill as best as possible while keeping types consistent.
    - Write empty fields exactly as: `TODO`.

<OUTPUT>
- Respond **only** with a fenced code block as:
```json
{required_json_schema_here}
- Do not include any other text or comments outside the fenced code block.
</OUTPUT>
'''

async def execute_code(code: str) -> str:

    proc = await asyncio.subprocess.create_subprocess_exec(
        'uv', 'run', '-',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    output, error = await proc.communicate(input=code.encode())

    if proc.returncode != 0:
        return {'Error': error.decode().strip()}

    return {'Output': output.decode().strip()}




async def coding_llm(prompt: str, json_path: str, expected_output: str) -> dict:
    """Call the coding LLM to write code & execute it with 1000 iterations."""

    messages = [
        {"role": "system", "content": coding_system_prompt.substitute(JSON_PATH=json_path)},
        {"role": "user", "content": prompt},
        {"role": "user", "content": f"Expected JSON output format:\n{expected_output}"}
    ]

    code_history = {}

    try:
        while True:
            response = await anthropic_client.messages.create(
                model=anthropic_model,
                messages=messages,
                temperature=0.7
            )

            llm_response = response.choices[0].message.content.strip()
            if llm_response.strip() == "```Hello World!```":
                return {"status": "completed", "code_history": code_history, "json_path": json_path}

            code_match = re.findall(r"```python(.*?)```", llm_response, re.DOTALL)
            if not code_match:
                return {"status": "no_code_found", "code_history": code_history}

            code = code_match[0].strip()
            output = await execute_code(code)

            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": code})

            code_history[code] = output

            if output.get("Error"):
                prompt = f"Refine the following code to fix the error: {output['Error']}"
            else:
                prompt = f"The stdout ==> {output.get('Output', '')}"

    except asyncio.TimeoutError:
        return {"status": "timeout", "code_history": code_history}
    except Exception as e:
        return {"status": "error", "error": str(e), "code_history": code_history}


async def run_agent(agent):
    agent_id = f"{random.randint(0, 99):02}"
    prompt = agent.get("prompt", "")
    json_path = str(OUTPUT_DIR / f"output_{agent_id}.json")
    expected_output = agent.get("expected_output", "")

    result = await asyncio.wait_for(
        coding_llm(prompt, json_path, expected_output),
        timeout=140  # Allow 150 seconds for the coding LLM to complete
    )
    return {
        "agent_id": agent_id,
        "json_path": json_path,
        "expected_output": expected_output,
        "result": result
    }


async def final_schema(questions: str, collected: str):

    messages = [
        {'role': 'system', 'content': generator_prompt},
        {'role': 'user', 'content': f'Requiremetns: {questions}\n\nCollected answers:\n{collected}'},
    ]


    for _ in range(3):
        try:
            response = await openai_client.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=0.7
            )

            llm_response = response.choices[0].message.content.strip()

            # Extract JSON inside a ```json ... ``` block
            json_match = re.findall(r"```json(.*?)```", llm_response, re.DOTALL)
            schema_str = json_match[0].strip() if json_match else llm_response

            schema = json.loads(schema_str)
            return schema

        except Exception:
            continue

    return {}


async def pipeline(llm_payload: dict):
    # call the LLM && let it do the work for 3 minutes
    questions = llm_payload.get('questions.txt', '')
    file_data = llm_payload.get('files', {})
    file_data.pop('questions.txt', None)  # Remove questions.txt from file_data
    response = await openai_client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": orchestrator_system_prompt},
            {"role": "user", "content": f"{questions}\n\nFiles: {json.dumps(file_data, indent=2)}" if file_data else questions},
        ],
        response_format=response_format,
    )

    llm_response = response.choices[0].message.content.strip()
    try:
        meta_data = json.loads(llm_response)

        agents = meta_data.get('agents', [])  # this is certains going to exist

        tasks = [asyncio.create_task(run_agent(agent)) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        collected = []

        for res in results:
            if isinstance(res, Exception):
                print(f"Error running agent: {res}")

            json_path = res.get("json_path", "")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        content = json.load(f)
                        # if things are good then nothing much is required
                        collected.append({
                            "expected_output": res.get("expected_output", ""),
                            "result": str(content)
                        })
                except Exception as e:
                    collected.append({
                        "expected_output": res["expected_output"],
                        "error": f"Failed to read JSON: {str(e)}"
                    })

        collected = json.dumps(collected, indent=2)
        return await final_schema(questions, collected)


    except Exception as e:
        return {"error": f"Failed to parse LLM response: {str(e)}"}


@app.post("/api")
async def main(request: Request):

    start = time.monotonic()
    form = await request.form()
    files_data = {}
    questions_txt_content = None

    for key, value in form.items():
        raw_bytes = await value.read()
        if not raw_bytes:
            return JSONResponse(content={"error": "File content is empty"}, status_code=400)

        # detect encoding
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"

        if key == 'questions.txt':
            questions_txt_content = raw_bytes.decode(encoding, errors='ignore')

        short_id = f"{random.randint(0, 99):02}"  # two digits, zero-padded
        safe_name = f"{short_id}_{key}"

        file_path = UPLOAD_DIR / safe_name

        # write raw bytes safely
        with open(file_path, "wb") as f:
            f.write(raw_bytes)


        # store metadata for LLM
        files_data[key] = {
            "name": key,
            "path": str(file_path),
            "size": len(raw_bytes),
            "encoding": encoding
        }


    if 'questions.txt' not in files_data:
        return JSONResponse(content={"error": "questions.txt is required"}, status_code=400)

    llm_payload = {
        "questions.txt": questions_txt_content,
        "files": files_data,
    }

    elapsed = time.monotonic() - start

    output = await asyncio.wait_for(
        pipeline(llm_payload),
        timeout=180 - elapsed  # Ensure total time does not exceed 180 seconds
    )
    return JSONResponse(content=output, status_code=200)
