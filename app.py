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
from rich import print

load_dotenv()

openai_model = "o3-mini"
openai_model_ = "gpt-5-mini"
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

CACHE_DIR = pathlib.Path('/home/jaideep/app/cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
- Use ONLY the RESOURCES provided by the user as data sources.
</RULES>

<REASONING>
- Carefully **interpret the user’s requirements** and think step by step.
- **Break down the workflow** into smaller, clear tasks for individual agents.
</REASONING>
'''

coding_system_prompt = Template('''
<Role>
You are an Intelligent **Data Analysis LLM**.
You produce complete, self-contained Python scripts (step by step) to perform the data analysis tasks as specified by the User.
Choose the workflow suitable for the data source (local files, APIs, or web).
</Role>

<code_standards>
- Always output a full runnable Python script with a `uv` header:
  # /// script
  # dependencies = ["pandas", "duckdb", ...]   # only non-stdlib packages that are imported in code; but nothing else
  # ///
  {your_code_here}
- Do not include standard library modules in dependencies.
- # examples of preferred libraries:
  - **duckdb/pandas** → tabular data; SQL Querying & Analysis of datasets
  - **httpx/requests/playwright(chromium-engine)/pandas** → API/web fetch
  - **lxml/beautifulsoup4** → HTML parsing
  - **matplotlib/seaborn/plotly** → visualization
  - **networkx** → graph analysis
  - **faster-whisper** → audio processing
- **all the above libraries are indicative; choose the best libraries for the task.**; You are free to choose the most effective libraries.
- Scripts must be idempotent: use `$CACHE_DIR` to cache intermediate results and avoid recomputation.
- Use Parquet for large tables, JSON for small intermediates.
- print the the neccessary information to `stdout` for the next iteration.
- Do not call interactive display (`plt.show()`); save plots or encode as base64 PNG.
- Final output must be written to `$JSON_PATH` STRICTLY ADHERING TO THE SCHEMA in the REQUIREMENTS DOCUMENT.
- Handle erros gracefully: don't catch the erros but log them to `stderr` and continue based on the error occurred.
- DO verify the output JSON briefly before finishing.
</code_standards>

<data_sourcing>
- Phase 1: **Inspect** → load or fetch once, summarize structure (schema, keys, tags, etc.)
- Phase 2: **Extract/Transform** → reuse cache, process into structured form,
- Phase 3: **Analyze** → perform analysis, generate insights, and visualizations.
</data_sourcing>

<env_outline>
- system will capture the output code and runs using `uv` runner.
- the `stdout` and `stderr` will be captured and returned to you.
- work iteratively, step by step. you will receive the output of the code executed in the previous steps.
</env_outline>

<caching>
- use `$CACHE_DIR` for caching the data for Intermediate Results/WEB/API responses
- use the cached data for the next iterations
- Completely avoid the recomputations when inputs are unchanged.
</caching>

<step_control>
- Use `"stop": false` for inspection or intermediate steps; `"stop": true` only when final output at `$JSON_PATH` is ready. as per required output format.
</step_control>

<io_policy>
- Always set timeouts and retries with backoff for external I/O.
- Use descriptive headers as User-Agent for HTTP requests.
</io_policy>

<robustness>
- Validate outputs before finishing: ensure `$JSON_PATH` exists and is non-empty.
- Sanity-check on the data; handle missing or malformed data gracefully. DON'T over-engineer.
</robustness>

<output_format>
Your output must strictly match:
{
  "code": "<full Python code>",
  "stop": <true|false>
}
</output_format>
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

coding_llm_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "coding_llm_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The full Python code snippet to be executed. Must be a self-contained program."
                },
                "stop": {
                    "type": "boolean",
                    "description": "Indicates whether code writing should stop (true) if confident or continue refining (false)."
                }
            },
            "required": ["code", "stop"],
            "additionalProperties": False
        },
    },
}


generator_prompt = '''
<Role>
You are a **JSON Schema Merger**. Output only **fenced Python code blocks** that build a `final_output` JSON object by merging values from given files.
</Role>

<Input>
- REQUIRED STRUCTURE that `stdout` of the code must conform to:
- Collected array of JSON objects:
  - json_path → path to a JSON file
  - expected_output → the required schema
  - result → structure of the JSON file
</Input>

<REQUIREMENTS>
- Build a Python variable `final_output` **STRICTLY CONFORMING to the Schema in the `REQUIRED STRUCTURE`.**
- Recursively update `final_output` using values from JSON files in `json_path`.
- If a key is missing, leave dummy values.
- Ignore extra keys in files. || ignore if the file is missing
- Use only standard Python libraries (`json`, `os`).
- Print only `final_output` to `stdout`.
</REQUIREMENTS>

<OUTPUT>
```python
{your_code_here}
```
</OUTPUT>
'''


async def generate_fallback_template_string(questions_txt: str) -> dict:
    """
    Generate a fallback JSON template from questions.txt.
    Returns a dict with a single key "structure":
      - The value is a string containing the JSON template (can be loaded via json.loads).
      - Dummy values are used: 'TODO' for strings, 0 for int, 0.0 for float, [] for list, {} for dict.
    """

    system_prompt = """
    You are a JSON Template Generator.
    Your goal is to produce a **strict JSON template** from the user's requirements.
    You will receive a text containing questions and requirements. which explain the structure of the expected JSON output.

    RULES:
    - Output ONLY a JSON object with a single key "structure".
    - The value of "structure" must be a string containing the JSON template.
    - The JSON template contains all keys mentioned in the requirements with dummy values:
      - 'TODO' for string, 0 for int, 0.0 for float, [] for list, {} for dict.
    - Do NOT include extra text, commentary, or instructions.
    - Preserve nested structures.
    """

    response = await openai_client.chat.completions.create(
        model=openai_model_,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions_txt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "fallback_template_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "structure": {
                            "type": "string",
                            "description": "JSON template as a string, can be loaded using json.loads"
                        }
                    },
                    "required": ["structure"],
                    "additionalProperties": False
                }
            }
        },
        service_tier="priority",
    )

    fallback_template = json.loads(response.choices[0].message.content)
    return fallback_template


def extract_structure(obj):
    """
    Recursively extracts the keys and their value types from a nested JSON object.
    Returns a dict of the same structure, but values are the types as strings.
    """
    if isinstance(obj, dict):
        return {k: extract_structure(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        if not obj:
            return []
        # Preserve every element's structure (don't deduplicate or merge)
        return [extract_structure(e) for e in obj]
    else:
        return type(obj).__name__  # e.g., "str", "int", "list", "dict"


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


async def run_agent(agent):
    agent_id = f"{random.randint(0, 99):02}"
    prompt = agent.get("prompt", "")
    json_path = str(OUTPUT_DIR / f"output_{agent_id}.json")
    expected_output = agent.get("expected_output", "")
    print('Running agent:', agent_id)
    cache_dir = CACHE_DIR / f"agent_{agent_id}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(cache_dir)

    messages = [
        {"role": "system", "content": coding_system_prompt.substitute(JSON_PATH=json_path, CACHE_DIR=cache_dir)},
        {"role": "user", "content": prompt},
        {"role": "user", "content": f"Expected JSON output format:\n{expected_output}"}
    ]

    code_history = []

    try:
        while True:
            response = await openai_client.chat.completions.create(
                model=openai_model_,
                messages=messages,
                response_format=coding_llm_schema,
                service_tier="priority"
            )

            print("LLM Response:", response)

            llm_response = json.loads(response.choices[0].message.content.strip())

            print('\n\n')
            print("LLM Response JSON:", llm_response)

            code = llm_response.get("code", "")

            print('\n\n')
            print("Code to execute:", code)

            result = await execute_code(code)

            print('\n\n')
            print("Execution Result:", result)

            code_history.append({'code': code, 'output': result})

            if 'Error' in result:
                prompt = f"Error executing code: {result['Error']}\nPlease fix the code and try again."
                messages.append({"role": "user", "content": prompt})
                messages.append({"role": "assistant", "content": code})
                continue

            else:
                output = result.get('Output')
                prompt = f'Continue with the execution output from the execution: {output}'

            if llm_response.get("stop"):
                break


    except asyncio.TimeoutError:
        return {"status": "timeout", "code_history": code_history}
    except Exception as e:
        return {"status": "error", "error": str(e), "code_history": code_history}

    return {
        "agent_id": agent_id,
        "json_path": json_path,
        "expected_output": expected_output,
    }


async def final_schema(structure: dict, collected_structure: str):

    messages = [
        {'role': 'system', 'content': generator_prompt},
        {'role': 'user', 'content': f'REQUIRED STRUCTURE: {structure}\n\nCollected responses structure:\n{collected_structure}'},
    ]

    for _ in range(3):
        try:
            response = await openai_client.chat.completions.create(
                model=openai_model_,
                messages=messages,
                service_tier="priority",
            )

            llm_response = response.choices[0].message.content.strip()
            print("Schema LLM Response:", llm_response)

            messages.append({"role": "user", "content": "re-write the code to produce a valid JSON schema."})

            code = re.findall(r"```python(.*?)```", llm_response, re.DOTALL)
            print(code)

            messages.append({"role": "assistant", "content": code})

            if not code:
                continue

            output = await execute_code(code[0]) if code else {}
            schema_str = output.get('Output', '')
            print(schema_str)
            if output.get('Error', ''):
                messages.append({"role": "user", "content": f"Error: {output['Error']}\nPlease fix the code and try again."})
            if not schema_str:
                continue
            schema = json.loads(schema_str)
            print(schema)
            return schema

        except Exception as e:
            messages.append({"role": "user", "content": f"This stdout is not a valid JSON schema: {e}\nPlease fix the code and try again."})

        return None


async def pipeline(llm_payload: dict, structure: dict) -> dict:
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
                            "json_path": json_path,
                            "expected_output": res.get("expected_output", ""),
                            "result": content
                        })
                except Exception as e:
                    collected.append({
                        "json_path": json_path,
                        "expected_output": res.get("expected_output", ""),
                        "error": f"Failed to read JSON: {str(e)}"
                    })

        if len(collected) == 1:
            with open(collected[0]['json_path'], 'r') as f:
                content = json.load(f)
            return content


        structure = [
            {
                'json_path': item['json_path'],
                'expected_output': item['expected_output'],
                'result': extract_structure(item['result']) if 'result' in item else item.get('error', 'No result found')
            }
            for item in collected
        ]


        print("Collected structure:", json.dumps(structure, indent=2))
        output = await final_schema(structure, json.dumps(structure, indent=2))
        return output


    except Exception as e:
        print({"error": f"Failed to parse LLM response: {str(e)}"})
        return None


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

    fall_back_template = await generate_fallback_template_string(questions_txt_content)
    structure = fall_back_template.get('structure', {})
    structure = json.loads(structure) if isinstance(structure, str) else structure


    elapsed = time.monotonic() - start

    try:
        output = await asyncio.wait_for(
            pipeline(llm_payload, structure),
            timeout=180 - elapsed  # Ensure total time does not exceed 180 seconds
        )

        if not output:
            return JSONResponse(content=structure, status_code=200)

    except Exception:
        return JSONResponse(content=structure, status_code=200)
    return JSONResponse(content=output, status_code=200)
