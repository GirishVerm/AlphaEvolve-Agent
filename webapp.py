from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import difflib
import os
import asyncio
from evo_agent.guided_agent import GuidedAgent, AgentConfig, TaskSpec

app = FastAPI()

# Set up templates directory
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')
os.makedirs(TEMPLATES_DIR, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Mount static directory for CSS/JS
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

# Singleton agent for demo
agent = GuidedAgent(AgentConfig())

# Helper for async in sync context
loop = asyncio.get_event_loop()

def get_diff(old, new):
    diff = difflib.ndiff(old.splitlines(), new.splitlines())
    html = ''
    for line in diff:
        if line.startswith('+'):
            html += f'<div style="color:green;">{line}</div>'
        elif line.startswith('-'):
            html += f'<div style="color:red;">{line}</div>'
        else:
            html += f'<div>{line}</div>'
    return html

@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    prompts = list(agent.prompts.values())
    code = [agent.initial_code or ""]
    if agent.task_history:
        code = [h['code'] for h in agent.task_history]
    prompt_diff = get_diff(prompts[-2] if len(prompts) > 1 else prompts[0], prompts[-1])
    code_diff = get_diff(code[-2] if len(code) > 1 else code[0], code[-1])
    return templates.TemplateResponse('index.html', {
        'request': request,
        'task': agent.task.task_name if agent.task else '',
        'prompts': prompts,
        'code': code,
        'prompt_diff': prompt_diff,
        'code_diff': code_diff,
    })

@app.post('/set_task')
def set_task(task: str = Form(...)):
    # For demo, treat input as task name, use defaults for other fields
    agent.task = TaskSpec(
        task_name=task,
        description=f"Task: {task}",
        requirements=["Requirement 1", "Requirement 2"],
        success_criteria=["Success 1", "Success 2"]
    )
    agent.prompts = agent.initial_prompts.copy()
    agent.task_history = []
    agent.initial_code = None
    return {'ok': True}

@app.post('/evolve')
def evolve():
    # Evolve prompt and code (single step)
    async def do_evolve():
        # Evolve prompts (just code_generation for demo)
        prev_prompt = agent.prompts['code_generation']
        await agent.evolve_agent_components()
        new_prompt = agent.prompts['code_generation']
        # Evolve code
        if not agent.initial_code:
            code = await agent.generate_initial_code()
            agent.initial_code = code
            agent.task_history.append({'code': code, 'evaluation': {'overall': 0.5}})
            prev_code = ""
        else:
            prev_code = agent.task_history[-1]['code']
            code = await agent.improve_code(prev_code)
            agent.task_history.append({'code': code, 'evaluation': {'overall': 0.5}})
        return {
            'prompt_diff': get_diff(prev_prompt, new_prompt),
            'code_diff': get_diff(prev_code, code),
            'prompts': list(agent.prompts.values()),
            'code': [h['code'] for h in agent.task_history],
        }
    data = loop.run_until_complete(do_evolve())
    return JSONResponse(content=data)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
