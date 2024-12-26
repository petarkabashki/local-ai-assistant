#%%

#%%

from datetime import datetime
import torch
# from transformers import pipeline
import ollama
from pydantic import BaseModel, Field
import gradio as gr
import os
import pandas as pd
import io  # Add this import at the top of your file
from faster_whisper import WhisperModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def create_file(filename, content):
    with open(f'data/{filename}', 'w') as file:
        file.write(content)
    print(f"File '{filename}' created with content: {content}")

def read_file(filename):
    try:
        with open(f'data/{filename}', 'r') as file:
            content = file.read()
        print(f"Content of '{filename}': {content}")
        return content
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None

def delete_file(filename):
    try:
        os.remove(f'data/{filename}')
        print(f"File '{filename}' deleted.")
    except FileNotFoundError:
        print(f"File '{filename}' not found.")


def load_or_create_tasks_csv(file_path="data/tasks.csv"):
    """
    Reads the 'tasks.csv' file if it exists.
    If it does not exist, creates an empty CSV with
    columns: create_date, description, status.
    Returns the tasks DataFrame.
    """
    if os.path.isfile(file_path):
        # If tasks.csv exists, read it into a DataFrame
        tasks_df = pd.read_csv(file_path)
    else:
        # Create an empty DataFrame with desired columns
        tasks_df = pd.DataFrame(columns=["create_date", "description", "status"])
        tasks_df.to_csv(file_path, index=False)
    
    return tasks_df


tasks_df = load_or_create_tasks_csv("data/tasks.csv")

def create_task(task_description):
    """ Creates a new task in the tasks.csv file.

    Args:
        description (string): The description of the task.
    """
    global tasks_df
    
    new_row = {
        "create_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "description": task_description,
        "status": "Not Started"
    }
    tasks_df = pd.concat([tasks_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    tasks_df.to_csv("data/tasks.csv", index=False)

    return tasks_df



def get_response(prompt):
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'create_file',
                'description': 'Create a new file with given content',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'filename': {
                            'type': 'string',
                            'description': 'The name of the file to create',
                        },
                        'content': {
                            'type': 'string',
                            'description': 'The content to write to the file',
                        },
                    },
                    'required': ['filename', 'content'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'read_file',
                'description': 'Read the content of a file',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'filename': {
                            'type': 'string',
                            'description': 'The name of the file to read',
                        },
                    },
                    'required': ['filename'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'delete_file',
                'description': 'Delete a file',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'filename': {
                            'type': 'string',
                            'description': 'The name of the file to delete',
                        },
                    },
                    'required': ['filename'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'create_task',
                'description': 'Create a task from description and current date as create_date',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'task_description': {
                            'type': 'string',
                            'description': 'Task description',
                        },
                    },
                    'required': ['description'],
                },
            },
        },
        
    ]

    response = ollama.chat(
        # model='hf.co/mradermacher/BgGPT-Gemma-2-2.6B-IT-v1.0-i1-GGUF:Q4_K_M',
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}],
        tools=tools,
    )

    print(response)

    
    tool_calls = response['message']['tool_calls']
    
    results = []
    task_table = None
    for tool_call in tool_calls:
        function_name = tool_call['function']['name']
        arguments = tool_call['function']['arguments']
        
        if function_name == 'create_file':
            create_file(arguments['filename'], arguments['content'])
            results.append(f"Created file: {arguments['filename']}")
        elif function_name == 'read_file':
            content = read_file(arguments['filename'])
            results.append(f"Read file: {arguments['filename']}, Content: {content}")
        elif function_name == 'delete_file':
            delete_file(arguments['filename'])
            results.append(f"Deleted file: {arguments['filename']}")
        elif function_name == 'create_task':
            task = create_task(arguments['description'])
            results.append(f"Created task :\n{task}")
        # elif function_name == 'manage_tasks':
        #     if task_table is not None:
        #         task_table, result = manage_tasks(task_table, arguments['action'], arguments['task_index'])
        #         results.append(f"Task management result: {result}")
        #         results.append(f"Updated task table:\n{task_table.to_string()}")
        #     else:
        #         results.append("Error: Task table not created yet.")

    return results, task_table


# pipe = pipeline("automatic-speech-recognition",
#                "openai/whisper-large-v3-turbo",
#                 torch_dtype=torch.float16,
#                 device=device)


model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")

# 2. Define your "system prompt" and how to build the conversation
# SYSTEM_PROMPT = """You are a helpful AI assistant that strictly follows user instructions. 
# You respond only with factual, concise, and direct answers. If unsure, say "I am not certain." 
# Do not add any unrelated information."""
SYSTEM_PROMPT = """Вие сте полезен AI асистент, който стриктно следва потребителските инструкции. 
Вие отговаряте само с фактически, кратки и директни отговори. Ако не сте сигурни, кажете „Не съм сигурен“. 
Не добавяйте несвързана информация."""
def build_prompt(user_message):
    """
        Construct the prompt for the Llama 3.2 Chat model.
        The system prompt is embedded in <<SYS>> ... <</SYS>> inside the [INST] block.
        """
    prompt = f"""[INST] <<SYS>>
    {SYSTEM_PROMPT}
    <</SYS>>

    {user_message}
    [/INST]
    """
    return prompt

def transcribe_and_respond(audio_input):
    if audio_input is None:
        raise gr.Error("No audio file")
    
    # Transcribe the audio
    # transcription = pipe(audio_input, generate_kwargs={"task": "transcribe"}, return_timestamps=True)["text"]
    
    segments, info = model.transcribe(audio_input)
    # print(segments)
    transcription = " ".join(segment.text for segment in segments)
    request = build_prompt(transcription)
    response, task_table = get_response(request)
    
    return transcription, "\n".join(response), tasks_df

# def create_task_table(tasks, statuses, priorities):
#     data = {
#         'Task': tasks.split(','),
#         'Status': statuses.split(','),
#         'Priority': priorities.split(',')
#     }
#     df = pd.DataFrame(data)
#     return df

demo = gr.Interface(
    fn=transcribe_and_respond,
    inputs=[gr.Audio(sources=["microphone", "upload"], type="filepath")],
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="AI Response"),
        gr.Dataframe(label="Table Output")
    ],
    title="Whisper Large V3 Turbo: Transcribe Audio and Get AI Response",
    description="Transcribe audio inputs and get AI responses. Thanks to HuggingFace and Ollama.",
    allow_flagging="never",
)

demo.launch()

#%%