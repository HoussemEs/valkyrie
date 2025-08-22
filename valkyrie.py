import os
import asyncio
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# ===============================
# ENV (Only GEMKEY + PROJECTPATH fixed here)
# ===============================
GEMKEY = "API_KEY"
PROJECTPATH = "C:/path/to/projects_folder"

# Framework modes
SUPPORTED_FRAMEWORKS = {
    "spring": "You are a Spring Boot coding assistant. Generate Java classes such as Controllers, Services, Entities. Only return valid Java code files.",
    "angular": "You are an Angular coding assistant. Generate Angular components, services, and modules using TypeScript. Only return valid Angular code files.",
}

# ===============================
# WebSocket Logger
# ===============================
class WebSocketLogger:
    def __init__(self):
        self.clients = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.clients.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.clients:
            self.clients.remove(websocket)

    async def send(self, msg: str):
        """Send message to all clients immediately (real-time)."""
        dead_clients = []
        for ws in self.clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead_clients.append(ws)
        for dc in dead_clients:
            self.disconnect(dc)

    async def log(self, msg: str):
        """Log locally + send to clients instantly."""
        print(msg, flush=True)   # flush ensures console updates immediately
        await self.send(msg)

logger = WebSocketLogger()

# ===============================
# Init AI & Chroma
# ===============================
print("🔑 Configuring Gemini API...")
genai.configure(api_key=GEMKEY)

print("🗂️ Initializing ChromaDB client...")
chroma_client = chromadb.Client()

print("📦 Creating/getting collection 'project_code'...")
collection = chroma_client.get_or_create_collection(
    name="project_code",
    embedding_function=None
)

print("🧠 Loading local embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# Embedding + retrieval (real-time logs)
# ===============================
async def embed_project(path=PROJECTPATH):
    await logger.log(f"📂 Scanning project folder: {path}")
    file_count = 0

    if not os.path.exists(path):
        await logger.log(f"⚠️ Path does not exist: {path}")
        return

    # Walk project structure
    for root, dirs, files in os.walk(path):
        await logger.log(f"🔎 Entering directory: {root}")

        if len(files) == 0:
            await logger.log(" (No files in this directory)")

        for f in files:
            file_path = os.path.join(root, f)
            await logger.log(f" 📝 Found file: {file_path}")

            if f.endswith(".java") or (f.endswith(".ts") and not f.endswith(".spec.ts")):
                try:
                    # Read file content in a thread to avoid blocking
                    code = await asyncio.to_thread(lambda: open(file_path, encoding="utf-8").read())

                    # Generate embedding in a thread
                    vec = await asyncio.to_thread(lambda: embedder.encode([code])[0].tolist())

                    # Add to collection (thread-safe)
                    await asyncio.to_thread(lambda: collection.add(
                        documents=[code],
                        embeddings=[vec],
                        metadatas=[{"path": file_path}],
                        ids=[file_path]
                    ))

                    file_count += 1
                    await logger.log(f"   ✅ Indexed {f}")

                except Exception as e:
                    await logger.log(f"   ⚠️ Skipped {f} (error: {e})")

    await logger.log(f"✅ Finished indexing. Total indexed: {file_count} source files")


async def retrieve_context(query, conformity_rate: float = 0.75, max_files: int = 150):
    """
    Retrieves context based on similarity score and conformity rate.
    
    :param query: search query
    :param conformity_rate: min similarity (0-1). Higher = stricter.
    :param max_files: max files to include, even if many pass threshold
    """
    await logger.log(f"🔎 Searching in embeddings for: '{query}' (threshold={conformity_rate})")
    
    vec = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[vec], n_results=50)  # ask for more, filter later

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0] if "distances" in results else None

    if not docs:
        await logger.log("⚠️ No matching files found for this query.")
        return ""

    selected_docs = []
    await logger.log("📂 Scoring candidate files...")

    for i, doc in enumerate(docs):
        path = metas[i]["path"]

        if path.endswith(".spec.ts"):
            await logger.log(f"   Skipping test file: {path}")
            continue
    
        score = 1 - dists[i] if dists else 0  # convert distance → similarity (0-1)
        await logger.log(f"   {i+1}. {path} (score={score:.2f})")

        if score >= conformity_rate:
            selected_docs.append(doc)

        if len(selected_docs) >= max_files:
            break

    if not selected_docs:
        await logger.log("⚠️ No files met conformity threshold.")
    else:
        await logger.log(f"✅ Selected {len(selected_docs)} files (rate ≥ {conformity_rate})")

    return "\n\n".join(selected_docs)

# ===============================
# Code Generation (threaded Gemini call)
# ===============================
async def generate_code(task, context, framework="spring"):
    if framework not in SUPPORTED_FRAMEWORKS:
        await logger.log(f"⚠️ Unsupported framework: {framework}, defaulting to Spring Boot")
        framework = "spring"

    await logger.log(f"🤖 Sending task to Gemini for {framework}: '{task}'")

    await logger.log(f"🤖 assumed project context: \n\n{context}\n\n")

    system_prompt = SUPPORTED_FRAMEWORKS[framework]
    prompt = f"""
{system_prompt}

Task: {task}

Relevant project context:
{context}

⚠️ Only return the new source code files needed.
"""

    model = genai.GenerativeModel("gemini-2.0-flash")

    # ✅ run blocking Gemini call in separate thread
    def blocking_call():
        return model.generate_content(prompt)

    await logger.log("📡 Waiting for Gemini response...")
    response = await asyncio.to_thread(blocking_call)

    await logger.log("✅ Gemini responded with code")
    await logger.log("\n📄 ==== GENERATED CODE START ====\n")
    await logger.log(response.text)
    await logger.log("\n📄 ==== GENERATED CODE END ====\n")

    return response.text


# ===============================
# Helper: List all projects and their frameworks for Gemini
# ===============================
def list_projects_gemini():
    """
    Returns a summary string of all projects under PROJECTPATH,
    including their framework.
    """
    summaries = []
    for name in os.listdir(PROJECTPATH):
        path = os.path.join(PROJECTPATH, name)
        if os.path.isdir(path):
            framework = "Angular" if "angular" in name.lower() or "ff" in name.lower() else "Spring Boot"
            summaries.append(f"- {name} (Framework: {framework})")
    return "\n".join(summaries)

# ===============================
# Updated: Gemini file request step (project_summary preserved)
# ===============================
async def request_needed_files(task, framework, project_summary=""):
    """
    Ask Gemini which files are required for generating the requested task.
    Preserves the optional project_summary argument.
    Dynamically includes all projects and frameworks from PROJECTPATH.
    """
    # Append dynamic project summary to existing summary
    dynamic_summary = list_projects_gemini()
    if project_summary:
        full_summary = f"{project_summary}\n\nAvailable projects:\n{dynamic_summary}"
    else:
        full_summary = f"Available projects:\n{dynamic_summary}"

    system_prompt = SUPPORTED_FRAMEWORKS[framework]
    
    prompt = f"""
{system_prompt}

Task: {task}

Project summary / context:
{full_summary}

⚠️ Only list the files (paths or class names) that are strictly required to generate the requested code, either Controllers, Entities, Models, Config Files..
Do not generate the code yet.
Provide output as a JSON array of filenames or class names.
AND DO NOT ASK FOR A FILE THAT DOES NOT EXIST ON THE GIVEN LIST
"""
    model = genai.GenerativeModel("gemini-2.0-flash")

    def blocking_call():
        return model.generate_content(prompt)

    await logger.log("📡 Asking Gemini which files are needed...")
    response = await asyncio.to_thread(blocking_call)
    text = response.text.strip()
    
    # Remove code fences if present
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])

    await logger.log(f"📄 Gemini requested files:\n{text.strip()}")
    
    # Try parsing JSON, fallback to raw lines
    try:
        import json
        files_list = json.loads(text)
        if not isinstance(files_list, list):
            files_list = text.splitlines()
    except Exception:
        files_list = text.splitlines()

    return [f.strip() for f in files_list if f.strip()]


# ===============================
# Log Candidate Files For Gemini
# ===============================
async def log_candidate_files_for_gemini(docs, metas):
    """
    Logs which files and metadata are being offered to Gemini for selection.
    """
    await logger.log("📂 Sending the following files and metadata to Gemini for selection:")
    for i, doc in enumerate(docs):
        path = metas[i].get("path", "unknown")
        snippet = doc[:100].replace("\n", "\\n")  # first 100 chars
        await logger.log(f"   {i+1}. {path} | snippet: {snippet} ...")



# ===============================
# Updated run_workflow
# ===============================
async def run_workflow(task, context_query, framework="spring", folder_name="MasterMP", include_backend=0):
    project_path = os.path.join(PROJECTPATH, folder_name, "src")
    await logger.log("🚀 Starting agent workflow...")
    await embed_project(project_path)

    # Extra Angular analysis
    if framework == "angular":
        await analyze_angular_project(project_path)

    # Secondary embedding for backend if requested
    if include_backend == 1:
        await logger.log("🔁 Backend mode enabled → scanning for additional API/backend files...")
        for subfolder in os.listdir(PROJECTPATH):
            if subfolder != folder_name:
                backend_path = os.path.join(PROJECTPATH, subfolder, "src")
                if os.path.exists(backend_path):
                    await logger.log(f"📂 Including backend project: {subfolder}")
                    await embed_project(backend_path)

    # Step Sub_Zero: Log candidate files
    vec = embedder.encode([context_query])[0].tolist()
    results = collection.query(query_embeddings=[vec], n_results=50)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Log files and metadata
    await log_candidate_files_for_gemini(docs, metas)

    # Now send these files to Gemini to ask which ones are needed
    needed_files = await request_needed_files(task, framework, project_summary="\n".join(docs))

    if needed_files:
        await logger.log(f"📂 Fetching embeddings for requested files: {needed_files}")
        # Build context with only selected files
        context_parts = []
        for f in needed_files:
            context_text = await retrieve_context(f, conformity_rate=0.0, max_files=1)  # fetch 1 match per requested file
            if context_text:
                context_parts.append(context_text)
        ctx = "\n\n".join(context_parts)
    else:
        # fallback: retrieve by query as before
        ctx = await retrieve_context(context_query)

    # Step 2: Generate code
    await generate_code(task, ctx, framework)

# ===============================
# FastAPI + WebSocket
# ===============================
app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await logger.connect(websocket)
    try:
        await logger.log("👤 Client connected to WebSocket console")

        while True:
            data = await websocket.receive_text()
            try:
                msg = eval(data) if data.strip().startswith("{") else None
            except Exception:
                msg = None
                
            if msg.get("action") == "list_projects":
                projects = list_projects()
                await websocket.send_json({"type":"project_list", "projects":projects})
                continue

            if msg and "task" in msg and "context" in msg:
                task = msg["task"]
                context = msg["context"]
                framework = msg.get("framework", "spring")
                folder = msg.get("folder", "MasterMP")  # default folder
                include_backend = msg.get("include_backend", 0)
                asyncio.create_task(run_workflow(task, context, framework, folder, include_backend))
            else:
                await logger.log(f"⚠️ Invalid message received: {data}")
    except WebSocketDisconnect:
        logger.disconnect(websocket)


# ===============================
# Angular Project Helpers
# ===============================
ANGULAR_KEY_FEATURES = [
    "interceptor",
    "authguard",
    "guard",
    "service",
    "module",
    "component",
    "directive",
    "pipe",
]

async def analyze_angular_project(path: str):
    """Scans Angular project for important features (interceptors, guards, etc.)."""
    found_features = []
    if not os.path.exists(path):
        await logger.log(f"⚠️ Angular project path not found: {path}")
        return found_features

    await logger.log("🔎 Analyzing Angular project for key features...")
    
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".ts") and not f.endswith(".spec.ts"):
                lower_name = f.lower()
                for feature in ANGULAR_KEY_FEATURES:
                    if feature in lower_name:
                        feature_path = os.path.join(root, f)
                        await logger.log(f"✨ Found Angular {feature}: {feature_path}")
                        try:
                            # Read file asynchronously
                            code = await asyncio.to_thread(lambda: open(feature_path, encoding="utf-8").read())

                            # Compute embedding asynchronously
                            vec = await asyncio.to_thread(lambda: embedder.encode([code])[0].tolist())

                            # Add to collection asynchronously
                            await asyncio.to_thread(lambda: collection.add(
                                documents=[code],
                                embeddings=[vec],
                                metadatas=[{"path": feature_path}],
                                ids=[feature_path]
                            ))

                            found_features.append(feature_path)

                        except Exception as e:
                            await logger.log(f"⚠️ Could not index {feature_path}: {e}")

    if not found_features:
        await logger.log("⚠️ No special Angular features found.")
    else:
        await logger.log(f"✅ Collected {len(found_features)} Angular feature files")

    return found_features

# ===============================
# PROJECT DIR AWARENESS
# ===============================
def list_projects():
    projects = []
    for name in os.listdir(PROJECTPATH):
        path = os.path.join(PROJECTPATH, name)
        if os.path.isdir(path):
            if "angular" in name.lower() or "ff" in name.lower():
                projects.append({"name": name, "type": "Angular", "emoji": "⚡"})
            else:
                projects.append({"name": name, "type": "Java", "emoji": "🌱"})
    return projects


# ===============================
# Entry point
# ===============================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
