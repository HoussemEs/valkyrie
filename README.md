# ⚜️ Valkyrie - AI Coding Assistant

A **real-time AI coding assistant** for Angular and Spring Boot projects. Generate, explore, and manage source code with a live console and file explorer interface, powered by **Gemini AI**, **FastAPI**, **WebSockets**, and **ChromaDB**.

![Valkyrie](https://www.houssemhosni.com/valkyrie.png)
*Screenshot of Generated Files*

---

## 🚀 Features

- **Live Console:** Real-time logging and workflow status via WebSockets.
- **Code Generation:** Automatically generate Java/Spring or Angular/TypeScript files based on your task and project context.
- **File Explorer:** Explore generated files, copy to clipboard, or download.
- **Project Embeddings:** Uses ChromaDB + SentenceTransformer for embedding and context retrieval.
- **Smart Context Handling:** Only includes necessary files for code generation.
- **Framework Awareness:** Supports Angular (components, modules, services) and Spring Boot (Controllers, Services, Entities).
- **Project Awareness:** Valkyrie Generates code aligned with your existing project structure and style.
- **Include Backend Option:** Scan and include backend/API structure dynamically.
- **Automatic Reconnection:** WebSocket reconnects automatically if server is unavailable.
- **Angular Feature Analysis:** Detects interceptors, guards, services, modules, components, directives, and pipes.

---

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, FastAPI, WebSocket
- **AI Integration:** Google Gemini API
- **Vector Search & Embeddings:** ChromaDB + SentenceTransformer
- **Deployment:** Runs locally, WebSocket-based communication

---

## ⚡ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/HoussemEs/valkyrie.git
cd valkyrie
```

### 2. Set up Environment

- Python3.9+ is required
- Install Python dependencies:

```bash
pip install fastapi uvicorn chromadb sentence-transformers google-generativeai
```

- Make sure you have a **Google Gemini API key** (replace in `valkyrie.py`):

```python
GEMKEY = "YOUR_API_KEY_HERE" //Free Api Keys work also
```

- Set your local project path:

```python
PROJECTPATH = "C:/Path/To/Your/Projects"
```

### 3. Run the Backend Server

```bash
python valkyrie.py
```

- Server runs at: `ws://127.0.0.1:8000/ws`

### 4. Open Frontend

- Open `index.html` in your browser
- Enter task and context
- Select project folder
- Click **🚀 Run Workflow** to generate code

---

## 📁 File Explorer

- Expand/collapse all files
- Copy content to clipboard
- Download files locally
- Automatic file naming detection for Java & TypeScript classes

---

## 🔧 Workflow

1. **Embed Project Files:** `valkyrie.py` scans project files and indexes them with embeddings.
2. **Retrieve Context:** Finds relevant project files based on your task description.
3. **Ask Gemini AI:** Determines which files are needed for the task.
4. **Retrieve More Accurate Context:** Fetch relevant embeddings for the requested files.
5. **Generate Code:** Gemini generates the requested code and outputs to the file explorer.
6. **Real-Time Updates:** All logs appear in the console in real-time.

---

## 🔧 Best Practices

1. Use **meaningful, conventional naming** for classes and files.
2. Avoid **.spec.ts** test files in embeddings.
3. Provide **precise context** to Gemini for better results.
4. Maintain **consistent project structure** for easier retrieval.
5. Ask **top to bottom**, (Controller/Service then Service/Component then CustomSpecifications)
6. Prepare some presets for Gemini to follow, **automatically.** 

---

## 🖌️ File Naming & Language Support

- **Java:** Classes, Interfaces, Enums → `.java`
- **TypeScript:** Angular classes, interfaces, enums → `.ts`
- **Other:** `.json`, `.yaml`, `.yml`, `.txt` fallback

---

## 📦 Project Structure

```
valkyrie/
├─ index.html         # Frontend interface
├─ valkyrie.py       # Backend AI logic + WebSocket server
├─ README.md          # Project README
```

---

## 🌟 Contributing

Contributions are welcome!  
1. Fork the repository  
2. Create your feature branch (`git checkout -b feature/MyFeature`)  
3. Commit your changes (`git commit -m 'Add some feature'`)  
4. Push to the branch (`git push origin feature/MyFeature`)  
5. Open a Pull Request

---

## ⚖️ License

MIT License © 2025 [Houssem Hosni](https://github.com/HoussemEs)

---

## 🔗 Links

- **Gemini API:** https://developers.generativeai.google/  
- **ChromaDB:** https://www.trychroma.com/  
- **SentenceTransformer:** https://www.sbert.net/

