/* ========================================
   MERYEM SAKIN - MULTI-MODE PORTFOLIO
   Interactive JavaScript
   ======================================== */

document.addEventListener('DOMContentLoaded', () => {
    initModeSwitcher();
    initVSCode();
    initTerminal();
    initInferenceWidget();
    initCopyButtons();
    initProjectModal();

    // Set default background for appstore mode
    document.body.style.background = '#f5f5f7';
});

// ========================================
// PROJECT MODAL
// ========================================
const projectData = {
    callcenter: {
        title: "AI Call Center",
        subtitle: "Real-time End-to-End Voice Pipeline",
        icon: "fas fa-phone-volume",
        description: "Real-time voice AI pipeline: Customer calls are received, transcribed via ASR (Speech-to-Text), processed by LLM for intelligent responses, and delivered back through TTS (Text-to-Speech) with natural voice. Custom fine-tuned models for Turkish language.",
        tech: ["Whisper ASR", "LangChain", "XTTS", "FastAPI", "WebSocket", "Redis"],
        features: [
            "Real-time Turkish ASR transcription",
            "LLM-based intent detection & response generation",
            "Personalized TTS with voice cloning",
            "Sub-500ms latency streaming",
            "Concurrent call handling"
        ]
    },
    nlsql: {
        title: "NL-to-SQL Agent",
        subtitle: "Natural Language → Database Query",
        icon: "fas fa-database",
        description: "LangChain-based agent that translates natural language queries (Turkish/English) into SQL. Supports complex multi-table queries, aggregation, and filtering. Maintains conversational context with memory.",
        tech: ["LangChain", "GPT-4", "PostgreSQL", "ChromaDB", "FastAPI"],
        features: [
            "'Show last month's top products' → SELECT...",
            "Automatic multi-table JOIN generation",
            "Schema-aware query generation",
            "SQL injection protection",
            "Query explanation & optimization suggestions"
        ]
    },
    ocr: {
        title: "Turkish OCR",
        subtitle: "Document Processing Pipeline",
        icon: "fas fa-file-invoice",
        description: "OCR system optimized for Turkish documents. Processes invoices, IDs, and receipts into structured JSON output. Special handling for Turkish characters (ş, ğ, ı, ö, ü, ç).",
        tech: ["PaddleOCR", "OpenCV", "FastAPI", "Tesseract", "PIL"],
        features: [
            "Automatic Invoice No, Date, Amount extraction",
            "Turkish character support (95%+ accuracy)",
            "Handwriting recognition",
            "Table detection & extraction",
            "Batch processing support"
        ]
    },
    defect: {
        title: "Fabric Defect Detection",
        subtitle: "+15% Accuracy Improvement",
        icon: "fas fa-eye",
        description: "Fabric defect detection system developed for AGTEKS textile factory. Achieved +15% accuracy improvement using transfer learning and custom augmentation. Running in real-time on production line.",
        tech: ["YOLOv8", "PyTorch", "OpenCV", "ONNX", "TensorRT"],
        features: [
            "Hole, stain, tear, thread defect detection",
            "Real-time video analysis",
            "Fast adaptation with transfer learning",
            "GPU-optimized inference",
            "Production deployed at AGTEKS"
        ]
    },
    tts: {
        title: "Turkish TTS",
        subtitle: "Voice Cloning & Synthesis",
        icon: "fas fa-volume-up",
        description: "Turkish text-to-speech system. XTTS-based voice cloning creates new voices from 3-5 second audio samples. Custom Turkish preprocessing for natural prosody and intonation.",
        tech: ["XTTS v2", "Coqui TTS", "PyTorch", "FastAPI", "FFmpeg"],
        features: [
            "Voice cloning from 3-5 sec samples",
            "Natural Turkish pronunciation",
            "Emotion & tone control",
            "Streaming audio output",
            "Multi-speaker support"
        ]
    },
    supportiq: {
        title: "SupportIQ",
        subtitle: "AI-Powered Ticket Router",
        icon: "fab fa-github",
        description: "Open-source customer support ticket routing system. Uses GPT-4 for ticket classification, sentiment analysis, and priority scoring. RAG integration for knowledge base.",
        tech: ["FastAPI", "GPT-4", "ChromaDB", "Redis", "PostgreSQL", "React"],
        features: [
            "Automatic ticket categorization",
            "Sentiment & urgency analysis",
            "Smart agent assignment",
            "RAG-based auto-response suggestions",
            "GitHub: github.com/meryemsakin/supportiq"
        ]
    }
};

function initProjectModal() {
    const modal = document.getElementById('project-modal');
    const closeBtn = modal?.querySelector('.modal-close');
    const projectLinks = document.querySelectorAll('.project-link');

    projectLinks.forEach(link => {
        link.addEventListener('click', () => {
            const projectId = link.dataset.project;
            const project = projectData[projectId];
            if (project) {
                openProjectModal(project, projectId);
            }
        });
    });

    closeBtn?.addEventListener('click', () => {
        modal.classList.remove('active');
    });

    modal?.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });
}

function openProjectModal(project, projectId) {
    const modal = document.getElementById('project-modal');

    document.getElementById('modal-icon').innerHTML = `<i class="${project.icon}"></i>`;
    document.getElementById('modal-icon').className = `modal-icon ${projectId}`;
    document.getElementById('modal-title').textContent = project.title;
    document.getElementById('modal-subtitle').textContent = project.subtitle;
    document.getElementById('modal-description').textContent = project.description;

    document.getElementById('modal-tech').innerHTML = project.tech.map(t => `<span>${t}</span>`).join('');
    document.getElementById('modal-features').innerHTML = project.features.map(f => `<li>${f}</li>`).join('');

    modal.classList.add('active');
}

// ========================================
// MODE SWITCHER
// ========================================
function initModeSwitcher() {
    const buttons = document.querySelectorAll('.mode-btn');
    const modes = document.querySelectorAll('.mode-content');

    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;

            // Update buttons
            buttons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update content
            modes.forEach(m => {
                m.classList.remove('active');
                if (m.id === `${mode}-mode`) {
                    m.classList.add('active');
                }
            });

            // Update body background
            updateBodyBackground(mode);
        });
    });
}

function updateBodyBackground(mode) {
    const body = document.body;
    switch (mode) {
        case 'vscode':
            body.style.background = '#1f1f1f';
            break;
        case 'appstore':
            body.style.background = '#f5f5f7';
            break;
        case 'huggingface':
            body.style.background = '#ffffff';
            break;
    }
}

// ========================================
// VS CODE MODE
// ========================================
const codeFiles = {
    lena: {
        filename: 'lena_software.py',
        language: 'python',
        content: `<span class="comment"># lena_software.py - Current Position</span>
<span class="comment"># LENA SOFTWARE | ML Engineering Team</span>
<span class="comment"># Jul 2023 - Present | Istanbul, Turkey</span>

<span class="keyword">class</span> <span class="class">LenaMLEngineer</span>:
    <span class="string">"""
    Production ML Engineer at LENA Software
    Building enterprise AI solutions
    """</span>
    
    <span class="keyword">def</span> <span class="function">__init__</span>(<span class="parameter">self</span>):
        <span class="parameter">self</span>.company = <span class="string">"LENA SOFTWARE"</span>
        <span class="parameter">self</span>.role = <span class="string">"ML Engineer"</span>
        <span class="parameter">self</span>.duration = <span class="string">"Jul 2023 - Present"</span>
        <span class="parameter">self</span>.location = <span class="string">"Istanbul, Turkey"</span>
    
    <span class="keyword">def</span> <span class="function">build_llm_agents</span>(<span class="parameter">self</span>):
        <span class="string">"""NL-to-SQL translation agents with LangChain"""</span>
        <span class="keyword">from</span> langchain <span class="keyword">import</span> LLMChain
        
        <span class="comment"># Natural language → SQL translation</span>
        agent = LLMChain(
            model=<span class="string">"gpt-4"</span>,
            task=<span class="string">"nl_to_sql"</span>
        )
        <span class="keyword">return</span> agent
    
    <span class="keyword">def</span> <span class="function">document_qa_system</span>(<span class="parameter">self</span>):
        <span class="string">"""RAG-based Document & Video QA pipeline"""</span>
        pipeline = {
            <span class="string">"retrieval"</span>: <span class="string">"vector_db"</span>,
            <span class="string">"generation"</span>: <span class="string">"llm"</span>,
            <span class="string">"sources"</span>: [<span class="string">"documents"</span>, <span class="string">"videos"</span>]
        }
        <span class="keyword">return</span> pipeline
    
    <span class="keyword">def</span> <span class="function">turkish_tts</span>(<span class="parameter">self</span>):
        <span class="string">"""Turkish text-to-speech + voice cloning"""</span>
        <span class="keyword">return</span> {
            <span class="string">"model"</span>: <span class="string">"custom_tts"</span>,
            <span class="string">"language"</span>: <span class="string">"tr-TR"</span>,
            <span class="string">"features"</span>: [<span class="string">"voice_cloning"</span>, <span class="string">"emotion"</span>]
        }
    
    <span class="keyword">def</span> <span class="function">vlm_fault_diagnosis</span>(<span class="parameter">self</span>):
        <span class="string">"""Vision-Language Model for industrial QA"""</span>
        <span class="keyword">return</span> <span class="string">"VLM-based defect detection system"</span>

<span class="comment"># Initialize</span>
engineer = LenaMLEngineer()
engineer.build_llm_agents()
<span class="comment"># Output: Agent ready for NL-to-SQL ✓</span>`
    },
    agteks: {
        filename: 'agteks.py',
        language: 'python',
        content: `<span class="comment"># agteks.py - Computer Vision Engineer</span>
<span class="comment"># AGTEKS | Dec 2022 - Jul 2023</span>

<span class="keyword">class</span> <span class="class">AGTEKSEngineer</span>:
    <span class="string">"""
    Computer Vision at AGTEKS
    Focus: Industrial defect detection
    """</span>
    
    <span class="keyword">def</span> <span class="function">__init__</span>(<span class="parameter">self</span>):
        <span class="parameter">self</span>.company = <span class="string">"AGTEKS"</span>
        <span class="parameter">self</span>.role = <span class="string">"CV Engineer"</span>
        <span class="parameter">self</span>.duration = <span class="string">"Dec 2022 - Jul 2023"</span>
    
    <span class="keyword">def</span> <span class="function">fabric_defect_detection</span>(<span class="parameter">self</span>):
        <span class="string">"""YOLOv8 + Transfer Learning"""</span>
        <span class="keyword">from</span> ultralytics <span class="keyword">import</span> YOLO
        
        model = YOLO(<span class="string">"yolov8n.pt"</span>)
        
        <span class="comment"># Transfer learning for fabric defects</span>
        results = model.train(
            data=<span class="string">"fabric_dataset"</span>,
            epochs=<span class="number">100</span>,
            imgsz=<span class="number">640</span>
        )
        
        <span class="comment"># Achieved +15% accuracy improvement</span>
        <span class="keyword">return</span> results
    
    <span class="keyword">def</span> <span class="function">deploy_to_production</span>(<span class="parameter">self</span>):
        <span class="string">"""Docker + Linux deployment"""</span>
        config = {
            <span class="string">"runtime"</span>: <span class="string">"docker"</span>,
            <span class="string">"os"</span>: <span class="string">"linux"</span>,
            <span class="string">"gpu"</span>: <span class="keyword">True</span>
        }
        <span class="keyword">return</span> config

<span class="comment"># Defect detection: +15% accuracy ✓</span>`
    },
    numondial: {
        filename: 'numondial.py',
        language: 'python',
        content: `<span class="comment"># numondial.py - First Professional Role</span>
<span class="comment"># NUMONDIAL DIGITAL | Sep 2022 - Dec 2022</span>

<span class="keyword">class</span> <span class="class">NumondialEngineer</span>:
    <span class="string">"""
    AI/ML at Numondial Digital
    Focus: Surveillance & Detection
    """</span>
    
    <span class="keyword">def</span> <span class="function">__init__</span>(<span class="parameter">self</span>):
        <span class="parameter">self</span>.company = <span class="string">"NUMONDIAL DIGITAL"</span>
        <span class="parameter">self</span>.role = <span class="string">"AI/ML Engineer"</span>
        <span class="parameter">self</span>.duration = <span class="string">"Sep 2022 - Dec 2022"</span>
    
    <span class="keyword">def</span> <span class="function">object_tracking</span>(<span class="parameter">self</span>):
        <span class="string">"""Real-time object detection & tracking"""</span>
        <span class="keyword">return</span> <span class="string">"Multi-object tracking system"</span>
    
    <span class="keyword">def</span> <span class="function">pose_estimation</span>(<span class="parameter">self</span>):
        <span class="string">"""Human pose estimation"""</span>
        <span class="comment"># Achieved +20% accuracy improvement</span>
        <span class="keyword">return</span> {
            <span class="string">"model"</span>: <span class="string">"pose_estimator"</span>,
            <span class="string">"accuracy_boost"</span>: <span class="string">"+20%"</span>
        }
    
    <span class="keyword">def</span> <span class="function">surveillance_system</span>(<span class="parameter">self</span>):
        <span class="string">"""Public transport AI surveillance"""</span>
        <span class="keyword">return</span> <span class="string">"Real-time video analytics"</span>

<span class="comment"># Foundation of ML engineering career</span>`
    },
    readme: {
        filename: 'README.md',
        language: 'markdown',
        content: `<span class="comment"># 👋 Meryem Sakin</span>

<span class="comment">## ML Engineer | 3+ Years Experience</span>

Building production-ready ML systems.

<span class="comment">### 🛠️ Tech Stack</span>

<span class="string">**Languages:**</span> Python, C++, SQL
<span class="string">**ML/DL:**</span> PyTorch, TensorFlow, YOLOv8
<span class="string">**LLM:**</span> LangChain, RAG, Agents
<span class="string">**Deploy:**</span> Docker, Linux, FastAPI

<span class="comment">### 📊 Key Metrics</span>

- +15% Defect Detection Accuracy
- +20% Human Pose Estimation
- 12+ Production Models
- 99.9% System Uptime

<span class="comment">### 📫 Contact</span>

<span class="string">Email:</span> meryemmsakinn@gmail.com
<span class="string">Location:</span> Istanbul, Turkey`
    },
    requirements: {
        filename: 'requirements.txt',
        language: 'text',
        content: `<span class="comment"># Core ML</span>
torch==2.1.0
tensorflow==2.15.0
ultralytics==8.0.0

<span class="comment"># LLM & NLP</span>
langchain==0.1.0
openai==1.3.0
transformers==4.35.0

<span class="comment"># Computer Vision</span>
opencv-python==4.8.0
pillow==10.1.0

<span class="comment"># API & Deploy</span>
fastapi==0.104.0
uvicorn==0.24.0
docker==6.1.0

<span class="comment"># Data</span>
pandas==2.1.0
numpy==1.26.0`
    },
    dockerfile: {
        filename: 'Dockerfile',
        language: 'dockerfile',
        content: `<span class="keyword">FROM</span> python:3.10-slim

<span class="keyword">WORKDIR</span> /app

<span class="comment"># Install dependencies</span>
<span class="keyword">COPY</span> requirements.txt .
<span class="keyword">RUN</span> pip install --no-cache-dir -r requirements.txt

<span class="comment"># Copy application</span>
<span class="keyword">COPY</span> . .

<span class="comment"># Expose port</span>
<span class="keyword">EXPOSE</span> <span class="number">8000</span>

<span class="comment"># Run application</span>
<span class="keyword">CMD</span> [<span class="string">"uvicorn"</span>, <span class="string">"main:app"</span>, <span class="string">"--host"</span>, <span class="string">"0.0.0.0"</span>]`
    },
    config: {
        filename: 'config.json',
        language: 'json',
        content: `{
    <span class="string">"name"</span>: <span class="string">"Meryem Sakin"</span>,
    <span class="string">"role"</span>: <span class="string">"ML Engineer"</span>,
    <span class="string">"experience"</span>: <span class="string">"3+ years"</span>,
    <span class="string">"skills"</span>: {
        <span class="string">"languages"</span>: [<span class="string">"Python"</span>, <span class="string">"C++"</span>, <span class="string">"SQL"</span>],
        <span class="string">"ml_frameworks"</span>: [<span class="string">"PyTorch"</span>, <span class="string">"TensorFlow"</span>],
        <span class="string">"cv"</span>: [<span class="string">"YOLOv8"</span>, <span class="string">"OpenCV"</span>],
        <span class="string">"llm"</span>: [<span class="string">"LangChain"</span>, <span class="string">"RAG"</span>],
        <span class="string">"deploy"</span>: [<span class="string">"Docker"</span>, <span class="string">"Linux"</span>, <span class="string">"FastAPI"</span>]
    },
    <span class="string">"contact"</span>: {
        <span class="string">"email"</span>: <span class="string">"meryemmsakinn@gmail.com"</span>,
        <span class="string">"linkedin"</span>: <span class="string">"https://www.linkedin.com/in/meryem-s-510423221/"</span>,
        <span class="string">"github"</span>: <span class="string">"github.com/meryemsakin"</span>
    },
    <span class="string">"availability"</span>: <span class="string">"immediate"</span>
}`
    },
    education: {
        filename: 'education.py',
        language: 'python',
        content: `<span class="comment"># education.py - Academic Background</span>
<span class="comment"># Yildiz Technical University | Physics Department</span>

<span class="keyword">class</span> <span class="class">Education</span>:
    <span class="string">"""
    Bachelor's in Physics
    YTU - Yildiz Technical University
    Istanbul, Turkey
    """</span>
    
    <span class="keyword">def</span> <span class="function">__init__</span>(<span class="parameter">self</span>):
        <span class="parameter">self</span>.university = <span class="string">"Yildiz Technical University"</span>
        <span class="parameter">self</span>.department = <span class="string">"Physics"</span>
        <span class="parameter">self</span>.degree = <span class="string">"Bachelor of Science"</span>
        <span class="parameter">self</span>.location = <span class="string">"Istanbul, Turkey"</span>
        <span class="parameter">self</span>.graduation = <span class="string">"Expected 2026"</span>
    
    <span class="keyword">def</span> <span class="function">relevant_courses</span>(<span class="parameter">self</span>):
        <span class="string">"""ML-related coursework"""</span>
        <span class="keyword">return</span> [
            <span class="string">"Computational Physics"</span>,
            <span class="string">"Numerical Methods"</span>,
            <span class="string">"Data Analysis"</span>,
            <span class="string">"Statistical Mechanics"</span>,
            <span class="string">"Linear Algebra"</span>,
            <span class="string">"Differential Equations"</span>
        ]
    
    <span class="keyword">def</span> <span class="function">transferable_skills</span>(<span class="parameter">self</span>):
        <span class="string">"""Physics → ML skills"""</span>
        <span class="keyword">return</span> {
            <span class="string">"Mathematical modeling"</span>: <span class="keyword">True</span>,
            <span class="string">"Scientific computing"</span>: <span class="keyword">True</span>,
            <span class="string">"Complex problem solving"</span>: <span class="keyword">True</span>,
            <span class="string">"Data-driven research"</span>: <span class="keyword">True</span>
        }

<span class="comment"># Strong physics background → ML intuition ✓</span>`
    },
    projects: {
        filename: 'projects.py',
        language: 'python',
        content: `<span class="comment"># projects.py - Production Projects Portfolio</span>

<span class="keyword">class</span> <span class="class">ProductionProjects</span>:
    <span class="string">"""
    End-to-end ML projects deployed to production
    """</span>

    <span class="comment"># ═══════════════════════════════════════════</span>
    <span class="comment"># 1. AI CALL CENTER (End-to-End Pipeline)</span>
    <span class="comment"># ═══════════════════════════════════════════</span>
    call_center = {
        <span class="string">"name"</span>: <span class="string">"AI Call Center System"</span>,
        <span class="string">"description"</span>: <span class="string">"""
            Real-time voice AI pipeline:
            ASR (Speech-to-Text) → LLM Processing → TTS (Text-to-Speech)
            
            Features:
            - Turkish ASR with custom fine-tuning
            - LLM for intent detection & response
            - Turkish TTS with voice cloning
            - Low-latency streaming (<500ms)
        """</span>,
        <span class="string">"tech"</span>: [<span class="string">"Whisper"</span>, <span class="string">"LLM"</span>, <span class="string">"TTS"</span>, <span class="string">"FastAPI"</span>, <span class="string">"WebSocket"</span>]
    }

    <span class="comment"># ═══════════════════════════════════════════</span>
    <span class="comment"># 2. TURKISH OCR SYSTEM</span>
    <span class="comment"># ═══════════════════════════════════════════</span>
    turkish_ocr = {
        <span class="string">"name"</span>: <span class="string">"Turkish Document OCR"</span>,
        <span class="string">"description"</span>: <span class="string">"""
            Turkish-optimized document processing:
            - Invoice, ID, receipt extraction
            - Custom Turkish character handling
            - Structured data output (JSON)
        """</span>,
        <span class="string">"accuracy"</span>: <span class="string">"95%+ on Turkish docs"</span>
    }

    <span class="comment"># ═══════════════════════════════════════════</span>
    <span class="comment"># 3. NL-to-SQL AGENT (LangChain)</span>
    <span class="comment"># ═══════════════════════════════════════════</span>
    nl_to_sql = {
        <span class="string">"name"</span>: <span class="string">"Natural Language to SQL"</span>,
        <span class="string">"description"</span>: <span class="string">"""
            Turkish/English → SQL query translation:
            - LangChain-based agent orchestration
            - Multi-table query support
            - Conversational context handling
            
            Example: "Geçen ayın satışlarını göster"
                    → SELECT * FROM sales WHERE date...
        """</span>,
        <span class="string">"tech"</span>: [<span class="string">"LangChain"</span>, <span class="string">"GPT-4"</span>, <span class="string">"PostgreSQL"</span>]
    }

    <span class="comment"># ═══════════════════════════════════════════</span>
    <span class="comment"># 4. FABRIC DEFECT DETECTION (+15%)</span>
    <span class="comment"># ═══════════════════════════════════════════</span>
    fabric_defect = {
        <span class="string">"name"</span>: <span class="string">"Fabric Defect Detection"</span>,
        <span class="string">"description"</span>: <span class="string">"""
            Real-time quality control for textile:
            - YOLOv8-based defect detection
            - Transfer learning optimization
            - +15% accuracy improvement
        """</span>,
        <span class="string">"result"</span>: <span class="string">"Production deployed at AGTEKS"</span>
    }

    <span class="comment"># ═══════════════════════════════════════════</span>
    <span class="comment"># 5. SUPPORTIQ (Open Source)</span>
    <span class="comment"># ═══════════════════════════════════════════</span>
    supportiq = {
        <span class="string">"name"</span>: <span class="string">"SupportIQ - AI Ticket Router"</span>,
        <span class="string">"github"</span>: <span class="string">"github.com/meryemsakin/supportiq"</span>,
        <span class="string">"description"</span>: <span class="string">"""
            AI-powered customer support routing:
            - GPT-4 for ticket classification
            - Sentiment analysis & priority scoring
            - Smart agent assignment
            - RAG for knowledge base
        """</span>,
        <span class="string">"tech"</span>: [<span class="string">"FastAPI"</span>, <span class="string">"GPT-4"</span>, <span class="string">"ChromaDB"</span>, <span class="string">"Redis"</span>]
    }

<span class="comment"># All projects: Production-deployed ✓</span>`
    }
};

function initVSCode() {
    // Load default file
    loadFile('lena');

    // File tree click handlers
    document.querySelectorAll('.file').forEach(file => {
        file.addEventListener('click', () => {
            const fileKey = file.dataset.file;
            if (codeFiles[fileKey]) {
                loadFile(fileKey);

                // Update active state
                document.querySelectorAll('.file').forEach(f => f.classList.remove('active'));
                file.classList.add('active');

                // Add or activate tab
                addOrActivateTab(fileKey);
            }
        });
    });

    // Tab click handlers
    document.querySelector('.editor-tabs').addEventListener('click', (e) => {
        const tab = e.target.closest('.tab');
        if (tab && !e.target.classList.contains('tab-close')) {
            const fileKey = tab.dataset.file;
            loadFile(fileKey);

            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
        }

        // Close tab
        if (e.target.classList.contains('tab-close')) {
            const tab = e.target.parentElement;
            if (document.querySelectorAll('.tab').length > 1) {
                tab.remove();
                // Activate another tab
                const remaining = document.querySelector('.tab');
                if (remaining) {
                    remaining.classList.add('active');
                    loadFile(remaining.dataset.file);
                }
            }
        }
    });
}

function loadFile(fileKey) {
    const file = codeFiles[fileKey];
    if (!file) return;

    const codeContent = document.getElementById('code-content');
    const lineNumbers = document.getElementById('line-numbers');
    const breadcrumb = document.querySelector('.breadcrumb .current');

    // Set code content
    codeContent.innerHTML = file.content;

    // Generate line numbers
    const lines = file.content.split('\n').length;
    let lineNumsHtml = '';
    for (let i = 1; i <= lines; i++) {
        lineNumsHtml += `<span>${i}</span>`;
    }
    lineNumbers.innerHTML = lineNumsHtml;

    // Update breadcrumb
    if (breadcrumb) {
        breadcrumb.textContent = file.filename;
    }
}

function addOrActivateTab(fileKey) {
    const file = codeFiles[fileKey];
    const tabsContainer = document.querySelector('.editor-tabs');
    let existingTab = tabsContainer.querySelector(`[data-file="${fileKey}"]`);

    if (!existingTab) {
        // Create new tab
        const iconClass = getFileIcon(fileKey);
        const newTab = document.createElement('div');
        newTab.className = 'tab';
        newTab.dataset.file = fileKey;
        newTab.innerHTML = `
            <i class="${iconClass.class} file-icon ${iconClass.color}"></i>
            <span>${file.filename}</span>
            <i class="fas fa-times tab-close"></i>
        `;
        tabsContainer.appendChild(newTab);
        existingTab = newTab;
    }

    // Activate tab
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    existingTab.classList.add('active');
}

function getFileIcon(fileKey) {
    const icons = {
        lena: { class: 'fab fa-python', color: 'python' },
        agteks: { class: 'fab fa-python', color: 'python' },
        numondial: { class: 'fab fa-python', color: 'python' },
        education: { class: 'fab fa-python', color: 'python' },
        projects: { class: 'fab fa-python', color: 'python' },
        readme: { class: 'fab fa-markdown', color: 'markdown' },
        requirements: { class: 'fas fa-file-alt', color: 'txt' },
        dockerfile: { class: 'fab fa-docker', color: 'docker' },
        config: { class: 'fas fa-cog', color: 'json' }
    };
    return icons[fileKey] || { class: 'fas fa-file', color: '' };
}

// ========================================
// TERMINAL
// ========================================
function initTerminal() {
    const commands = [
        { cmd: 'python -c "print(\'Loading ML engineer profile...\')"', delay: 0 },
        { cmd: '', output: '\n<span class="success">✓ Profile loaded successfully!</span>\n<span class="info">📊 3+ years experience</span>\n<span class="info">🚀 15+ production models</span>\n<span class="info">📍 Istanbul, Turkey</span>\n<span class="warning">💼 Status: Open to opportunities</span>', delay: 2000 }
    ];

    const commandEl = document.getElementById('terminal-command');
    const outputEl = document.getElementById('terminal-output');

    let charIndex = 0;
    const typeCommand = () => {
        if (charIndex < commands[0].cmd.length) {
            commandEl.textContent += commands[0].cmd[charIndex];
            charIndex++;
            setTimeout(typeCommand, 30);
        } else {
            // Show output after typing
            setTimeout(() => {
                outputEl.innerHTML = commands[1].output;
            }, 500);
        }
    };

    setTimeout(typeCommand, 800);
}

// ========================================
// HUGGING FACE INFERENCE
// ========================================
function initInferenceWidget() {
    const input = document.getElementById('inference-input');
    const button = document.getElementById('compute-btn');
    const output = document.getElementById('inference-output');

    if (!button) return;

    const responses = {
        'llm': "I've built LLM-based agents for NL-to-SQL translation at LENA Software, using LangChain for orchestration. I've also developed RAG pipelines for Document & Video QA systems.",
        'experience': "I have 3+ years of professional ML experience. Started at Numondial Digital, then AGTEKS for computer vision, and currently at LENA Software building LLM agents, TTS systems, and end-to-end AI call center solutions.",
        'computer vision': "Extensive experience with YOLOv8, object detection, and tracking. At AGTEKS, I improved fabric defect detection by 15% using transfer learning. Also worked on real-time pose estimation.",
        'deploy': "I deploy ML models to production using Docker containers on Linux servers. I use FastAPI for APIs and have experience with GPU optimization for inference.",
        'default': "I'm a production-ready ML engineer with 3+ years of experience in LLM agents, computer vision, and TTS systems. Currently at LENA Software, I build NL-to-SQL translation agents, Document QA pipelines, Turkish TTS with voice cloning, and end-to-end AI call center solutions. Feel free to ask about any specific area!"
    };

    const getResponse = (query) => {
        query = query.toLowerCase();
        if (query.includes('llm') || query.includes('langchain') || query.includes('agent')) {
            return responses['llm'];
        } else if (query.includes('experience') || query.includes('year') || query.includes('background')) {
            return responses['experience'];
        } else if (query.includes('vision') || query.includes('yolo') || query.includes('detection') || query.includes('cv')) {
            return responses['computer vision'];
        } else if (query.includes('deploy') || query.includes('docker') || query.includes('production')) {
            return responses['deploy'];
        }
        return responses['default'];
    };

    button.addEventListener('click', () => {
        const query = input.value.trim();
        if (!query) {
            output.textContent = responses['default'];
        } else {
            output.textContent = getResponse(query);
        }
        output.classList.add('show');
    });

    // Also trigger on Enter key
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            button.click();
        }
    });
}

// ========================================
// COPY BUTTONS
// ========================================
function initCopyButtons() {
    document.querySelectorAll('.copy-code-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const codeBlock = btn.closest('.code-block');
            const code = codeBlock.querySelector('code').textContent;

            navigator.clipboard.writeText(code).then(() => {
                const originalHTML = btn.innerHTML;
                btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                setTimeout(() => {
                    btn.innerHTML = originalHTML;
                }, 2000);
            });
        });
    });

    // Model path copy button
    const pathCopyBtn = document.querySelector('.model-path .copy-btn');
    if (pathCopyBtn) {
        pathCopyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText('meryemsakin/production-ml-engineer-v3');
        });
    }
}
