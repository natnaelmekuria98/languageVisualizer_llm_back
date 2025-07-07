import os
import logging
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, validator
import uvicorn
import pandas as pd
from dotenv import load_dotenv
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import time
import sys
from PIL import Image
import io
import base64
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
from dataclasses import field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available options
AVAILABLE_OPENAI_MODELS = [
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-32k",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k"
]

AVAILABLE_LIBRARIES = [
    "plotly",
    "altair",
    "geopandas",
    "folium",
    "matplotlib",
    "seaborn",
    "pydeck",
    "networkx",
    "pyvis",
    "dash"
]

# FastAPI App Setup
app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://langviz-llm-app.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Request/Response Models
class DatasetInfo(BaseModel):
    label: str
    url: Optional[str] = None

class SummarizeRequest(BaseModel):
    dataset_path: str
    summary_method: str = "llm"
    model: str = "gpt-4"
    temperature: float = 0.0
    use_cache: bool = True

    @validator('model')
    def validate_model(cls, v):
        if v not in AVAILABLE_OPENAI_MODELS:
            raise ValueError(f"Model must be one of {AVAILABLE_OPENAI_MODELS}")
        return v

class GoalResponse(BaseModel):
    question: str
    visualization: str
    rationale: str

class VisualizationResponse(BaseModel):
    code: str
    specification: Dict
    raster: Optional[str] = None
    vector: Optional[str] = None

class FileUploadResponse(BaseModel):
    status: str
    file_path: str
    message: Optional[str] = None

class EditVisualizationRequest(BaseModel):
    summary: Dict
    code: str
    instructions: str
    library: str = "plotly"
    model: str = "gpt-4"
    temperature: float = 0.0
    use_cache: bool = True

    @validator('model')
    def validate_model(cls, v):
        if v not in AVAILABLE_OPENAI_MODELS:
            raise ValueError(f"Model must be one of {AVAILABLE_OPENAI_MODELS}")
        return v

    @validator('library')
    def validate_library(cls, v):
        if v.lower() not in AVAILABLE_LIBRARIES:
            raise ValueError(f"Library must be one of {AVAILABLE_LIBRARIES}")
        return v.lower()

# Service Class (no changes needed to the service class)
class LIDAService:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")

        self.data_dir = Path("data").absolute()
        self.data_dir.mkdir(exist_ok=True)
        self.start_time = time.time()

    async def upload_dataset(self, file: UploadFile) -> str:
        """Handle file upload and return filename"""
        try:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ['.csv', '.json']:
                raise ValueError("Only CSV and JSON files are supported")
            
            contents = await file.read()
            if len(contents) == 0:
                raise ValueError("Uploaded file is empty")
            
            original_name = Path(file.filename).stem
            safe_name = f"{original_name}{file_ext}"
            save_path = self.data_dir / safe_name
            
            counter = 1
            while save_path.exists():
                safe_name = f"{original_name}_{counter}{file_ext}"
                save_path = self.data_dir / safe_name
                counter += 1
                
            with open(save_path, 'wb') as f:
                f.write(contents)
                
            if not save_path.exists():
                raise IOError("Failed to save file")
                
            return safe_name
            
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            raise

    async def summarize_data(self, filename: str, **kwargs) -> Dict:
        """Summarize data using filename"""
        try:
            dataset_path = self.data_dir / filename

            if not dataset_path.exists():
                available = "\n".join([f.name for f in self.data_dir.glob('*')])
                raise FileNotFoundError(
                    f"File not found: {filename}\n"
                    f"Available files: {available}"
                )

            try:
                if dataset_path.suffix == '.csv':
                    df = pd.read_csv(dataset_path)
                elif dataset_path.suffix == '.json':
                    df = pd.read_json(dataset_path)
                else:
                    raise ValueError("Unsupported file format")
            except Exception as e:
                raise ValueError(f"Failed to read dataset: {str(e)}")

            textgen_config = TextGenerationConfig(
                n=1,
                temperature=kwargs.get('temperature', 0.5),
                model=kwargs.get('model', 'gpt-4'),
                use_cache=kwargs.get('use_cache', True)
            )
            lida = Manager(text_gen=llm("openai", api_key=self.openai_key))
            summary = await run_in_threadpool(
                lida.summarize,
                str(dataset_path),
                summary_method=kwargs.get('summary_method', 'llm'),
                textgen_config=textgen_config
            )

            if "fields" in summary:
                summary["fields"] = [
                    {
                        "column": field["column"],
                        **{k: str(v) if k == "samples" else v
                           for k, v in field["properties"].items()}
                    }
                    for field in summary["fields"]
                ]

            summary["file_name"] = str(dataset_path)
            return summary

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise

    async def generate_goals(self, summary: Dict, **kwargs) -> List[Dict]:
        """Generate visualization goals"""
        try:
            textgen_config = TextGenerationConfig(
                n=1,
                temperature=kwargs.get('temperature', 0.0),
                model=kwargs.get('model', 'gpt-4'),
                use_cache=kwargs.get('use_cache', True)
            )

            lida = Manager(text_gen=llm("openai", api_key=self.openai_key))
            goals = await run_in_threadpool(
                lida.goals,
                summary,
                n=kwargs.get('num_goals', 4),
                textgen_config=textgen_config
            )
            
            goal_dicts = [goal.__dict__ for goal in goals]

            if kwargs.get('user_goal'):
                new_goal = Goal(
                    question=kwargs['user_goal'],
                    visualization=kwargs['user_goal'],
                    rationale=""
                )
                goal_dicts.append(new_goal.__dict__)

            return goal_dicts
        except Exception as e:
            logger.error(f"Goal generation failed: {str(e)}")
            raise

    async def generate_visualizations(
        self,
        summary: Dict,
        goal: Dict,
        library: str = "plotly",
        num_visualizations: int = 4,
        model: str = "gpt-4",
        temperature: float = 0.0,
        use_cache: bool = True
    ) -> Dict:
        """Generate visualizations"""
        try:
            if library.lower() not in AVAILABLE_LIBRARIES:
                raise ValueError(f"Unsupported library. Must be one of {AVAILABLE_LIBRARIES}")

            textgen_config = TextGenerationConfig(
                n=num_visualizations,
                temperature=temperature,
                model=model,
                use_cache=use_cache
            )
            
            if isinstance(summary, dict):
                summary_obj = SimpleNamespace(**summary)
            else:
                summary_obj = summary

            goal_obj = Goal(**goal)

            lida = Manager(text_gen=llm("openai", api_key=self.openai_key))
            visualizations = await run_in_threadpool(
                lida.visualize,
                summary_obj,
                goal_obj,
                textgen_config,
                library.lower()  # Ensure lowercase
            )
            
            viz_dicts = []
            images = []
            for viz in visualizations:
                viz_dicts.append(viz.__dict__)
                images.append(viz.raster or None)

            return {
                "visualizations": viz_dicts,
                "images": images
            }

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            raise

# Initialize service
try:
    lida_service = LIDAService()
except Exception as e:
    logger.critical(f"Failed to initialize LIDA service: {str(e)}")
    sys.exit(1)

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise

# API Endpoints
@app.get("/available_options")
async def get_available_options():
    return {
        "available_models": AVAILABLE_OPENAI_MODELS,
        "available_libraries": AVAILABLE_LIBRARIES
    }

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        filename = await lida_service.upload_dataset(file)
        return {
            "status": "success",
            "file_path": filename,
            "message": "File uploaded successfully"
        }
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(500, detail=f"Upload failed: {str(e)}")

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    try:
        if not request.dataset_path:
            raise HTTPException(400, detail="Dataset path is required")
            
        return await lida_service.summarize_data(
            request.dataset_path,
            summary_method=request.summary_method,
            model=request.model,
            temperature=request.temperature,
            use_cache=request.use_cache
        )
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        logger.error(f"Summarize error: {str(e)}")
        raise HTTPException(500, detail=f"Summarization failed: {str(e)}")
    
@app.post("/generate_goals", response_model=List[GoalResponse])
async def generate_goals(
    summary: Dict, 
    num_goals: int = 4,
    model: str = "gpt-4", 
    temperature: float = 0.0, 
    use_cache: bool = True, 
    user_goal: Optional[str] = None
):
    try:
        if model not in AVAILABLE_OPENAI_MODELS:
            raise HTTPException(400, detail=f"Invalid model. Must be one of {AVAILABLE_OPENAI_MODELS}")
            
        return await lida_service.generate_goals(
            summary,
            num_goals=num_goals,
            model=model,
            temperature=temperature,
            use_cache=use_cache,
            user_goal=user_goal
        )
    except Exception as e:
        logger.error(f"Goal generation error: {str(e)}")
        raise HTTPException(500, detail=str(e))

@app.post("/generate_visualizations")
async def generate_visualizations(
    summary: Dict, 
    goal: Dict, 
    library: str = "plotly",
    num_visualizations: int = 4, 
    model: str = "gpt-4",
    temperature: float = 0.5, 
    use_cache: bool = True
):
    try:
        if model not in AVAILABLE_OPENAI_MODELS:
            raise HTTPException(400, detail=f"Invalid model. Must be one of {AVAILABLE_OPENAI_MODELS}")
        
        if library.lower() not in AVAILABLE_LIBRARIES:
            raise HTTPException(400, detail=f"Invalid library. Must be one of {AVAILABLE_LIBRARIES}")
            
        return await lida_service.generate_visualizations(
            summary,
            goal,
            library=library.lower(),
            num_visualizations=num_visualizations,
            model=model,
            temperature=temperature,
            use_cache=use_cache
        )
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        raise HTTPException(500, detail=str(e))
    
@app.post("/edit_visualization")
async def edit_visualization(request: EditVisualizationRequest):
    try:
        textgen_config = TextGenerationConfig(
            n=1,
            temperature=request.temperature,
            model=request.model,
            use_cache=request.use_cache
        )
        
        summary_obj = SimpleNamespace(**request.summary) if isinstance(request.summary, dict) else request.summary

        lida = Manager(text_gen=llm("openai", api_key=os.getenv("OPENAI_API_KEY")))
        edited_visualizations = await run_in_threadpool(
            lida.edit,
            code=request.code,
            summary=summary_obj,
            instructions=request.instructions,
            library=request.library,
            textgen_config=textgen_config
        )

        return {
            "visualizations": [viz.__dict__ for viz in edited_visualizations],
            "message": "Visualization edited successfully"
        }

    except Exception as e:
        logger.error(f"Visualization edit error: {str(e)}")
        raise HTTPException(500, detail=str(e))

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
