"""
OpinionBalancer State Management
KB-Free LangGraph implementation with local Ollama LLM
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Metrics(BaseModel):
    """Quantified metrics for opinion balance and quality assessment"""
    bias_probs: Dict[str, float] = Field(description="Stance probabilities (Left/Center/Right)")
    bias_target: Dict[str, float] = Field(description="Target stance distribution")
    bias_delta: float = Field(description="Deviation from target distribution")
    frame_distribution: Dict[str, float] = Field(description="Distribution of framing categories")
    frame_entropy: float = Field(description="Shannon entropy of frame diversity")
    flesch_kincaid: float = Field(description="Flesch-Kincaid grade level")
    dale_chall: float = Field(description="Dale-Chall readability score")
    coherence_score: float = Field(description="Mean cosine similarity between paragraphs")


class PassLog(BaseModel):
    """Log entry for each refinement pass"""
    pass_id: int
    draft: str
    critique: str
    metrics: Metrics
    timestamp: datetime = Field(default_factory=datetime.now)


class GraphState(BaseModel):
    """Complete state for the OpinionBalancer workflow"""
    # Input parameters
    topic: str = ""
    audience: str = "general US reader"
    length: int = 750
    
    # Constraints and targets
    constraints: Dict[str, Any] = Field(default_factory=lambda: {
        "grade_min": 10,
        "grade_max": 13,
        "frame_entropy_min": 0.6,
        "bias_delta_max": 0.05,
        "max_passes": 3
    })
    target_distribution: Dict[str, float] = Field(default_factory=lambda: {
        "Left": 0.5, 
        "Right": 0.5
    })
    
    # Working state
    draft: str = ""
    critique: str = ""
    metrics: Optional[Metrics] = None
    
    # Process tracking
    history: List[PassLog] = Field(default_factory=list)
    pass_count: int = 0
    converged: bool = False
    
    # Metadata
    run_id: str = ""
    start_time: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class TopicIntakeInput(BaseModel):
    """Input for topic intake normalization"""
    raw_topic: str
    audience: Optional[str] = None
    length: Optional[int] = None
    target_stance: Optional[str] = None  # e.g., "Left=0.5,Right=0.5"


class EditInstruction(BaseModel):
    """Structured edit instruction from critique synthesis"""
    edit_id: int
    paragraph_index: int
    sentence_index: Optional[int] = None
    instruction: str
    reason: str
