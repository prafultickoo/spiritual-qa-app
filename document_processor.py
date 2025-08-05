"""
Document processor using CrewAI framework for agentic document loading and chunking.
"""
import os
import sys
import yaml
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tasks.task_output import TaskOutput

# Import Langchain utilities and agent tools
from utils.langchain_utils import load_documents, chunk_documents, DocumentChunk
from utils.agent_tools import document_loader, document_chunker, chunk_verifier, chunk_saver
from utils.pdf_orientation_agent_tools import (
    pdf_orientation_analyzer,
    pdf_orientation_corrector,
    pdf_ocr_processor,
    batch_pdf_analyzer
)
from utils.vectorization_agent_tools import (
    vectorize_chunks,
    vectorize_from_json,
    similarity_search
)
from prompts.document_prompts import CHUNKING_AGENT_PROMPT, VERIFICATION_AGENT_PROMPT

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_agent_config(config_path: str) -> Dict:
    """
    Load agent configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        Dict: Agent configuration
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Failed to load agent config from {config_path}: {str(e)}")
        sys.exit(1)

def load_task_config(config_path: str) -> Dict:
    """
    Load task configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        Dict: Task configuration
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Failed to load task config from {config_path}: {str(e)}")
        sys.exit(1)

def create_chunking_agent(config: Dict) -> Agent:
    """
    Create document chunking agent.
    
    Args:
        config (Dict): Agent configuration
        
    Returns:
        Agent: CrewAI agent for document chunking
    """
    agent_config = config.get('chunking_agent', {})
    
    return Agent(
        role=agent_config.get('role'),
        goal=agent_config.get('goal'),
        backstory=agent_config.get('backstory'),
        verbose=agent_config.get('verbose', True),
        llm=agent_config.get('llm', {}).get('model'),
        tools=[document_loader, document_chunker]
    )

def create_verification_agent(config: Dict) -> Agent:
    """
    Create chunk verification agent.
    
    Args:
        config (Dict): Agent configuration
        
    Returns:
        Agent: CrewAI agent for chunk verification
    """
    agent_config = config.get('verification_agent', {})
    
    return Agent(
        role=agent_config.get('role'),
        goal=agent_config.get('goal'),
        backstory=agent_config.get('backstory'),
        verbose=agent_config.get('verbose', True),
        llm=agent_config.get('llm', {}).get('model'),
        tools=[chunk_verifier]
    )


def create_vectorization_agent(config: Dict) -> Agent:
    """
    Create document vectorization agent.
    
    Args:
        config (Dict): Agent configuration
        
    Returns:
        Agent: CrewAI agent for document vectorization
    """
    agent_config = config.get('vectorization_agent', {})
    
    return Agent(
        role=agent_config.get('role'),
        goal=agent_config.get('goal'),
        backstory=agent_config.get('backstory'),
        verbose=agent_config.get('verbose', True),
        llm=agent_config.get('llm', {}).get('model'),
        tools=[vectorize_chunks, vectorize_from_json, similarity_search]
    )

def create_chunking_task(config: Dict, document_dir: str) -> Task:
    """
    Create document chunking task.
    
    Args:
        config (Dict): Task configuration
        document_dir (str): Path to documents directory
        
    Returns:
        Task: CrewAI task for document chunking
    """
    task_config = config.get('chunking_task', {})
    
    return Task(
        description=task_config.get('description'),
        expected_output=task_config.get('expected_output'),
        agent=chunking_agent,
        async_execution=task_config.get('async_execution', False),
        context=f"{task_config.get('context')} Process documents from: {document_dir}"
    )

def create_verification_task(config: Dict) -> Task:
    """
    Create chunk verification task.
    
    Args:
        config (Dict): Task configuration
        
    Returns:
        Task: CrewAI task for chunk verification
    """
    task_config = config.get('verification_task', {})
    
    return Task(
        description=task_config.get('description'),
        expected_output=task_config.get('expected_output'),
        agent=verification_agent,
        async_execution=task_config.get('async_execution', False),
        output_file=task_config.get('output_file')
    )

def create_vectorization_task(config: Dict, output_dir: str, vector_dir: str) -> Task:
    """
    Create document vectorization task.
    
    Args:
        config (Dict): Task configuration
        output_dir (str): Directory containing processed chunks
        vector_dir (str): Directory to store vector database
        
    Returns:
        Task: CrewAI task for document vectorization
    """
    task_config = config.get('vectorization_task', {})
    
    # Get the path to processed chunks
    json_path = os.path.join(output_dir, 'processed_chunks.json')
    
    # Create task description with actual paths
    description = task_config.get('description', '')
    description = description.replace('{json_path}', json_path)
    description = description.replace('{vector_dir}', vector_dir)
    
    return Task(
        description=description,
        expected_output=task_config.get('expected_output'),
        agent=vectorization_agent,
        context=[f"JSON chunks path: {json_path}", f"Vector DB path: {vector_dir}"],
        async_execution=task_config.get('async_execution', False),
        output_file=task_config.get('output_file')
    )

# Note: Tool functions for agents are now imported from utils.agent_tools
# We no longer need to define them here as they are already available as CrewAI Tool objects

def save_chunks_to_json(chunks: List[DocumentChunk], output_path: str) -> None:
    """
    Save chunks to JSON file.
    
    Args:
        chunks (List[DocumentChunk]): Chunks to save
        output_path (str): Path to output file
    """
    try:
        # Convert Pydantic models to dictionaries for JSON serialization
        chunks_dict = [chunk.dict() for chunk in chunks]
        
        with open(output_path, 'w') as file:
            json.dump(chunks_dict, file, indent=2)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save chunks to {output_path}: {str(e)}")

def process_documents(document_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Process documents using CrewAI agents with Langchain.
    
    Args:
        document_dir (str): Path to documents directory
        output_dir (str): Path to output directory
        
    Returns:
        Dict[str, Any]: Processing results
    """
    # Load agent and task configurations
    agent_config = load_agent_config('agents/document_agents.yaml')
    task_config = load_task_config('tasks/document_tasks.yaml')
    
    # Create agents with Langchain tools
    global chunking_agent
    chunking_agent = create_chunking_agent(agent_config)
    if isinstance(result.raw_output, list) and result.raw_output and isinstance(result.raw_output[0], DocumentChunk):
        save_chunks_to_json(result.raw_output, output_path)
        chunks_count = len(result.raw_output)
    else:
        # If the output structure is different (e.g., verification results object),
        # extract the verified chunks if possible
        try:
            if isinstance(result.raw_output, dict) and 'verified_chunks' in result.raw_output:
                chunks = result.raw_output['verified_chunks']
                save_chunks_to_json(chunks, output_path)
                chunks_count = len(chunks)
            else:
                # Fallback: save raw output
                with open(output_path, 'w') as f:
                    json.dump(result.raw_output, f, indent=2)
                chunks_count = 0
        except Exception as e:
            logger.error(f"Error saving chunks: {str(e)}")
            chunks_count = 0
    
    return {
        "status": "success",
        "output_path": output_path,
        "chunks_processed": chunks_count
    }

if __name__ == "__main__":
    # Default paths
    document_dir = "Documents"
    output_dir = "Processed"
    
    # Process command line arguments
    if len(sys.argv) > 1:
        document_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Process documents
    result = process_documents(document_dir, output_dir)
    print(json.dumps(result, indent=2))
